"""
scratch/analyze_retrieval_gaps.py
===================================
Analyze which true positive pairs are NOT being retrieved by the current
10-channel pipeline. Categorize the missed links and identify what additional
retrieval strategies could recover them.

Output:
  - experiments/phase2/retrieval_gap_analysis.json
  - Console report with actionable insights
"""
import sys
import time
import json
import re
from pathlib import Path
from collections import Counter, defaultdict

import polars as pl
import pandas as pd
import numpy as np

ROOT = Path("z:/Amazon ML")
sys.path.append(str(ROOT / "scratch"))
sys.path.append(str(ROOT / "src"))

CACHE_DIR = ROOT / "experiments" / "cache"
PHASE2_DIR = ROOT / "experiments" / "phase2"
OUT_JSON = PHASE2_DIR / "retrieval_gap_analysis.json"

def flush(msg):
    print(msg, flush=True)

# ── Load cached normalized data ─────────────────────────────────────────────
def load_data():
    flush("Loading normalized queries (train)...")
    q_norm = pl.read_parquet(CACHE_DIR / "queries_norm_v2.parquet")
    flush(f"  Queries: {len(q_norm):,}")
    
    flush("Loading normalized targets...")
    t_norm = pl.read_parquet(CACHE_DIR / "targets_norm_v2.parquet")
    flush(f"  Targets: {len(t_norm):,}")
    
    flush("Loading ground truth labels...")
    train_s1 = pl.read_csv(ROOT / "dataset/raw/train/train_source1.tsv",
                           separator="\t", quote_char=None)
    flush(f"  Train source1 rows: {len(train_s1):,}")
    
    flush("Loading 10-channel candidates (val set)...")
    cands = pl.read_parquet(PHASE2_DIR / "phase2c_10ch_val_candidates.parquet")
    flush(f"  Candidate pairs: {len(cands):,}")
    
    return q_norm, t_norm, train_s1, cands

# ── Parse ground truth links ─────────────────────────────────────────────────
def parse_gt_links(train_s1):
    """Extract (query_id, target_id) positive pairs from ground truth."""
    flush("\nParsing ground truth links...")
    gt_pairs = set()
    # Ground truth columns: entity_id, s2_entity_ids, s3_entity_ids (or similar)
    flush(f"  Columns: {train_s1.columns}")
    
    # Identify link columns
    link_cols = [c for c in train_s1.columns if "entity_id" in c.lower() and c != "entity_id"]
    flush(f"  Link columns found: {link_cols}")
    
    for row in train_s1.iter_rows(named=True):
        qid = str(row["entity_id"])
        for lc in link_cols:
            val = row.get(lc, "")
            if val and str(val).strip():
                for tid in str(val).split(","):
                    tid = tid.strip()
                    if tid:
                        gt_pairs.add((qid, tid))
    
    flush(f"  Total GT positive pairs: {len(gt_pairs):,}")
    return gt_pairs

# ── Find missed pairs ─────────────────────────────────────────────────────────
def find_missed_pairs(gt_pairs, cands):
    flush("\nBuilding retrieved pair set...")
    retrieved = set(zip(
        cands["entity_id"].cast(str).to_list(),
        cands["target_id"].cast(str).to_list()
    ))
    flush(f"  Retrieved pairs: {len(retrieved):,}")
    
    missed = gt_pairs - retrieved
    flush(f"  Missed GT pairs: {len(missed):,} / {len(gt_pairs):,} ({100*len(missed)/len(gt_pairs):.1f}%)")
    
    return missed, retrieved

# ── Name similarity analysis on missed pairs ─────────────────────────────────
def analyze_missed_pairs(missed, q_norm, t_norm):
    flush("\nBuilding lookup dicts for query/target fields...")
    
    q_dict = {str(r["entity_id"]): r for r in q_norm.select([
        "entity_id", "clean_name", "stripped_name", "sorted_name",
        "spaceless_name", "std_address", "house_no", "postal_code", "country"
    ]).iter_rows(named=True)}
    
    t_dict = {str(r["entity_id"]): r for r in t_norm.select([
        "entity_id", "clean_name", "stripped_name", "sorted_name",
        "spaceless_name", "std_address", "house_no", "postal_code", "country"
    ]).iter_rows(named=True)}
    
    flush(f"  Q dict size: {len(q_dict):,} | T dict size: {len(t_dict):,}")
    
    # Sample up to 10,000 missed pairs for analysis
    missed_list = list(missed)
    sample_size = min(10000, len(missed_list))
    np.random.seed(42)
    sample_idx = np.random.choice(len(missed_list), sample_size, replace=False)
    sample = [missed_list[i] for i in sample_idx]
    
    flush(f"\nAnalyzing {sample_size:,} sampled missed pairs...")
    
    from rapidfuzz import fuzz as rfuzz
    from rapidfuzz.distance import JaroWinkler
    
    categories = Counter()
    name_jw_scores = []
    addr_match_scores = []
    examples = defaultdict(list)
    
    for qid, tid in sample:
        q = q_dict.get(qid, {})
        t = t_dict.get(tid, {})
        
        qn = (q.get("clean_name") or "").lower()
        tn = (t.get("clean_name") or "").lower()
        qns = (q.get("sorted_name") or "").lower()
        tns = (t.get("sorted_name") or "").lower()
        qa = (q.get("std_address") or "").lower()
        ta = (t.get("std_address") or "").lower()
        qpin = (q.get("postal_code") or "")
        tpin = (t.get("postal_code") or "")
        
        jw = JaroWinkler.similarity(qn, tn)
        tok_sort = rfuzz.token_sort_ratio(qn, tn) / 100.0
        tok_set = rfuzz.token_set_ratio(qn, tn) / 100.0
        partial = rfuzz.partial_ratio(qn, tn) / 100.0
        
        name_jw_scores.append(jw)
        
        # Categorize
        if qn == tn:
            cat = "exact_name_match"
        elif qns == tns:
            cat = "sorted_name_match"
        elif tok_sort >= 0.95:
            cat = "high_token_sort_95+"
        elif tok_sort >= 0.85:
            cat = "high_token_sort_85-95"
        elif jw >= 0.92:
            cat = "high_jw_match_92+"
        elif tok_set >= 0.90:
            cat = "high_token_set_90+"
        elif partial >= 0.90:
            cat = "high_partial_90+"
        elif jw >= 0.80:
            cat = "medium_jw_80-92"
        elif tok_sort >= 0.70:
            cat = "medium_token_sort_70-85"
        else:
            cat = "low_similarity_needs_semantic"
        
        categories[cat] += 1
        if len(examples[cat]) < 5:
            examples[cat].append({
                "qid": qid, "tid": tid,
                "qn": qn, "tn": tn,
                "jw": round(jw, 3), "tok_sort": round(tok_sort, 3),
                "tok_set": round(tok_set, 3), "partial": round(partial, 3),
                "qa": qa[:60], "ta": ta[:60]
            })
    
    # Summary stats
    flush("\n" + "="*60)
    flush("MISSED PAIR ANALYSIS (sampled {})".format(sample_size))
    flush("="*60)
    flush(f"Name JW score distribution:")
    arr = np.array(name_jw_scores)
    flush(f"  mean={arr.mean():.3f} | median={np.median(arr):.3f} | p25={np.percentile(arr,25):.3f} | p75={np.percentile(arr,75):.3f}")
    
    flush("\nCategories:")
    for cat, cnt in categories.most_common():
        pct = 100 * cnt / sample_size
        flush(f"  {cat:40s}: {cnt:5d} ({pct:.1f}%)")
    
    return categories, examples, name_jw_scores

# ── Actionable recommendations ────────────────────────────────────────────────
def generate_recommendations(categories, total_missed):
    flush("\n" + "="*60)
    flush("ACTIONABLE RETRIEVAL IMPROVEMENTS")
    flush("="*60)
    
    # High-similarity misses → retrieval threshold issue
    high_sim = (categories.get("exact_name_match", 0) + 
                categories.get("sorted_name_match", 0) +
                categories.get("high_token_sort_95+", 0) +
                categories.get("high_jw_match_92+", 0))
    
    # Medium → add more fuzzy channels
    medium_sim = (categories.get("high_token_sort_85-95", 0) +
                  categories.get("high_token_set_90+", 0) +
                  categories.get("high_partial_90+", 0))
    
    # Low → need semantic search
    low_sim = (categories.get("medium_jw_80-92", 0) +
               categories.get("medium_token_sort_70-85", 0) +
               categories.get("low_similarity_needs_semantic", 0))
    
    sample_size = sum(categories.values())
    
    flush(f"\n1. HIGH-SIMILARITY misses ({high_sim}/{sample_size} = {100*high_sim/sample_size:.1f}%):")
    flush("   → These pairs SHOULD be retrieved. Check retrieval channel coverage.")
    flush("   → Fix: expand TF-IDF candidate budget (top-100 instead of top-40)")
    flush("   → Fix: add BM25 retrieval channel")
    flush("   → Fix: add char-trigram retrieval channel")
    
    flush(f"\n2. MEDIUM-SIMILARITY misses ({medium_sim}/{sample_size} = {100*medium_sim/sample_size:.1f}%):")
    flush("   → Fuzzy but valid matches. Phonetic/transliteration differences.")
    flush("   → Fix: Soundex/Metaphone phonetic channel")
    flush("   → Fix: character-level edit distance channel (Levenshtein <= 3)")
    flush("   → Fix: BM25 with 3-gram tokenization")
    
    flush(f"\n3. LOW-SIMILARITY / SEMANTIC misses ({low_sim}/{sample_size} = {100*low_sim/sample_size:.1f}%):")
    flush("   → Abbreviations, language variations, semantic duplicates.")
    flush("   → Fix: sentence-transformers (multilingual-MiniLM-L6-v2)")
    flush("   → Fix: FAISS ANN index for embedding-based retrieval")
    
    recs = {
        "high_sim_pct": round(100 * high_sim / sample_size, 1),
        "medium_sim_pct": round(100 * medium_sim / sample_size, 1),
        "low_sim_pct": round(100 * low_sim / sample_size, 1),
        "total_missed": total_missed,
    }
    return recs

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    flush("=" * 60)
    flush("RETRIEVAL GAP ANALYSIS")
    flush("=" * 60)
    
    q_norm, t_norm, train_s1, cands = load_data()
    gt_pairs = parse_gt_links(train_s1)
    missed, retrieved = find_missed_pairs(gt_pairs, cands)
    categories, examples, name_jw_scores = analyze_missed_pairs(missed, q_norm, t_norm)
    recs = generate_recommendations(categories, len(missed))
    
    # Save results
    result = {
        "total_gt_pairs": len(gt_pairs),
        "total_retrieved": len(retrieved),
        "total_missed": len(missed),
        "retrieval_recall_pct": round(100 * (len(gt_pairs) - len(missed)) / len(gt_pairs), 2),
        "categories": dict(categories),
        "recommendations": recs,
        "examples_per_category": {k: v for k, v in examples.items()}
    }
    
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    
    flush(f"\nSaved gap analysis to: {OUT_JSON}")
    flush(f"\nFINAL: Retrieval Recall = {result['retrieval_recall_pct']:.2f}%")
    flush(f"FINAL: Total missed links = {result['total_missed']:,} / {result['total_gt_pairs']:,}")

if __name__ == "__main__":
    main()
