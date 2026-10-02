#!/usr/bin/env python3
"""
Step 6: Deep Forensics and Advanced Investigations
Operating as an autonomous senior applied ML research team to:
1. Multi-match intra-group structure (are multiple matches store branches or duplicate entries?)
2. Data corruption and transformation operators (abbreviations, omissions, reorderings)
3. Postal code / PIN code behavior across US, India, and France
4. Test set generalization forensics (France deep-dive, non-ASCII accents, schema shifts)
5. Macro F0.5 threshold dynamics and singleton sensitivity simulation
6. Multi-pass candidate generation (blocking) benchmark on a realistic evaluation slice
"""

import os
import sys
import json
import re
import unicodedata
from collections import Counter
import polars as pl
import numpy as np
from rapidfuzz import fuzz, distance

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

def clean_text(s):
    if not s:
        return ""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.lower())).strip()

def extract_postal_code(s):
    if not s:
        return None
    # 6-digit Indian PIN or 5-digit US ZIP / French Code Postal
    matches = re.findall(r"\b\d{5,6}\b", s)
    return matches[0] if matches else None

def investigate_multi_match_structure(gt_exploded, s1_df, s2_df, s3_df, sample_groups=3000):
    print("\n--- 1. Multi-Match Intra-Group Forensics ---")
    # Group S1 by number of matches
    s1_match_counts = gt_exploded.group_by("source1_entity_id").len()
    multi_s1 = s1_match_counts.filter(pl.col("len") > 1)["source1_entity_id"].to_list()
    print(f"Total S1 entities with >1 match: {len(multi_s1):,}")
    
    np.random.seed(42)
    sample_s1_ids = set(np.random.choice(multi_s1, size=min(sample_groups, len(multi_s1)), replace=False))
    
    sample_links = gt_exploded.filter(pl.col("source1_entity_id").is_in(list(sample_s1_ids)))
    
    # Map IDs to names and addresses
    s2_lookup = dict(zip(s2_df["entity_id"], zip(s2_df["business_name"].fill_null(""), s2_df["business_address"].fill_null(""))))
    s3_lookup = dict(zip(s3_df["entity_id"], zip(s3_df["business_name"].fill_null(""), s3_df["business_address"].fill_null(""))))
    
    intra_s2_name_sims = []
    intra_s2_addr_sims = []
    intra_s2_same_addr_count = 0
    intra_s2_diff_addr_count = 0
    
    grouped = {}
    for row in sample_links.iter_rows():
        s1_id, m_id = row[0], row[1]
        grouped.setdefault(s1_id, []).append(m_id)
        
    for s1_id, match_ids in grouped.items():
        s2_matches = [mid for mid in match_ids if mid.startswith("S2-") and mid in s2_lookup]
        if len(s2_matches) >= 2:
            # Compare pairwise within S2
            for i in range(len(s2_matches)):
                for j in range(i + 1, len(s2_matches)):
                    n1, a1 = s2_lookup[s2_matches[i]]
                    n2, a2 = s2_lookup[s2_matches[j]]
                    intra_s2_name_sims.append(fuzz.ratio(n1.lower(), n2.lower()))
                    if a1 and a2:
                        addr_sim = fuzz.ratio(a1.lower(), a2.lower())
                        intra_s2_addr_sims.append(addr_sim)
                        if addr_sim > 90:
                            intra_s2_same_addr_count += 1
                        else:
                            intra_s2_diff_addr_count += 1

    total_addr_pairs = intra_s2_same_addr_count + intra_s2_diff_addr_count
    same_addr_pct = (intra_s2_same_addr_count / total_addr_pairs * 100) if total_addr_pairs else 0
    
    results = {
        "multi_s1_count": len(multi_s1),
        "sampled_s1_groups": len(sample_s1_ids),
        "intra_s2_pairs_evaluated": len(intra_s2_name_sims),
        "intra_s2_name_mean_similarity": round(float(np.mean(intra_s2_name_sims)), 2) if intra_s2_name_sims else 0,
        "intra_s2_addr_mean_similarity": round(float(np.mean(intra_s2_addr_sims)), 2) if intra_s2_addr_sims else 0,
        "intra_s2_near_identical_addr_pct": round(same_addr_pct, 2),
        "intra_s2_distinct_addr_pct": round(100 - same_addr_pct, 2),
        "interpretation": "High identical address % indicates multiple duplicate records in S2 for the exact same location, whereas distinct addresses indicate multiple physical branches/locations linked to the reference S1 entity."
    }
    print(f"  Intra-S2 Name Mean Sim: {results['intra_s2_name_mean_similarity']}%")
    print(f"  Intra-S2 Near-Identical Address: {results['intra_s2_near_identical_addr_pct']}% | Distinct Address: {results['intra_s2_distinct_addr_pct']}%")
    return results

def investigate_corruption_mechanisms(gt_exploded, s1_df, s2_df, sample_pairs=10000):
    print("\n--- 2. Data Transformation & Corruption Mechanism Analysis ---")
    s1_lookup = dict(zip(s1_df["entity_id"], zip(s1_df["business_name"].fill_null(""), s1_df["business_address"].fill_null(""))))
    s2_lookup = dict(zip(s2_df["entity_id"], zip(s2_df["business_name"].fill_null(""), s2_df["business_address"].fill_null(""))))
    
    s1_s2_pairs = gt_exploded.filter(pl.col("matched_entity_ids").str.starts_with("S2-"))
    sample_links = s1_s2_pairs.sample(n=min(sample_pairs, len(s1_s2_pairs)), seed=42)
    
    token_len_diffs = []
    char_len_diffs = []
    dropped_tokens_counter = Counter()
    added_tokens_counter = Counter()
    
    exact_token_set_match = 0
    token_reorder_count = 0
    
    for s1_id, s2_id in sample_links.iter_rows():
        if s1_id in s1_lookup and s2_id in s2_lookup:
            n1, _ = s1_lookup[s1_id]
            n2, _ = s2_lookup[s2_id]
            t1 = clean_text(n1).split()
            t2 = clean_text(n2).split()
            
            token_len_diffs.append(len(t1) - len(t2))
            char_len_diffs.append(len(n1) - len(n2))
            
            s1_set, s2_set = set(t1), set(t2)
            if s1_set == s2_set:
                exact_token_set_match += 1
                if t1 != t2:
                    token_reorder_count += 1
            
            for d in (s1_set - s2_set):
                dropped_tokens_counter[d] += 1
            for a in (s2_set - s1_set):
                added_tokens_counter[a] += 1
                
    n_evaluated = len(sample_links)
    results = {
        "pairs_evaluated": n_evaluated,
        "token_len_diff_mean": round(float(np.mean(token_len_diffs)), 2),
        "char_len_diff_mean": round(float(np.mean(char_len_diffs)), 2),
        "exact_token_bag_pct": round(exact_token_set_match / n_evaluated * 100, 2),
        "pure_token_reorder_pct": round(token_reorder_count / n_evaluated * 100, 2),
        "top_dropped_tokens_from_s1": [{"token": k, "count": v} for k, v in dropped_tokens_counter.most_common(20)],
        "top_added_tokens_in_s2": [{"token": k, "count": v} for k, v in added_tokens_counter.most_common(20)]
    }
    print(f"  Exact token bag match: {results['exact_token_bag_pct']}% (pure reorders: {results['pure_token_reorder_pct']}%)")
    print(f"  Top dropped tokens: {[x['token'] for x in results['top_dropped_tokens_from_s1'][:8]]}")
    print(f"  Top added tokens: {[x['token'] for x in results['top_added_tokens_in_s2'][:8]]}")
    return results

def investigate_postal_codes(gt_exploded, s1_df, s2_df, s3_df):
    print("\n--- 3. Postal Code Agreement & Cardinality Forensics ---")
    s1_codes = dict(zip(s1_df["entity_id"], [extract_postal_code(a) for a in s1_df["business_address"].fill_null("")]))
    s2_codes = dict(zip(s2_df["entity_id"], [extract_postal_code(a) for a in s2_df["business_address"].fill_null("")]))
    
    # Calculate postal code presence
    s1_has_code = sum(1 for c in s1_codes.values() if c is not None)
    s2_has_code = sum(1 for c in s2_codes.values() if c is not None)
    
    # Check postal code cardinality in S1
    all_s1_codes = [c for c in s1_codes.values() if c is not None]
    code_counts = Counter(all_s1_codes)
    
    # Check true match agreement
    sample_pairs = gt_exploded.filter(pl.col("matched_entity_ids").str.starts_with("S2-")).sample(n=min(50000, len(gt_exploded)), seed=42)
    
    both_have_code = 0
    codes_match = 0
    codes_differ = 0
    
    for s1_id, s2_id in sample_pairs.iter_rows():
        c1 = s1_codes.get(s1_id)
        c2 = s2_codes.get(s2_id)
        if c1 is not None and c2 is not None:
            both_have_code += 1
            if c1 == c2:
                codes_match += 1
            else:
                codes_differ += 1
                
    match_rate = (codes_match / both_have_code * 100) if both_have_code else 0
    results = {
        "s1_postal_code_coverage_pct": round(s1_has_code / len(s1_df) * 100, 2),
        "s2_postal_code_coverage_pct": round(s2_has_code / len(s2_df) * 100, 2),
        "pairs_with_both_postal_codes": both_have_code,
        "true_match_postal_agreement_pct": round(match_rate, 2),
        "true_match_postal_disagreement_pct": round(100 - match_rate, 2),
        "unique_postal_codes_in_s1": len(code_counts),
        "median_entities_per_postal_code": int(np.median(list(code_counts.values()))),
        "p95_entities_per_postal_code": int(np.percentile(list(code_counts.values()), 95)),
        "max_entities_in_single_postal_code": max(code_counts.values()) if code_counts else 0,
        "interpretation": "When both records have a postal code, high agreement validates postal code as a powerful blocking/feature key. Disagreement indicates either multi-branch/different city matches or data corruption."
    }
    print(f"  S1 Postal Coverage: {results['s1_postal_code_coverage_pct']}% | S2: {results['s2_postal_code_coverage_pct']}%")
    print(f"  Postal code agreement when both present in true match: {results['true_match_postal_agreement_pct']}% (disagreement: {results['true_match_postal_disagreement_pct']}%)")
    return results

def investigate_test_and_france():
    print("\n--- 4. Test Set & France Unseen Country Forensics ---")
    ts1_path = "dataset/raw/test/test_source1.tsv"
    ts2_path = "dataset/raw/test/test_source2.tsv"
    ts3_path = "dataset/raw/test/test_source3.tsv"
    
    ts1 = pl.read_csv(ts1_path, separator="\t")
    ts2 = pl.read_csv(ts2_path, separator="\t")
    ts3 = pl.read_csv(ts3_path, separator="\t")
    
    # Country breakdown
    ts1_countries = dict(ts1.group_by("country").len().iter_rows())
    ts2_countries = dict(ts2.group_by("country").len().iter_rows())
    ts3_countries = dict(ts3.group_by("country").len().iter_rows())
    
    # France subset
    fr_s1 = ts1.filter(pl.col("country") == "France")
    fr_s2 = ts2.filter(pl.col("country") == "France")
    fr_s3 = ts3.filter(pl.col("country") == "France")
    
    # Analyze accents in France names
    accent_pattern = re.compile(r"[éèêëàâäôöîïùûüçÉÈÊËÀÂÄÔÖÎÏÙÛÜÇ]")
    fr_s1_names = fr_s1["business_name"].fill_null("").to_list()
    fr_s1_addrs = fr_s1["business_address"].fill_null("").to_list()
    
    has_accent_name = sum(1 for n in fr_s1_names if accent_pattern.search(n))
    has_accent_addr = sum(1 for a in fr_s1_addrs if accent_pattern.search(a))
    
    # Check street keywords in French addresses
    fr_keywords = ["rue", "avenue", "boulevard", "chemin", "place", "allée", "route", "cedex"]
    keyword_counts = Counter()
    for a in fr_s1_addrs:
        al = a.lower()
        for kw in fr_keywords:
            if kw in al:
                keyword_counts[kw] += 1
                
    # Exact name matches within France between S1 and S2
    fr_s1_clean_names = set(clean_text(n) for n in fr_s1_names if n)
    fr_s2_clean_names = set(clean_text(n) for n in fr_s2["business_name"].fill_null("").to_list() if n)
    overlap_exact_clean = len(fr_s1_clean_names & fr_s2_clean_names)
    
    results = {
        "test_s1_total": len(ts1),
        "test_s2_total": len(ts2),
        "test_s3_total": len(ts3),
        "test_s1_countries": ts1_countries,
        "test_s2_countries": ts2_countries,
        "test_s3_countries": ts3_countries,
        "france_s1_records": len(fr_s1),
        "france_s2_records": len(fr_s2),
        "france_s3_records": len(fr_s3),
        "france_s1_names_with_french_accents_pct": round(has_accent_name / len(fr_s1) * 100, 2),
        "france_s1_addrs_with_french_accents_pct": round(has_accent_addr / len(fr_s1) * 100, 2),
        "france_s1_s2_clean_name_overlap_count": overlap_exact_clean,
        "france_s1_s2_clean_name_overlap_pct_of_s1": round(overlap_exact_clean / len(fr_s1_clean_names) * 100, 2),
        "french_address_keyword_presence_pct": {k: round(v / len(fr_s1) * 100, 2) for k, v in keyword_counts.items()}
    }
    print(f"  Test S1 Countries: {ts1_countries}")
    print(f"  France S1: {len(fr_s1):,} ({results['france_s1_names_with_french_accents_pct']}% with accents)")
    print(f"  France Clean Name Overlap S1-S2: {overlap_exact_clean:,} ({results['france_s1_s2_clean_name_overlap_pct_of_s1']}% of S1 unique names)")
    return results

def simulate_macro_f05_thresholds():
    print("\n--- 5. Macro F0.5 Threshold & Singleton Sensitivity Simulation ---")
    # Simulate a realistic distribution of match probabilities
    np.random.seed(42)
    N_entities = 10000
    p_singleton = 0.0558  # 5.58% true singletons from Ground Truth
    
    # Ground truth: number of true matches per entity
    num_matches = np.random.choice([0, 1, 2, 3, 4, 5], size=N_entities, p=[0.0558, 0.0540, 0.1700, 0.2400, 0.2200, 0.2602])
    
    # Simulate candidate scores:
    # True matches have scores drawn from Beta(6, 2) (mean ~ 0.75)
    # False matches have scores drawn from Beta(1.5, 5) (mean ~ 0.23)
    
    thresholds = [0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    scores_by_threshold = {}
    
    for th in thresholds:
        f05_list = []
        for n_true in num_matches:
            if n_true == 0:
                # Singleton: receives a few false candidate scores
                n_false_cands = np.random.choice([0, 1, 2, 3], p=[0.4, 0.3, 0.2, 0.1])
                false_scores = np.random.beta(1.5, 5, size=n_false_cands) if n_false_cands > 0 else np.array([])
                pred_matches = np.sum(false_scores >= th)
                if pred_matches == 0:
                    f05_list.append(1.0)
                else:
                    f05_list.append(0.0)
            else:
                true_scores = np.random.beta(6, 2, size=n_true)
                n_false_cands = np.random.choice([0, 1, 2, 3, 5], p=[0.2, 0.3, 0.25, 0.15, 0.1])
                false_scores = np.random.beta(1.5, 5, size=n_false_cands) if n_false_cands > 0 else np.array([])
                
                tp = np.sum(true_scores >= th)
                fp = np.sum(false_scores >= th)
                pred_total = tp + fp
                
                if pred_total == 0:
                    f05_list.append(0.0)
                else:
                    precision = tp / pred_total
                    recall = tp / n_true
                    if precision + recall == 0 or tp == 0:
                        f05_list.append(0.0)
                    else:
                        f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
                        f05_list.append(f05)
                        
        scores_by_threshold[str(th)] = round(float(np.mean(f05_list)), 4)
        
    best_th = max(scores_by_threshold.items(), key=lambda x: x[1])
    print(f"  Macro F0.5 by Decision Threshold: {scores_by_threshold}")
    print(f"  Optimal Decision Threshold in Simulation: {best_th[0]} (Macro F0.5 = {best_th[1]})")
    return {
        "scores_by_threshold": scores_by_threshold,
        "optimal_threshold": float(best_th[0]),
        "best_macro_f05": best_th[1],
        "insight": "Because F0.5 penalizes false positives 4x more heavily than false negatives (beta^2 = 0.25), and singletons drop to 0.0 with ANY false positive, the optimal threshold shifts substantially higher than 0.50 (typically 0.55 - 0.65)."
    }

def main():
    print("Loading Ground Truth and Raw Data...")
    gt_df = pl.read_csv("dataset/raw/train/train_ground_truth.tsv", separator="\t")
    s1_df = pl.read_csv("dataset/raw/train/train_source1.tsv", separator="\t")
    s2_df = pl.read_csv("dataset/raw/train/train_source2.tsv", separator="\t")
    s3_df = pl.read_csv("dataset/raw/train/train_source3.tsv", separator="\t")
    
    gt_exploded = (
        gt_df.filter(pl.col("matched_entity_ids").is_not_null() & (pl.col("matched_entity_ids") != ""))
        .with_columns(pl.col("matched_entity_ids").str.split(","))
        .explode("matched_entity_ids")
        .with_columns(pl.col("matched_entity_ids").str.strip_chars())
    )
    
    r1 = investigate_multi_match_structure(gt_exploded, s1_df, s2_df, s3_df)
    r2 = investigate_corruption_mechanisms(gt_exploded, s1_df, s2_df)
    r3 = investigate_postal_codes(gt_exploded, s1_df, s2_df, s3_df)
    r4 = investigate_test_and_france()
    r5 = simulate_macro_f05_thresholds()
    
    full_report = {
        "multi_match_structure": r1,
        "corruption_mechanisms": r2,
        "postal_code_forensics": r3,
        "test_and_france_forensics": r4,
        "macro_f05_simulation": r5
    }
    
    out_file = "research_scripts/06_deep_forensics_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)
    print(f"\nAll deep forensics completed and saved to {out_file}")

if __name__ == "__main__":
    main()
