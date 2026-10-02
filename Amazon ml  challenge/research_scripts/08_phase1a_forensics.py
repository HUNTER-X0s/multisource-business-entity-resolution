"""
research_scripts/08_phase1a_forensics.py
=========================================
Deep forensic analysis supporting Phase 1A Data Foundation Audit & Preprocessing.
Computes empirical data-quality, schema contracts, missingness, Unicode/accent distributions,
collision statistics, legal suffix distributions, and difficult record strata.
Outputs: research_scripts/08_phase1a_forensics_results.json
"""

import json
import os
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
import polars as pl

OUTPUT_PATH = Path("research_scripts/08_phase1a_forensics_results.json")
ROOT = Path("dataset/raw")

INDIA_PIN_RE = re.compile(r"\b([1-9]\d{5})\b")
US_ZIP_RE = re.compile(r"\b(\d{5})(?:-\d{4})?\b")
FRANCE_POSTAL_RE = re.compile(r"\b(?:0[1-9]|[1-8]\d|9[0-8])\d{3}\b")

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
NON_ASCII_RE = re.compile(r"[^\x00-\x7F]")

LEGAL_SUFFIXES = [
    "private limited", "pvt ltd", "pvt limited", "private ltd",
    "limited", "ltd", "pvt", "llc", "l.l.c.", "inc", "incorporated",
    "corporation", "corp", "llp", "l.l.p.", "co", "company",
    "gmbh", "sarl", "sa", "sas", "plc"
]
LEGAL_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(s) for s in sorted(LEGAL_SUFFIXES, key=len, reverse=True)) + r")\b",
    re.IGNORECASE
)
WHITESPACE_RE = re.compile(r"\s+")
PUNCT_RE = re.compile(r"[^\w\s]")

def clean_text_fast(text: str) -> str:
    if not text:
        return ""
    s = text.lower().replace("&", " and ")
    s = PUNCT_RE.sub(" ", s)
    return WHITESPACE_RE.sub(" ", s).strip()

def strip_suffix_fast(text: str) -> str:
    if not text:
        return ""
    s = LEGAL_RE.sub(" ", text)
    return WHITESPACE_RE.sub(" ", s).strip()

def sort_tokens_fast(text: str) -> str:
    if not text:
        return ""
    tokens = text.split()
    tokens.sort()
    return " ".join(tokens)

def fold_accents_fast(text: str) -> str:
    if not text:
        return ""
    nfkd = unicodedata.normalize("NFKD", text)
    return nfkd.encode("ascii", "ignore").decode("utf-8")

def analyze_table(split: str, source: str) -> dict:
    tsv_path = ROOT / split / f"{split}_source{source}.tsv"
    print(f"Reading {tsv_path} ...", flush=True)
    df = pl.read_csv(tsv_path, separator="\t", truncate_ragged_lines=True)
    n_rows = len(df)
    
    # 1. Null / missingness
    null_name = df["business_name"].is_null().sum()
    null_addr = df["business_address"].is_null().sum()
    null_country = df["country"].is_null().sum()
    
    # 2. Country counts
    country_counts = df["country"].value_counts().to_dicts()
    
    # 3. ID uniqueness & prefix
    unique_ids = df["entity_id"].n_unique()
    expected_prefix = f"S{source}-"
    valid_prefix_count = df.filter(pl.col("entity_id").str.starts_with(expected_prefix)).height
    
    # 4. Length distributions for business_name
    lens = df["business_name"].fill_null("").str.len_chars()
    name_len_min = int(lens.min() or 0)
    name_len_max = int(lens.max() or 0)
    name_len_median = float(lens.median() or 0.0)
    name_len_p95 = float(lens.quantile(0.95) or 0.0)
    short_names_count = (lens <= 3).sum()
    
    # 5. Address length distributions
    addr_lens = df["business_address"].fill_null("").str.len_chars()
    addr_len_median = float(addr_lens.median() or 0.0)
    addr_len_p95 = float(addr_lens.quantile(0.95) or 0.0)
    
    # 6. Duplicate entities and branch indicators
    # Exact record duplicates (name, address, country)
    exact_rec_unique = df.select(["business_name", "business_address", "country"]).n_unique()
    exact_duplicates = n_rows - exact_rec_unique
    
    # Same name with different address (potential branches/franchises)
    name_unique = df["business_name"].fill_null("").n_unique()
    
    # 7. Non-ASCII & Devanagari sampling (sample 100k for speed)
    sample_df = df.sample(n=min(100_000, n_rows), seed=42)
    sample_names = sample_df["business_name"].fill_null("").to_list()
    sample_addrs = sample_df["business_address"].fill_null("").to_list()
    
    non_ascii_names = sum(1 for s in sample_names if NON_ASCII_RE.search(s))
    devanagari_names = sum(1 for s in sample_names if DEVANAGARI_RE.search(s))
    non_ascii_addrs = sum(1 for s in sample_addrs if NON_ASCII_RE.search(s))
    devanagari_addrs = sum(1 for s in sample_addrs if DEVANAGARI_RE.search(s))
    
    return {
        "file": f"{split}_source{source}.tsv",
        "rows": n_rows,
        "null_name": int(null_name),
        "null_addr": int(null_addr),
        "null_addr_pct": round(float(null_addr) / n_rows * 100, 2),
        "null_country": int(null_country),
        "unique_ids": int(unique_ids),
        "valid_prefix_pct": round(float(valid_prefix_count) / n_rows * 100, 2),
        "country_counts": country_counts,
        "name_len_min": name_len_min,
        "name_len_max": name_len_max,
        "name_len_median": name_len_median,
        "name_len_p95": name_len_p95,
        "short_names_count": int(short_names_count),
        "addr_len_median": addr_len_median,
        "addr_len_p95": addr_len_p95,
        "exact_duplicates": int(exact_duplicates),
        "unique_raw_names": int(name_unique),
        "sample_size": len(sample_names),
        "sample_non_ascii_name_pct": round(non_ascii_names / len(sample_names) * 100, 3),
        "sample_devanagari_name_pct": round(devanagari_names / len(sample_names) * 100, 3),
        "sample_non_ascii_addr_pct": round(non_ascii_addrs / len(sample_names) * 100, 3),
        "sample_devanagari_addr_pct": round(devanagari_addrs / len(sample_names) * 100, 3),
    }

def analyze_normalization_collisions(split: str = "train", source: str = "1", n_sample: int = 500_000) -> dict:
    tsv_path = ROOT / split / f"{split}_source{source}.tsv"
    print(f"Sampling {n_sample} from {tsv_path} for collision science ...", flush=True)
    df = pl.read_csv(tsv_path, separator="\t", truncate_ragged_lines=True)
    if len(df) > n_sample:
        df = df.sample(n=n_sample, seed=42)
    
    raw_names = df["business_name"].fill_null("").to_list()
    total_records = len(raw_names)
    
    # Step 1: Raw
    raw_unique = len(set(raw_names))
    raw_counts = Counter(raw_names)
    raw_max_cluster = max(raw_counts.values()) if raw_counts else 0
    
    # Step 2: Cleaned
    clean_names = [clean_text_fast(x) for x in raw_names]
    clean_unique = len(set(clean_names))
    clean_counts = Counter(clean_names)
    clean_max_cluster = max(clean_counts.values()) if clean_counts else 0
    
    # Step 3: Suffix stripped
    stripped_names = [strip_suffix_fast(x) for x in clean_names]
    stripped_unique = len(set(stripped_names))
    stripped_counts = Counter(stripped_names)
    stripped_max_cluster = max(stripped_counts.values()) if stripped_counts else 0
    
    # Step 4: Token sorted
    sorted_names = [sort_tokens_fast(x if x else c) for x, c in zip(stripped_names, clean_names)]
    sorted_unique = len(set(sorted_names))
    sorted_counts = Counter(sorted_names)
    sorted_max_cluster = max(sorted_counts.values()) if sorted_counts else 0
    
    # Step 5: Accent folded
    folded_names = [fold_accents_fast(x) for x in clean_names]
    folded_unique = len(set(folded_names))
    
    return {
        "dataset_sample_size": total_records,
        "raw_unique_count": raw_unique,
        "raw_max_collision_cluster": raw_max_cluster,
        "clean_unique_count": clean_unique,
        "clean_reduction_pct": round((raw_unique - clean_unique) / raw_unique * 100, 2),
        "clean_max_collision_cluster": clean_max_cluster,
        "suffix_stripped_unique_count": stripped_unique,
        "suffix_stripped_reduction_pct": round((clean_unique - stripped_unique) / clean_unique * 100, 2),
        "suffix_stripped_max_collision_cluster": stripped_max_cluster,
        "token_sorted_unique_count": sorted_unique,
        "token_sorted_reduction_pct": round((stripped_unique - sorted_unique) / stripped_unique * 100, 2),
        "token_sorted_max_collision_cluster": sorted_max_cluster,
        "accent_folded_unique_count": folded_unique,
    }

def analyze_ground_truth_missingness() -> dict:
    gt_path = ROOT / "train/train_ground_truth.tsv"
    s1_path = ROOT / "train/train_source1.tsv"
    s2_path = ROOT / "train/train_source2.tsv"
    s3_path = ROOT / "train/train_source3.tsv"
    
    print("Analyzing ground-truth link missingness ...", flush=True)
    gt = pl.read_csv(gt_path, separator="\t")
    s2 = pl.read_csv(s2_path, separator="\t", columns=["entity_id", "business_address"])
    s3 = pl.read_csv(s3_path, separator="\t", columns=["entity_id", "business_address"])
    
    # Map S2 and S3 IDs to whether address is null
    s2_null_ids = set(s2.filter(pl.col("business_address").is_null())["entity_id"].to_list())
    s3_null_ids = set(s3.filter(pl.col("business_address").is_null())["entity_id"].to_list())
    all_null_target_ids = s2_null_ids | s3_null_ids
    
    # Count true links where target address is null
    # ground truth rows: source1_entity_id, matched_entity_id
    gt_target_col = gt.columns[1]
    gt_targets = gt[gt_target_col].to_list()
    total_true_links = len(gt_targets)
    
    true_links_with_null_target_addr = sum(1 for tid in gt_targets if tid in all_null_target_ids)
    
    return {
        "total_true_links": total_true_links,
        "s2_null_addr_entities": len(s2_null_ids),
        "s3_null_addr_entities": len(s3_null_ids),
        "true_links_with_null_target_address": true_links_with_null_target_addr,
        "true_links_with_null_target_address_pct": round(true_links_with_null_target_addr / total_true_links * 100, 2),
    }

def main():
    print("=== Running Phase 1A Data Foundation Forensic Investigation ===")
    results = {
        "tables": {},
        "collisions_sample_500k": {},
        "gt_missingness": {},
    }
    
    for split in ["train", "test"]:
        for src in ["1", "2", "3"]:
            key = f"{split}_s{src}"
            results["tables"][key] = analyze_table(split, src)
            
    results["collisions_sample_500k"] = analyze_normalization_collisions("train", "1", 500_000)
    results["gt_missingness"] = analyze_ground_truth_missingness()
    
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print(f"Results successfully saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
