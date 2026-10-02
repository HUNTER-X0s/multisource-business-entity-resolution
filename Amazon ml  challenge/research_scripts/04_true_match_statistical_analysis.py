#!/usr/bin/env python3
"""
Step 4: True Match Statistical Analysis
Analyzes true match relationships from train_ground_truth.tsv joined against
train_source1.tsv, train_source2.tsv, and train_source3.tsv.
Computes:
- Country agreement (cross-border matching check)
- Exact name and address match rates
- Rapidfuzz Levenshtein similarity, token sort ratio, token Jaccard similarity
- Numeric / PIN code agreement
- Separate statistical profiles for S1-S2 vs S1-S3
"""

import os
import json
import re
import polars as pl
from rapidfuzz import fuzz, distance
import numpy as np

def extract_numeric_tokens(s):
    if not s:
        return set()
    return set(re.findall(r"\b\d+\b", s))

def compute_percentiles(arr):
    if len(arr) == 0:
        return {}
    return {
        "min": round(float(np.min(arr)), 4),
        "p25": round(float(np.percentile(arr, 25)), 4),
        "median": round(float(np.median(arr)), 4),
        "mean": round(float(np.mean(arr)), 4),
        "p75": round(float(np.percentile(arr, 75)), 4),
        "p95": round(float(np.percentile(arr, 95)), 4),
        "p99": round(float(np.percentile(arr, 99)), 4),
        "max": round(float(np.max(arr)), 4),
    }

def analyze_pairs(df_joined, pair_label, sample_size=100000, seed=42):
    total_pairs = len(df_joined)
    print(f"\nAnalyzing {pair_label}: Total true pairs = {total_pairs:,}")
    
    # Country agreement across ALL pairs
    # Country agreement across ALL pairs
    country_exact = float(df_joined.select((pl.col("s1_country") == pl.col("s23_country")).fill_null(False).mean()).item())
    
    # Missingness across ALL pairs
    s1_name_empty = float(df_joined.select((pl.col("s1_name").is_null() | (pl.col("s1_name") == "")).mean()).item())
    s23_name_empty = float(df_joined.select((pl.col("s23_name").is_null() | (pl.col("s23_name") == "")).mean()).item())
    s1_addr_empty = float(df_joined.select((pl.col("s1_addr").is_null() | (pl.col("s1_addr") == "")).mean()).item())
    s23_addr_empty = float(df_joined.select((pl.col("s23_addr").is_null() | (pl.col("s23_addr") == "")).mean()).item())
    
    # Exact equality across ALL pairs
    name_exact_raw = float(df_joined.select((pl.col("s1_name") == pl.col("s23_name")).fill_null(False).mean()).item())
    addr_exact_raw = float(df_joined.select((pl.col("s1_addr") == pl.col("s23_addr")).fill_null(False).mean()).item())
    name_exact_lower = float(df_joined.select((pl.col("s1_name").str.to_lowercase() == pl.col("s23_name").str.to_lowercase()).fill_null(False).mean()).item())
    addr_exact_lower = float(df_joined.select((pl.col("s1_addr").str.to_lowercase() == pl.col("s23_addr").str.to_lowercase()).fill_null(False).mean()).item())
    
    # Subsample for computationally intensive string similarity metrics if total_pairs > sample_size
    if total_pairs > sample_size:
        print(f"Sampling {sample_size:,} pairs for intensive string similarity profiling...")
        df_sample = df_joined.sample(n=sample_size, seed=seed)
    else:
        df_sample = df_joined

    s1_names = df_sample["s1_name"].fill_null("").to_list()
    s23_names = df_sample["s23_name"].fill_null("").to_list()
    s1_addrs = df_sample["s1_addr"].fill_null("").to_list()
    s23_addrs = df_sample["s23_addr"].fill_null("").to_list()
    
    lev_names = []
    token_sort_names = []
    token_jaccard_names = []
    char_3gram_names = []
    
    lev_addrs = []
    token_sort_addrs = []
    token_jaccard_addrs = []
    num_overlap_addrs = []
    has_num_both_addrs = []

    for n1, n2, a1, a2 in zip(s1_names, s23_names, s1_addrs, s23_addrs):
        # Name metrics
        n1_low, n2_low = n1.lower(), n2.lower()
        lev_names.append(fuzz.ratio(n1_low, n2_low) / 100.0)
        token_sort_names.append(fuzz.token_sort_ratio(n1_low, n2_low) / 100.0)
        
        t1, t2 = set(n1_low.split()), set(n2_low.split())
        union_t = t1 | t2
        token_jaccard_names.append(len(t1 & t2) / len(union_t) if union_t else 1.0)
        
        # Char 3-grams
        g1 = {n1_low[i:i+3] for i in range(len(n1_low)-2)} if len(n1_low) >= 3 else {n1_low}
        g2 = {n2_low[i:i+3] for i in range(len(n2_low)-2)} if len(n2_low) >= 3 else {n2_low}
        union_g = g1 | g2
        char_3gram_names.append(len(g1 & g2) / len(union_g) if union_g else 1.0)
        
        # Address metrics
        a1_low, a2_low = a1.lower(), a2.lower()
        lev_addrs.append(fuzz.ratio(a1_low, a2_low) / 100.0)
        token_sort_addrs.append(fuzz.token_sort_ratio(a1_low, a2_low) / 100.0)
        
        at1, at2 = set(a1_low.split()), set(a2_low.split())
        union_at = at1 | at2
        token_jaccard_addrs.append(len(at1 & at2) / len(union_at) if union_at else 1.0)
        
        # Numeric tokens in address
        nums1 = extract_numeric_tokens(a1_low)
        nums2 = extract_numeric_tokens(a2_low)
        if nums1 and nums2:
            num_overlap_addrs.append(len(nums1 & nums2) / len(nums1 | nums2))
            has_num_both_addrs.append(1)
        else:
            has_num_both_addrs.append(0)

    return {
        "pair_type": pair_label,
        "total_true_pairs": total_pairs,
        "country_agreement_rate": round(float(country_exact), 5),
        "name_exact_raw_pct": round(float(name_exact_raw * 100), 2),
        "name_exact_lower_pct": round(float(name_exact_lower * 100), 2),
        "addr_exact_raw_pct": round(float(addr_exact_raw * 100), 2),
        "addr_exact_lower_pct": round(float(addr_exact_lower * 100), 2),
        "s1_name_empty_pct": round(float(s1_name_empty * 100), 4),
        "s23_name_empty_pct": round(float(s23_name_empty * 100), 4),
        "s1_addr_empty_pct": round(float(s1_addr_empty * 100), 4),
        "s23_addr_empty_pct": round(float(s23_addr_empty * 100), 4),
        "sample_size_evaluated": len(s1_names),
        "name_levenshtein_ratio": compute_percentiles(lev_names),
        "name_token_sort_ratio": compute_percentiles(token_sort_names),
        "name_token_jaccard": compute_percentiles(token_jaccard_names),
        "name_char_3gram_jaccard": compute_percentiles(char_3gram_names),
        "addr_levenshtein_ratio": compute_percentiles(lev_addrs),
        "addr_token_sort_ratio": compute_percentiles(token_sort_addrs),
        "addr_token_jaccard": compute_percentiles(token_jaccard_addrs),
        "addr_numeric_overlap": compute_percentiles(num_overlap_addrs) if num_overlap_addrs else {},
        "addr_both_have_numbers_pct": round(float(np.mean(has_num_both_addrs) * 100), 2),
    }

def main():
    print("Loading Ground Truth and Source Files via Polars...")
    gt_df = pl.read_csv("dataset/raw/train/train_ground_truth.tsv", separator="\t")
    s1_df = pl.read_csv("dataset/raw/train/train_source1.tsv", separator="\t")
    s2_df = pl.read_csv("dataset/raw/train/train_source2.tsv", separator="\t")
    s3_df = pl.read_csv("dataset/raw/train/train_source3.tsv", separator="\t")

    print(f"Loaded: GT={len(gt_df):,}, S1={len(s1_df):,}, S2={len(s2_df):,}, S3={len(s3_df):,}")

    # Explode ground truth matched IDs
    gt_exploded = (
        gt_df.filter(pl.col("matched_entity_ids").is_not_null() & (pl.col("matched_entity_ids") != ""))
        .with_columns(pl.col("matched_entity_ids").str.split(","))
        .explode("matched_entity_ids")
        .with_columns(pl.col("matched_entity_ids").str.strip_chars())
        .rename({"source1_entity_id": "s1_id", "matched_entity_ids": "match_id"})
    )

    print(f"Total positive links in ground truth: {len(gt_exploded):,}")

    # S1-S2 pairs
    gt_s2 = gt_exploded.filter(pl.col("match_id").str.starts_with("S2-"))
    # S1-S3 pairs
    gt_s3 = gt_exploded.filter(pl.col("match_id").str.starts_with("S3-"))

    print(f"S1-S2 links: {len(gt_s2):,}, S1-S3 links: {len(gt_s3):,}")

    # Join S1-S2
    s1_sub = s1_df.select([
        pl.col("entity_id").alias("s1_id"),
        pl.col("business_name").alias("s1_name"),
        pl.col("business_address").alias("s1_addr"),
        pl.col("country").alias("s1_country"),
    ])
    
    s2_sub = s2_df.select([
        pl.col("entity_id").alias("match_id"),
        pl.col("business_name").alias("s23_name"),
        pl.col("business_address").alias("s23_addr"),
        pl.col("country").alias("s23_country"),
    ])
    
    s3_sub = s3_df.select([
        pl.col("entity_id").alias("match_id"),
        pl.col("business_name").alias("s23_name"),
        pl.col("business_address").alias("s23_addr"),
        pl.col("country").alias("s23_country"),
    ])

    joined_s2 = gt_s2.join(s1_sub, on="s1_id", how="inner").join(s2_sub, on="match_id", how="inner")
    joined_s3 = gt_s3.join(s1_sub, on="s1_id", how="inner").join(s3_sub, on="match_id", how="inner")

    results_s2 = analyze_pairs(joined_s2, "S1_S2")
    results_s3 = analyze_pairs(joined_s3, "S1_S3")

    combined_results = {
        "S1_S2": results_s2,
        "S1_S3": results_s3
    }

    out_path = "research_scripts/04_true_match_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(combined_results, f, indent=2)
    print(f"\nAll true match analyses saved to {out_path}")

if __name__ == "__main__":
    main()
