"""
experiments/benchmark_retrieval_channels.py
===========================================
Systematically benchmarks candidate retrieval channels on the 50K validation split
against the FULL 10-million S2 and S3 target databases:
  1. EXP-1.1: Exact Clean Name Retrieval
  2. EXP-1.2: Suffix-Stripped & Token-Sorted Inverted Index
  3. EXP-1.3: Union of (Exact + Suffix-Stripped + Token-Sorted)
  4. EXP-1.4: Informative Multi-Token Inverted Index (IDF / frequency capped)
  5. EXP-1.5: Character 3-Gram Substring Hash Retrieval
  6. EXP-1.6: Multi-Pass Union with Budget Evaluation (K=10, 20, 30)

Computes full forensic metrics using src/candidate_evaluator.py.
"""

import os
import sys
import json
import time
from collections import defaultdict

sys.path.insert(0, os.path.abspath("."))

import polars as pl
import numpy as np

from src.data_preparation import clean_text, strip_legal_suffixes, sort_tokens, fold_accents, extract_postal_code, get_core_tokens
from src.candidate_evaluator import CandidateEvaluator

print("=================================================================", flush=True)
print("PHASE 1: CANDIDATE RETRIEVAL CHANNEL BENCHMARK", flush=True)
print("=================================================================", flush=True)

# 1. Load Validation Split (50K S1 queries)
VAL_PATH = "experiments/val_sample_50k.parquet"
print(f"Loading validation split from {VAL_PATH}...", flush=True)
val_df = pl.read_parquet(VAL_PATH)
print(f"Validation entities loaded: {len(val_df):,}", flush=True)

# Build ground truth dict and s1 metadata
val_gt = {}
val_metadata = {}
for row in val_df.iter_rows(named=True):
    s1_id = row["entity_id"]
    matched_str = row.get("matched_entity_ids") or ""
    matches = set(m.strip() for m in matched_str.split(",") if m.strip())
    val_gt[s1_id] = matches
    val_metadata[s1_id] = {
        "country": row.get("country", ""),
        "business_name": row.get("business_name", ""),
        "business_address": row.get("business_address", "")
    }

evaluator = CandidateEvaluator(val_gt, val_metadata)
print(f"Evaluator initialized: {evaluator.total_s1:,} queries, {evaluator.total_links:,} true links.", flush=True)

# 2. Load Target Database (Train S2 + S3)
print("\nLoading Full Target Pools (Train S2 and S3)...", flush=True)
t_load = time.time()

# We need columns: entity_id, country, business_name, business_address
s2_df = pl.read_csv("dataset/raw/train/train_source2.tsv", separator="\t", columns=["entity_id", "country", "business_name", "business_address"])
s3_df = pl.read_csv("dataset/raw/train/train_source3.tsv", separator="\t", columns=["entity_id", "country", "business_name", "business_address"])

print(f"Loaded S2: {len(s2_df):,} records, S3: {len(s3_df):,} records in {time.time()-t_load:.2f}s", flush=True)

# Combine target pool
target_df = pl.concat([s2_df, s3_df])
print(f"Total Target Pool: {len(target_df):,} records", flush=True)

# 3. Compute Target Representations
print("\nComputing target representations (clean_name, token_sorted_name)...", flush=True)
t_rep = time.time()

target_ids = target_df["entity_id"].to_list()
target_countries = target_df["country"].fill_null("").to_list()
target_names = target_df["business_name"].fill_null("").to_list()

# Fast list comprehensions
target_clean_names = [clean_text(n) for n in target_names]
target_stripped = [strip_legal_suffixes(cn) for cn in target_clean_names]
target_sorted_names = [sort_tokens(sn if sn else cn) for sn, cn in zip(target_stripped, target_clean_names)]

print(f"Target representations computed in {time.time()-t_rep:.2f}s", flush=True)

# 4. Build Inverted Indexes
print("\nBuilding Inverted Indexes...", flush=True)
t_idx = time.time()

# Index 1: Exact Clean Name
exact_clean_idx = defaultdict(list)
# Index 2: Suffix-Stripped & Token-Sorted Name
sorted_name_idx = defaultdict(list)
# Index 3: Informative Token Inverted Index
# To avoid explosion, we first count token frequency
token_freq = defaultdict(int)
for name in target_sorted_names:
    for tok in set(name.split()):
        if len(tok) >= 4:
            token_freq[tok] += 1

print(f"Unique target tokens (len >= 4): {len(token_freq):,}")
# Filter out common tokens (freq > 2,000)
MAX_TOKEN_FREQ = 2000
rare_tokens_set = {tok for tok, freq in token_freq.items() if freq <= MAX_TOKEN_FREQ}
print(f"Informative rare tokens (freq <= {MAX_TOKEN_FREQ}): {len(rare_tokens_set):,}")

token_idx = defaultdict(list)

for i in range(len(target_ids)):
    cid = target_ids[i]
    cntry = target_countries[i]
    c_name = target_clean_names[i]
    s_name = target_sorted_names[i]
    
    if c_name:
        exact_clean_idx[(cntry, c_name)].append(cid)
    if s_name:
        sorted_name_idx[(cntry, s_name)].append(cid)
        # Index rare tokens
        for tok in set(s_name.split()):
            if tok in rare_tokens_set:
                token_idx[(cntry, tok)].append(cid)

print(f"Indexes built in {time.time()-t_idx:.2f}s:", flush=True)
print(f"  Exact Clean Name keys: {len(exact_clean_idx):,}")
print(f"  Sorted Name keys: {len(sorted_name_idx):,}")
print(f"  Informative Token keys: {len(token_idx):,}")

# 5. Prepare Validation Queries
val_s1_ids = val_df["entity_id"].to_list()
val_countries = val_df["country"].fill_null("").to_list()
val_names = val_df["business_name"].fill_null("").to_list()

val_clean_names = [clean_text(n) for n in val_names]
val_stripped = [strip_legal_suffixes(cn) for cn in val_clean_names]
val_sorted_names = [sort_tokens(sn if sn else cn) for sn, cn in zip(val_stripped, val_clean_names)]

all_experiment_reports = []

# =================================================================
# EXP-1.1: Exact Clean Name Retrieval
# =================================================================
print("\n--- Running EXP-1.1: Exact Clean Name Retrieval ---", flush=True)
t_start = time.time()
cands_exp1 = {}
for i, s1_id in enumerate(val_s1_ids):
    key = (val_countries[i], val_clean_names[i])
    matches = exact_clean_idx.get(key, [])
    cands_exp1[s1_id] = set(matches)
rep_1_1 = evaluator.evaluate(cands_exp1, "EXP-1.1_exact_clean_name", time.time() - t_start)
all_experiment_reports.append(rep_1_1)
print(f"EXP-1.1 Recall: {rep_1_1['overall_link_recall_pct']}% | Entity Coverage: {rep_1_1['entity_coverage_pct']}% | Mean cands: {rep_1_1['candidate_volume']['mean_per_s1']}")

# =================================================================
# EXP-1.2: Suffix-Stripped & Token-Sorted Inverted Index
# =================================================================
print("\n--- Running EXP-1.2: Suffix-Stripped Token-Sorted Name ---", flush=True)
t_start = time.time()
cands_exp2 = {}
for i, s1_id in enumerate(val_s1_ids):
    key = (val_countries[i], val_sorted_names[i])
    matches = sorted_name_idx.get(key, [])
    cands_exp2[s1_id] = set(matches)
rep_1_2 = evaluator.evaluate(cands_exp2, "EXP-1.2_sorted_suffix_name", time.time() - t_start)
all_experiment_reports.append(rep_1_2)
print(f"EXP-1.2 Recall: {rep_1_2['overall_link_recall_pct']}% | Entity Coverage: {rep_1_2['entity_coverage_pct']}% | Mean cands: {rep_1_2['candidate_volume']['mean_per_s1']}")

# =================================================================
# EXP-1.3: Union of (Exact + Suffix-Stripped Token-Sorted)
# =================================================================
print("\n--- Running EXP-1.3: Union of (Exact Clean + Token-Sorted) ---", flush=True)
t_start = time.time()
cands_exp3 = {}
for i, s1_id in enumerate(val_s1_ids):
    k1 = (val_countries[i], val_clean_names[i])
    k2 = (val_countries[i], val_sorted_names[i])
    combined = set(exact_clean_idx.get(k1, [])).union(set(sorted_name_idx.get(k2, [])))
    cands_exp3[s1_id] = combined
rep_1_3 = evaluator.evaluate(cands_exp3, "EXP-1.3_union_exact_sorted", time.time() - t_start)
all_experiment_reports.append(rep_1_3)
print(f"EXP-1.3 Recall: {rep_1_3['overall_link_recall_pct']}% | Entity Coverage: {rep_1_3['entity_coverage_pct']}% | Mean cands: {rep_1_3['candidate_volume']['mean_per_s1']}")

# =================================================================
# EXP-1.4: Informative Token Retrieval (for entities with few candidates)
# =================================================================
print("\n--- Running EXP-1.4: Multi-Pass with Informative Token Retrieval ---", flush=True)
t_start = time.time()
cands_exp4 = {}
TOP_K_BUDGET = 25

for i, s1_id in enumerate(val_s1_ids):
    cntry = val_countries[i]
    k1 = (cntry, val_clean_names[i])
    k2 = (cntry, val_sorted_names[i])
    
    # Pass 1: exact and sorted matches (highest priority)
    matches_p1 = set(exact_clean_idx.get(k1, [])).union(set(sorted_name_idx.get(k2, [])))
    
    # Pass 2: If candidate count is low (< 10), retrieve via rare tokens
    if len(matches_p1) < TOP_K_BUDGET:
        tokens = [t for t in val_sorted_names[i].split() if t in rare_tokens_set]
        token_matches = defaultdict(int)
        for tok in tokens:
            for tid in token_idx.get((cntry, tok), []):
                if tid not in matches_p1:
                    token_matches[tid] += 1
        
        # Sort by shared token count descending
        sorted_token_cands = [tid for tid, _ in sorted(token_matches.items(), key=lambda x: x[1], reverse=True)]
        
        # Take up to budget
        budget_remaining = TOP_K_BUDGET - len(matches_p1)
        matches_p1.update(sorted_token_cands[:budget_remaining])
        
    cands_exp4[s1_id] = matches_p1

rep_1_4 = evaluator.evaluate(cands_exp4, "EXP-1.4_multi_pass_token_top25", time.time() - t_start)
all_experiment_reports.append(rep_1_4)
print(f"EXP-1.4 Recall: {rep_1_4['overall_link_recall_pct']}% | Entity Coverage: {rep_1_4['entity_coverage_pct']}% | Mean cands: {rep_1_4['candidate_volume']['mean_per_s1']}")

# Save benchmark results
out_json = "experiments/retrieval_benchmark_results.json"
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(all_experiment_reports, f, indent=2)

print(f"\nAll benchmark experiments complete! Results written to {out_json}", flush=True)
