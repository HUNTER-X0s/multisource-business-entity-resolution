"""
experiments/run_candidate_benchmark_fast.py
===========================================
High-speed candidate retrieval benchmark across complementary channels
using Polars vectorized multi-threading.
Evaluates on 50,000 validation queries against FULL 10.32M target pool:
  - EXP-1.1: Exact Clean Name
  - EXP-1.2: Token-Sorted Suffix-Stripped Name
  - EXP-1.3: Union (Exact + Token-Sorted)
  - EXP-1.4: Multi-Pass Union (Exact + Token-Sorted + Postal/FirstWord)
  - EXP-1.5: Budget Analysis (Top-K in [15, 20, 25, 30])
"""

import os
import sys
import json
import time

sys.path.insert(0, os.path.abspath("."))

import polars as pl
from src.vectorized_pipeline import normalize_dataframe, run_polars_candidate_generation
from src.candidate_evaluator import CandidateEvaluator

print("=" * 65, flush=True)
print("PHASE 1: HIGH-SPEED VECTORIZED CANDIDATE RETRIEVAL BENCHMARK", flush=True)
print("=" * 65, flush=True)

# 1. Load validation queries
VAL_PATH = "experiments/val_sample_50k.parquet"
print(f"Loading validation queries from {VAL_PATH}...", flush=True)
val_df = pl.read_parquet(VAL_PATH)
print(f"Loaded {len(val_df):,} validation queries.", flush=True)

# Build ground truth dict
val_gt = {}
for row in val_df.iter_rows(named=True):
    s1_id = row["entity_id"]
    m_str = row.get("matched_entity_ids") or ""
    val_gt[s1_id] = set(m.strip() for m in m_str.split(",") if m.strip())

evaluator = CandidateEvaluator(val_gt)
print(f"Evaluator initialized: {evaluator.total_s1:,} queries, {evaluator.total_links:,} true links.", flush=True)

# 2. Normalize validation queries
t0 = time.time()
print("\nNormalizing validation queries with Polars...", flush=True)
val_clean = normalize_dataframe(val_df)
print(f"Validation queries normalized in {time.time()-t0:.2f}s", flush=True)

# 3. Load or build cached target pool
CACHE_DIR = "experiments/cache"
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_PATH = os.path.join(CACHE_DIR, "targets_train_normalized.parquet")

if os.path.exists(CACHE_PATH):
    print(f"\nLoading precomputed normalized targets from {CACHE_PATH}...", flush=True)
    t_load = time.time()
    targets_clean = pl.read_parquet(CACHE_PATH)
    print(f"Loaded {len(targets_clean):,} normalized targets in {time.time()-t_load:.2f}s", flush=True)
else:
    print("\nPrecomputing normalized targets from raw files...", flush=True)
    t_raw = time.time()
    s2 = pl.read_csv("dataset/raw/train/train_source2.tsv", separator="\t", columns=["entity_id", "country", "business_name", "business_address"])
    s3 = pl.read_csv("dataset/raw/train/train_source3.tsv", separator="\t", columns=["entity_id", "country", "business_name", "business_address"])
    targets_raw = pl.concat([s2, s3])
    print(f"Raw targets loaded ({len(targets_raw):,} rows) in {time.time()-t_raw:.2f}s", flush=True)
    
    t_norm = time.time()
    targets_clean = normalize_dataframe(targets_raw)
    print(f"Vectorized normalization complete in {time.time()-t_norm:.2f}s", flush=True)
    
    print(f"Caching normalized targets to {CACHE_PATH}...", flush=True)
    targets_clean.write_parquet(CACHE_PATH)
    print("Cached successfully.", flush=True)

all_reports = []

def evaluate_pairs(pairs_df: pl.DataFrame, exp_id: str, elapsed: float) -> dict:
    # Convert pairs to dict {entity_id: set_of_targets}
    grouped = pairs_df.group_by("entity_id").agg(pl.col("target_id").alias("candidates"))
    cand_dict = {r["entity_id"]: set(r["candidates"]) for r in grouped.iter_rows(named=True)}
    # Fill queries with 0 candidates
    for s1_id in val_df["entity_id"].to_list():
        if s1_id not in cand_dict:
            cand_dict[s1_id] = set()
    return evaluator.evaluate(cand_dict, exp_id, elapsed)

# =================================================================
# EXP-1.1: Exact Clean Name
# =================================================================
print("\n--- Running EXP-1.1: Exact Clean Name Retrieval ---", flush=True)
t1 = time.time()
pairs_1_1 = val_clean.select(["entity_id", "country", "clean_name"]).filter(pl.col("clean_name") != "").join(
    targets_clean.select(["entity_id", "country", "clean_name"]).filter(pl.col("clean_name") != "").rename({"entity_id": "target_id"}),
    on=["country", "clean_name"],
    how="inner"
).select(["entity_id", "target_id"]).unique()

rep_1_1 = evaluate_pairs(pairs_1_1, "EXP-1.1_exact_clean_name", time.time() - t1)
all_reports.append(rep_1_1)
print(f"  Recall: {rep_1_1['overall_link_recall_pct']}% | Coverage: {rep_1_1['entity_coverage_pct']}% | Mean cands: {rep_1_1['candidate_volume']['mean_per_s1']} | Time: {rep_1_1['runtime_seconds']}s")

# =================================================================
# EXP-1.2: Suffix-Stripped Token-Sorted Name
# =================================================================
print("\n--- Running EXP-1.2: Suffix-Stripped Token-Sorted Retrieval ---", flush=True)
t2 = time.time()
pairs_1_2 = val_clean.select(["entity_id", "country", "sorted_name"]).filter(pl.col("sorted_name") != "").join(
    targets_clean.select(["entity_id", "country", "sorted_name"]).filter(pl.col("sorted_name") != "").rename({"entity_id": "target_id"}),
    on=["country", "sorted_name"],
    how="inner"
).select(["entity_id", "target_id"]).unique()

rep_1_2 = evaluate_pairs(pairs_1_2, "EXP-1.2_sorted_suffix_name", time.time() - t2)
all_reports.append(rep_1_2)
print(f"  Recall: {rep_1_2['overall_link_recall_pct']}% | Coverage: {rep_1_2['entity_coverage_pct']}% | Mean cands: {rep_1_2['candidate_volume']['mean_per_s1']} | Time: {rep_1_2['runtime_seconds']}s")

# =================================================================
# EXP-1.3: Union of (Exact + Token-Sorted)
# =================================================================
print("\n--- Running EXP-1.3: Union (Exact Clean + Token-Sorted) ---", flush=True)
t3 = time.time()
pairs_1_3 = pl.concat([pairs_1_1, pairs_1_2]).unique()
rep_1_3 = evaluate_pairs(pairs_1_3, "EXP-1.3_union_exact_sorted", time.time() - t3)
all_reports.append(rep_1_3)
print(f"  Recall: {rep_1_3['overall_link_recall_pct']}% | Coverage: {rep_1_3['entity_coverage_pct']}% | Mean cands: {rep_1_3['candidate_volume']['mean_per_s1']} | Time: {rep_1_3['runtime_seconds']}s")

# =================================================================
# EXP-1.4: Multi-Pass Union (Exact + Token-Sorted + Postal/FirstWord)
# =================================================================
print("\n--- Running EXP-1.4: Multi-Pass Union (+ Postal/FirstWord) ---", flush=True)
t4 = time.time()
pairs_1_4 = run_polars_candidate_generation(val_clean, targets_clean, top_k_per_query=30, include_postal=True)
rep_1_4 = evaluate_pairs(pairs_1_4, "EXP-1.4_multi_pass_union_top30", time.time() - t4)
all_reports.append(rep_1_4)
print(f"  Recall: {rep_1_4['overall_link_recall_pct']}% | Coverage: {rep_1_4['entity_coverage_pct']}% | Mean cands: {rep_1_4['candidate_volume']['mean_per_s1']} | Time: {rep_1_4['runtime_seconds']}s")

# =================================================================
# EXP-1.5: Candidate Budget Ablation (K=15, 20, 25, 30)
# =================================================================
print("\n--- Running EXP-1.5: Candidate Budget Ablation ---", flush=True)
for k in [15, 20, 25]:
    tk = time.time()
    pairs_k = pairs_1_4.group_by("entity_id").head(k)
    rep_k = evaluate_pairs(pairs_k, f"EXP-1.5_budget_K{k}", time.time() - tk)
    all_reports.append(rep_k)
    print(f"  K={k}: Recall {rep_k['overall_link_recall_pct']}% | P95 cands: {rep_k['candidate_volume']['p95']} | Max cands: {rep_k['candidate_volume']['max']}")

# Save benchmark results
out_json = "experiments/retrieval_benchmark_results.json"
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(all_reports, f, indent=2)

print("\n" + "=" * 65, flush=True)
print(f"BENCHMARK COMPLETE! Saved all metrics to {out_json}", flush=True)
print("=" * 65, flush=True)
