"""
experiments/test_token_and_sparse_retrieval.py
==============================================
Benchmarks sub-token and prefix retrieval channels:
  1. Prefix-2 Index: (country, token1, token2)
  2. Rare Token Index: (country, rarest_token) with frequency thresholding
  3. Multi-Channel Union with Exact & Sorted Name
Evaluates Link Recall and Entity Coverage on the 50K validation split.
"""

import os
import sys
import time
from collections import defaultdict, Counter

sys.path.insert(0, os.path.abspath("."))

import polars as pl
import numpy as np

from src.candidate_evaluator import CandidateEvaluator

print("=================================================================", flush=True)
print("BENCHMARKING SUB-TOKEN & PREFIX RETRIEVAL CHANNELS", flush=True)
print("=================================================================", flush=True)

# 1. Load validation queries
VAL_PATH = "experiments/val_sample_50k.parquet"
val_df = pl.read_parquet(VAL_PATH)

val_gt = {}
for row in val_df.iter_rows(named=True):
    s1_id = row["entity_id"]
    m_str = row.get("matched_entity_ids") or ""
    val_gt[s1_id] = set(m.strip() for m in m_str.split(",") if m.strip())

evaluator = CandidateEvaluator(val_gt)
print(f"Evaluator initialized: {evaluator.total_s1:,} queries, {evaluator.total_links:,} true links.", flush=True)

# 2. Load precomputed normalized targets from cache
CACHE_PATH = "experiments/cache/targets_train_normalized.parquet"
print(f"Loading normalized targets from {CACHE_PATH}...", flush=True)
t_load = time.time()
targets_clean = pl.read_parquet(CACHE_PATH)
print(f"Loaded {len(targets_clean):,} normalized targets in {time.time()-t_load:.2f}s", flush=True)

# Also load or compute normalized validation queries
from src.vectorized_pipeline import normalize_dataframe
val_clean = normalize_dataframe(val_df)

# 3. Add Prefix-2 token column in Polars
print("Computing Prefix-2 tokens...", flush=True)
# Take first 2 tokens of sorted_name or clean_name
targets_p2 = targets_clean.with_columns([
    pl.col("sorted_name").str.split(" ").list.slice(0, 2).list.join(" ").alias("prefix2")
])
val_p2 = val_clean.with_columns([
    pl.col("sorted_name").str.split(" ").list.slice(0, 2).list.join(" ").alias("prefix2")
])

# 4. Hash Join on Prefix-2!
print("Running Prefix-2 Columnar Join...", flush=True)
t_p2 = time.time()
p2_join = val_p2.select(["entity_id", "country", "prefix2"]).filter(pl.col("prefix2").str.len_chars() >= 4).join(
    targets_p2.select(["entity_id", "country", "prefix2"]).filter(pl.col("prefix2").str.len_chars() >= 4).rename({"entity_id": "target_id"}),
    on=["country", "prefix2"],
    how="inner"
).select(["entity_id", "target_id"]).unique()

print(f"Prefix-2 join produced {len(p2_join):,} pairs in {time.time()-t_p2:.2f}s", flush=True)

# 5. Also compute Exact Clean + Sorted pairs
print("Running Exact Clean & Sorted Joins...", flush=True)
exact_clean = val_clean.select(["entity_id", "country", "clean_name"]).filter(pl.col("clean_name") != "").join(
    targets_clean.select(["entity_id", "country", "clean_name"]).filter(pl.col("clean_name") != "").rename({"entity_id": "target_id"}),
    on=["country", "clean_name"],
    how="inner"
).select(["entity_id", "target_id"]).unique()

sorted_clean = val_clean.select(["entity_id", "country", "sorted_name"]).filter(pl.col("sorted_name") != "").join(
    targets_clean.select(["entity_id", "country", "sorted_name"]).filter(pl.col("sorted_name") != "").rename({"entity_id": "target_id"}),
    on=["country", "sorted_name"],
    how="inner"
).select(["entity_id", "target_id"]).unique()

# Combine exact, sorted, and prefix2
print("Combining channels...", flush=True)
t_union = time.time()
combined_pairs = pl.concat([exact_clean, sorted_clean, p2_join]).unique()
print(f"Combined pairs: {len(combined_pairs):,} in {time.time()-t_union:.2f}s", flush=True)

# Apply Top-K budget per S1
TOP_K = 30
budgeted_pairs = combined_pairs.group_by("entity_id").head(TOP_K)

# Convert to dict and evaluate
grouped = budgeted_pairs.group_by("entity_id").agg(pl.col("target_id").alias("candidates"))
cand_dict = {r["entity_id"]: set(r["candidates"]) for r in grouped.iter_rows(named=True)}
for s1_id in val_df["entity_id"].to_list():
    if s1_id not in cand_dict:
        cand_dict[s1_id] = set()

metrics = evaluator.evaluate(cand_dict, "EXP-1.7_exact_sorted_prefix2_top30", time.time() - t_load)

print("\n================ EVALUATION RESULTS ================")
print(f"Overall Link Recall:      {metrics['overall_link_recall_pct']}%")
print(f"Entity Coverage:          {metrics['entity_coverage_pct']}%")
print(f"S2 Link Recall:           {metrics['s2_link_recall_pct']}%")
print(f"S3 Link Recall:           {metrics['s3_link_recall_pct']}%")
print(f"S2-only Entity Recall:    {metrics['s2_only_entity_recall_pct']}%")
print(f"S3-only Entity Recall:    {metrics['s3_only_entity_recall_pct']}%")
print(f"Both-source Recall:       {metrics['both_source_entity_recall_pct']}%")
print(f"Candidate Volume (Total): {metrics['candidate_volume']['total_candidate_pairs']:,}")
print(f"Candidates per S1 (Mean): {metrics['candidate_volume']['mean_per_s1']}")
print(f"Candidates per S1 (P95):  {metrics['candidate_volume']['p95']}")
print(f"Candidates per S1 (Max):  {metrics['candidate_volume']['max']}")
print(f"Total Runtime:            {metrics['runtime_seconds']}s")
print("====================================================")
