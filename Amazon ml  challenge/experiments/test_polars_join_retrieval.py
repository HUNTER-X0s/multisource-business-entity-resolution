"""
experiments/test_polars_join_retrieval.py
=========================================
Tests lightning-fast, zero-overhead Polars columnar joins for candidate retrieval:
  - Measures runtime and RAM
  - Computes exact clean_name and token_sorted_name recall
"""

import os
import sys
import time
import polars as pl

sys.path.insert(0, os.path.abspath("."))
from src.data_preparation import clean_text, strip_legal_suffixes, sort_tokens
from src.candidate_evaluator import CandidateEvaluator

print("Testing Polars Columnar Join Retrieval...", flush=True)
t0 = time.time()

# 1. Load validation queries (50k)
val_df = pl.read_parquet("experiments/val_sample_50k.parquet")
print(f"Loaded validation queries: {len(val_df):,}")

# Prepare validation representations
print("Normalizing validation queries...", flush=True)
val_clean = val_df.with_columns([
    pl.col("business_name").fill_null("").map_elements(clean_text, return_dtype=pl.Utf8).alias("clean_name")
]).with_columns([
    pl.col("clean_name").map_elements(strip_legal_suffixes, return_dtype=pl.Utf8).alias("stripped_name")
]).with_columns([
    pl.col("stripped_name").map_elements(sort_tokens, return_dtype=pl.Utf8).alias("sorted_name")
])

# 2. Load Targets (S2 + S3) lazily or directly with selected columns
print("Loading and normalizing target pools (S2 + S3)...", flush=True)
t_load = time.time()

s2 = pl.read_csv("dataset/raw/train/train_source2.tsv", separator="\t", columns=["entity_id", "country", "business_name"])
s3 = pl.read_csv("dataset/raw/train/train_source3.tsv", separator="\t", columns=["entity_id", "country", "business_name"])

targets = pl.concat([s2, s3])
print(f"Targets loaded: {len(targets):,} in {time.time()-t_load:.2f}s")

# Normalize targets with Polars
t_norm = time.time()
targets_clean = targets.with_columns([
    pl.col("business_name").fill_null("").map_elements(clean_text, return_dtype=pl.Utf8).alias("clean_name")
]).with_columns([
    pl.col("clean_name").map_elements(strip_legal_suffixes, return_dtype=pl.Utf8).alias("stripped_name")
]).with_columns([
    pl.col("stripped_name").map_elements(sort_tokens, return_dtype=pl.Utf8).alias("sorted_name")
])
print(f"Targets normalized in {time.time()-t_norm:.2f}s")

# 3. Perform Fast Hash Joins in Polars!
print("Performing Polars Hash Joins...", flush=True)
t_join = time.time()

# Join 1: on (country, clean_name)
join1 = val_clean.select(["entity_id", "country", "clean_name"]).join(
    targets_clean.select(["entity_id", "country", "clean_name"]).rename({"entity_id": "target_id"}),
    on=["country", "clean_name"],
    how="inner"
)
print(f"Join 1 (Exact Clean Name) produced {len(join1):,} candidate pairs in {time.time()-t_join:.2f}s")

# Join 2: on (country, sorted_name)
t_j2 = time.time()
join2 = val_clean.select(["entity_id", "country", "sorted_name"]).join(
    targets_clean.select(["entity_id", "country", "sorted_name"]).rename({"entity_id": "target_id"}),
    on=["country", "sorted_name"],
    how="inner"
)
print(f"Join 2 (Sorted Name) produced {len(join2):,} candidate pairs in {time.time()-t_j2:.2f}s")

# Union candidate pairs and deduplicate
t_union = time.time()
all_pairs = pl.concat([
    join1.select(["entity_id", "target_id"]),
    join2.select(["entity_id", "target_id"])
]).unique()

print(f"Union produced {len(all_pairs):,} unique candidate pairs in {time.time()-t_union:.2f}s")

# Convert to dict for CandidateEvaluator
print("Evaluating candidate recall...", flush=True)
candidates_grouped = all_pairs.group_by("entity_id").agg(pl.col("target_id").alias("candidates"))
cand_dict = {row["entity_id"]: set(row["candidates"]) for row in candidates_grouped.iter_rows(named=True)}

# Fill empty sets for queries with 0 candidates
for s1_id in val_df["entity_id"].to_list():
    if s1_id not in cand_dict:
        cand_dict[s1_id] = set()

# Ground truth
val_gt = {}
for row in val_df.iter_rows(named=True):
    s1_id = row["entity_id"]
    m_str = row.get("matched_entity_ids") or ""
    val_gt[s1_id] = set(m.strip() for m in m_str.split(",") if m.strip())

evaluator = CandidateEvaluator(val_gt)
metrics = evaluator.evaluate(cand_dict, "EXP-1.3_polars_hash_join", time.time() - t0)

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
print(f"Singleton Mean Candidates:{metrics['candidate_volume']['singleton_mean_cands']}")
print(f"Total Runtime:            {metrics['runtime_seconds']}s")
print("====================================================")
