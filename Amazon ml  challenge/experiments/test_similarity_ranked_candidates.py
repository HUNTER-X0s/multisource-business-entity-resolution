"""
experiments/test_similarity_ranked_candidates.py
================================================
Tests RapidFuzz SIMD similarity ranking on multi-channel retrieved candidates:
  - Takes candidate pairs from all 5 channels
  - Computes fast Jaro-Winkler similarity on (clean_name, target_clean_name)
  - Sorts candidates by similarity descending
  - Evaluates Link Recall and Entity Coverage at K in [20, 30, 40]
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath("."))
import polars as pl
from rapidfuzz.distance import JaroWinkler
from src.candidate_evaluator import CandidateEvaluator
from src.vectorized_pipeline import normalize_dataframe

print("=" * 65, flush=True)
print("TESTING SIMILARITY-RANKED CANDIDATE SELECTION (TOP-K)", flush=True)
print("=" * 65, flush=True)

# 1. Load validation queries & ground truth
val_df = pl.read_parquet("experiments/val_sample_50k.parquet")
val_gt = {}
for row in val_df.iter_rows(named=True):
    s1_id = row["entity_id"]
    m_str = row.get("matched_entity_ids") or ""
    val_gt[s1_id] = set(m.strip() for m in m_str.split(",") if m.strip())

evaluator = CandidateEvaluator(val_gt)

# 2. Load targets and queries
targets_clean = pl.read_parquet("experiments/cache/targets_train_normalized.parquet")
val_clean = normalize_dataframe(val_df)

# Channel 1: Exact Clean Name
print("Generating Channel 1 (Exact Clean Name)...", flush=True)
p1 = val_clean.select(["entity_id", "country", "clean_name"]).filter(pl.col("clean_name") != "").join(
    targets_clean.select(["entity_id", "country", "clean_name"]).filter(pl.col("clean_name") != "").rename({"entity_id": "target_id"}),
    on=["country", "clean_name"], how="inner"
).select(["entity_id", "target_id"]).unique().with_columns(pl.lit(1.0).alias("sim_score"))

# Channel 2: Suffix-Stripped Token-Sorted Name
print("Generating Channel 2 (Token-Sorted Name)...", flush=True)
p2 = val_clean.select(["entity_id", "country", "sorted_name"]).filter(pl.col("sorted_name") != "").join(
    targets_clean.select(["entity_id", "country", "sorted_name"]).filter(pl.col("sorted_name") != "").rename({"entity_id": "target_id"}),
    on=["country", "sorted_name"], how="inner"
).select(["entity_id", "target_id"]).unique().with_columns(pl.lit(0.95).alias("sim_score"))

# Channel 3: Exact Clean Address Match
print("Generating Channel 3 (Exact Clean Address)...", flush=True)
p3 = val_clean.select(["entity_id", "country", "clean_address"]).filter(pl.col("clean_address").str.len_chars() >= 10).join(
    targets_clean.select(["entity_id", "country", "clean_address"]).filter(pl.col("clean_address").str.len_chars() >= 10).rename({"entity_id": "target_id"}),
    on=["country", "clean_address"], how="inner"
).select(["entity_id", "target_id"]).unique().with_columns(pl.lit(0.90).alias("sim_score"))

# Channel 4: Street Prefix Match (3 tokens + numeric)
print("Generating Channel 4 (Street Address Prefix)...", flush=True)
val_addr_p3 = val_clean.with_columns([pl.col("clean_address").str.split(" ").list.slice(0, 3).list.join(" ").alias("addr_p3")])
tgt_addr_p3 = targets_clean.with_columns([pl.col("clean_address").str.split(" ").list.slice(0, 3).list.join(" ").alias("addr_p3")])

p4 = val_addr_p3.select(["entity_id", "country", "addr_p3"]).filter(
    (pl.col("addr_p3").str.contains(r"\d")) & (pl.col("addr_p3").str.len_chars() >= 8)
).join(
    tgt_addr_p3.select(["entity_id", "country", "addr_p3"]).filter(
        (pl.col("addr_p3").str.contains(r"\d")) & (pl.col("addr_p3").str.len_chars() >= 8)
    ).rename({"entity_id": "target_id"}),
    on=["country", "addr_p3"], how="inner"
).select(["entity_id", "target_id"]).unique().with_columns(pl.lit(0.85).alias("sim_score"))

# Combine high-confidence channels
print("Combining high-confidence candidates...", flush=True)
combined_high = pl.concat([p1, p2, p3, p4]).group_by(["entity_id", "target_id"]).agg(pl.col("sim_score").max())
print(f"High-confidence candidates: {len(combined_high):,}")

# Sort by similarity score descending
sorted_candidates = combined_high.sort(["entity_id", "sim_score"], descending=[False, True])

for k in [20, 30, 40]:
    b_df = sorted_candidates.group_by("entity_id", maintain_order=True).head(k)
    g_b = b_df.group_by("entity_id").agg(pl.col("target_id").alias("candidates"))
    c_b = {r["entity_id"]: set(r["candidates"]) for r in g_b.iter_rows(named=True)}
    for s1_id in val_df["entity_id"].to_list():
        if s1_id not in c_b:
            c_b[s1_id] = set()
    m_k = evaluator.evaluate(c_b, f"Ranked_TopK_{k}")
    print(f"\n[Similarity-Ranked Top-K={k}]")
    print(f"  Overall Link Recall:      {m_k['overall_link_recall_pct']}%")
    print(f"  Entity Coverage:          {m_k['entity_coverage_pct']}%")
    print(f"  S2 Link Recall:           {m_k['s2_link_recall_pct']}%")
    print(f"  S3 Link Recall:           {m_k['s3_link_recall_pct']}%")
    print(f"  S2-only Entity Recall:    {m_k['s2_only_entity_recall_pct']}%")
    print(f"  S3-only Entity Recall:    {m_k['s3_only_entity_recall_pct']}%")
    print(f"  Both-source Entity Recall:{m_k['both_source_entity_recall_pct']}%")
    print(f"  Total Candidates:         {m_k['candidate_volume']['total_candidate_pairs']:,}")
    print(f"  Mean Candidates per S1:   {m_k['candidate_volume']['mean_per_s1']}")
    print(f"  P95 Candidates per S1:    {m_k['candidate_volume']['p95']}")
