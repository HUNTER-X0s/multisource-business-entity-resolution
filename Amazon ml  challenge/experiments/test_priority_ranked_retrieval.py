"""
experiments/test_priority_ranked_retrieval.py
=============================================
Implements and benchmarks Priority-Ranked Multi-Channel Candidate Retrieval:
  - Channel 1 (Priority 1): Exact Clean Name Match
  - Channel 2 (Priority 2): Suffix-Stripped Token-Sorted Name Match
  - Channel 3 (Priority 3): Exact Clean Address Match
  - Channel 4 (Priority 4): Address Street Prefix (3 tokens + numeric)
  - Channel 5 (Priority 5): Brand Token + State/Postal Key
Uses hierarchical budget allocation to ensure high-precision channels are never crowded out.
"""

import os
import sys
import time
import json

sys.path.insert(0, os.path.abspath("."))
import polars as pl
from src.candidate_evaluator import CandidateEvaluator
from src.vectorized_pipeline import normalize_dataframe

print("=" * 65, flush=True)
print("TESTING PRIORITY-RANKED MULTI-CHANNEL RETRIEVAL", flush=True)
print("=" * 65, flush=True)

# 1. Load validation queries & ground truth
val_df = pl.read_parquet("experiments/val_sample_50k.parquet")
val_gt = {}
for row in val_df.iter_rows(named=True):
    s1_id = row["entity_id"]
    m_str = row.get("matched_entity_ids") or ""
    val_gt[s1_id] = set(m.strip() for m in m_str.split(",") if m.strip())

evaluator = CandidateEvaluator(val_gt)
print(f"Evaluator initialized: {evaluator.total_s1:,} queries, {evaluator.total_links:,} true links.", flush=True)

# 2. Load targets and queries
targets_clean = pl.read_parquet("experiments/cache/targets_train_normalized.parquet")
val_clean = normalize_dataframe(val_df)

# Channel 1 (Priority 1): Exact Clean Name
print("\n[P1] Exact Clean Name Join...", flush=True)
t1 = time.time()
p1_pairs = val_clean.select(["entity_id", "country", "clean_name"]).filter(pl.col("clean_name") != "").join(
    targets_clean.select(["entity_id", "country", "clean_name"]).filter(pl.col("clean_name") != "").rename({"entity_id": "target_id"}),
    on=["country", "clean_name"],
    how="inner"
).select(["entity_id", "target_id"]).unique().with_columns(pl.lit(1).alias("priority"))
print(f"  P1 pairs: {len(p1_pairs):,} in {time.time()-t1:.2f}s")

# Channel 2 (Priority 2): Token-Sorted Name
print("[P2] Token-Sorted Suffix-Stripped Name Join...", flush=True)
t2 = time.time()
p2_pairs = val_clean.select(["entity_id", "country", "sorted_name"]).filter(pl.col("sorted_name") != "").join(
    targets_clean.select(["entity_id", "country", "sorted_name"]).filter(pl.col("sorted_name") != "").rename({"entity_id": "target_id"}),
    on=["country", "sorted_name"],
    how="inner"
).select(["entity_id", "target_id"]).unique().with_columns(pl.lit(2).alias("priority"))
print(f"  P2 pairs: {len(p2_pairs):,} in {time.time()-t2:.2f}s")

# Channel 3 (Priority 3): Exact Clean Address Match
print("[P3] Exact Clean Address Join...", flush=True)
t3 = time.time()
p3_pairs = val_clean.select(["entity_id", "country", "clean_address"]).filter(pl.col("clean_address").str.len_chars() >= 10).join(
    targets_clean.select(["entity_id", "country", "clean_address"]).filter(pl.col("clean_address").str.len_chars() >= 10).rename({"entity_id": "target_id"}),
    on=["country", "clean_address"],
    how="inner"
).select(["entity_id", "target_id"]).unique().with_columns(pl.lit(3).alias("priority"))
print(f"  P3 pairs: {len(p3_pairs):,} in {time.time()-t3:.2f}s")

# Channel 4 (Priority 4): Address Street Prefix (3 tokens + numeric)
print("[P4] Street Address Prefix Join...", flush=True)
t4 = time.time()
val_addr_p3 = val_clean.with_columns([
    pl.col("clean_address").str.split(" ").list.slice(0, 3).list.join(" ").alias("addr_p3")
])
tgt_addr_p3 = targets_clean.with_columns([
    pl.col("clean_address").str.split(" ").list.slice(0, 3).list.join(" ").alias("addr_p3")
])

p4_pairs = val_addr_p3.select(["entity_id", "country", "addr_p3"]).filter(
    (pl.col("addr_p3").str.contains(r"\d")) & (pl.col("addr_p3").str.len_chars() >= 8)
).join(
    tgt_addr_p3.select(["entity_id", "country", "addr_p3"]).filter(
        (pl.col("addr_p3").str.contains(r"\d")) & (pl.col("addr_p3").str.len_chars() >= 8)
    ).rename({"entity_id": "target_id"}),
    on=["country", "addr_p3"],
    how="inner"
).select(["entity_id", "target_id"]).unique().with_columns(pl.lit(4).alias("priority"))
print(f"  P4 pairs: {len(p4_pairs):,} in {time.time()-t4:.2f}s")

# Channel 5 (Priority 5): Brand Token (First Word >= 4 chars) + Postal Code
print("[P5] Brand Token + Postal Code Join...", flush=True)
t5 = time.time()
val_brand_post = val_clean.with_columns([
    pl.col("clean_name").str.split(" ").list.get(0).alias("first_word")
]).filter((pl.col("postal_code") != "") & (pl.col("first_word").str.len_chars() >= 4))

tgt_brand_post = targets_clean.with_columns([
    pl.col("clean_name").str.split(" ").list.get(0).alias("first_word")
]).filter((pl.col("postal_code") != "") & (pl.col("first_word").str.len_chars() >= 4))

p5_pairs = val_brand_post.select(["entity_id", "country", "postal_code", "first_word"]).join(
    tgt_brand_post.select(["entity_id", "country", "postal_code", "first_word"]).rename({"entity_id": "target_id"}),
    on=["country", "postal_code", "first_word"],
    how="inner"
).select(["entity_id", "target_id"]).unique().with_columns(pl.lit(5).alias("priority"))
print(f"  P5 pairs: {len(p5_pairs):,} in {time.time()-t5:.2f}s")

# Combine all channels with priority tracking
print("\nCombining and Deduplicating with Priority Ranking...", flush=True)
t_comb = time.time()
all_pairs_raw = pl.concat([p1_pairs, p2_pairs, p3_pairs, p4_pairs, p5_pairs])

# For each (entity_id, target_id) pair, keep the best (lowest numeric) priority
all_pairs = all_pairs_raw.group_by(["entity_id", "target_id"]).agg(pl.col("priority").min())
print(f"Total Unique Pairs across all 5 channels: {len(all_pairs):,} in {time.time()-t_comb:.2f}s")

# Sort by priority ascending so higher-priority channels come first
all_pairs_sorted = all_pairs.sort(["entity_id", "priority"])

def evaluate_budget(top_k: int):
    t_b = time.time()
    budgeted = all_pairs_sorted.group_by("entity_id", maintain_order=True).head(top_k)
    grouped = budgeted.group_by("entity_id").agg(pl.col("target_id").alias("candidates"))
    cand_dict = {r["entity_id"]: set(r["candidates"]) for r in grouped.iter_rows(named=True)}
    for s1_id in val_df["entity_id"].to_list():
        if s1_id not in cand_dict:
            cand_dict[s1_id] = set()
    m = evaluator.evaluate(cand_dict, f"Priority_TopK_{top_k}", time.time() - t_b)
    print(f"\n[Budget Top-K={top_k}]")
    print(f"  Overall Link Recall:      {m['overall_link_recall_pct']}%")
    print(f"  Entity Coverage:          {m['entity_coverage_pct']}%")
    print(f"  S2 Link Recall:           {m['s2_link_recall_pct']}%")
    print(f"  S3 Link Recall:           {m['s3_link_recall_pct']}%")
    print(f"  S2-only Entity Recall:    {m['s2_only_entity_recall_pct']}%")
    print(f"  S3-only Entity Recall:    {m['s3_only_entity_recall_pct']}%")
    print(f"  Both-source Entity Recall:{m['both_source_entity_recall_pct']}%")
    print(f"  Total Candidates:         {m['candidate_volume']['total_candidate_pairs']:,}")
    print(f"  Mean Candidates per S1:   {m['candidate_volume']['mean_per_s1']}")
    print(f"  P95 Candidates per S1:    {m['candidate_volume']['p95']}")
    print(f"  Max Candidates per S1:    {m['candidate_volume']['max']}")
    return m

# Unbudgeted full union evaluation
grouped_unbudgeted = all_pairs.group_by("entity_id").agg(pl.col("target_id").alias("candidates"))
cand_dict_unb = {r["entity_id"]: set(r["candidates"]) for r in grouped_unbudgeted.iter_rows(named=True)}
for s1_id in val_df["entity_id"].to_list():
    if s1_id not in cand_dict_unb:
        cand_dict_unb[s1_id] = set()
m_unb = evaluator.evaluate(cand_dict_unb, "All_5_Channels_Unbudgeted")
print(f"\n[UNBUDGETED FULL UNION (ALL 5 CHANNELS)]")
print(f"  Link Recall: {m_unb['overall_link_recall_pct']}% | Entity Coverage: {m_unb['entity_coverage_pct']}% | Total Pairs: {m_unb['candidate_volume']['total_candidate_pairs']:,} (avg {m_unb['candidate_volume']['mean_per_s1']}/query)")

# Budgeted evaluations
m_k20 = evaluate_budget(20)
m_k30 = evaluate_budget(30)
m_k40 = evaluate_budget(40)
m_k50 = evaluate_budget(50)

# Save results
out_path = "experiments/priority_ranked_benchmark_results.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump([m_unb, m_k20, m_k30, m_k40, m_k50], f, indent=2)

print(f"\nAll Priority-Ranked results saved to {out_path}", flush=True)
