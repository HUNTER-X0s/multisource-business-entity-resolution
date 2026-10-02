"""
experiments/test_address_and_brand_blocking.py
==============================================
Empirical evaluation of Address Matching & Core Brand Token Channels:
  - Channel 1: Exact Clean Name
  - Channel 2: Suffix-Stripped Token-Sorted Name
  - Channel 3: Exact Clean Address Match
  - Channel 4: Normalized Street Key (country + street_number + street_name_word)
  - Channel 5: Significant Brand Token Match (token length >= 5, frequency <= 1000)
Evaluates on the 50,000 S1 validation split against FULL 10.32M target pool.
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
print("TESTING ADDRESS & CORE BRAND TOKEN BLOCKING CHANNELS", flush=True)
print("=" * 65, flush=True)

# 1. Load validation queries
val_df = pl.read_parquet("experiments/val_sample_50k.parquet")
val_gt = {}
for row in val_df.iter_rows(named=True):
    s1_id = row["entity_id"]
    m_str = row.get("matched_entity_ids") or ""
    val_gt[s1_id] = set(m.strip() for m in m_str.split(",") if m.strip())

evaluator = CandidateEvaluator(val_gt)
print(f"Evaluator initialized: {evaluator.total_s1:,} queries, {evaluator.total_links:,} true links.", flush=True)

# 2. Load cached targets and normalized queries
targets_clean = pl.read_parquet("experiments/cache/targets_train_normalized.parquet")
val_clean = normalize_dataframe(val_df)

# Channel 1: Exact Clean Name
print("\n[Channel 1] Exact Clean Name Join...", flush=True)
t1 = time.time()
c1_pairs = val_clean.select(["entity_id", "country", "clean_name"]).filter(pl.col("clean_name") != "").join(
    targets_clean.select(["entity_id", "country", "clean_name"]).filter(pl.col("clean_name") != "").rename({"entity_id": "target_id"}),
    on=["country", "clean_name"],
    how="inner"
).select(["entity_id", "target_id"]).unique()
print(f"  Pairs: {len(c1_pairs):,} in {time.time()-t1:.2f}s")

# Channel 2: Suffix-Stripped Token-Sorted Name
print("[Channel 2] Token-Sorted Suffix-Stripped Name Join...", flush=True)
t2 = time.time()
c2_pairs = val_clean.select(["entity_id", "country", "sorted_name"]).filter(pl.col("sorted_name") != "").join(
    targets_clean.select(["entity_id", "country", "sorted_name"]).filter(pl.col("sorted_name") != "").rename({"entity_id": "target_id"}),
    on=["country", "sorted_name"],
    how="inner"
).select(["entity_id", "target_id"]).unique()
print(f"  Pairs: {len(c2_pairs):,} in {time.time()-t2:.2f}s")

# Channel 3: Exact Clean Address Match
print("[Channel 3] Exact Clean Address Join...", flush=True)
t3 = time.time()
c3_pairs = val_clean.select(["entity_id", "country", "clean_address"]).filter(pl.col("clean_address").str.len_chars() >= 10).join(
    targets_clean.select(["entity_id", "country", "clean_address"]).filter(pl.col("clean_address").str.len_chars() >= 10).rename({"entity_id": "target_id"}),
    on=["country", "clean_address"],
    how="inner"
).select(["entity_id", "target_id"]).unique()
print(f"  Pairs: {len(c3_pairs):,} in {time.time()-t3:.2f}s")

# Channel 4: First 3 words of address (Street Address Key)
print("[Channel 4] Address Street Prefix (First 3 tokens) Join...", flush=True)
t4 = time.time()
val_addr_p3 = val_clean.with_columns([
    pl.col("clean_address").str.split(" ").list.slice(0, 3).list.join(" ").alias("addr_p3")
])
tgt_addr_p3 = targets_clean.with_columns([
    pl.col("clean_address").str.split(" ").list.slice(0, 3).list.join(" ").alias("addr_p3")
])

# Only join if addr_p3 has a digit and at least 8 chars (e.g. '154 grecian gardens')
c4_pairs = val_addr_p3.select(["entity_id", "country", "addr_p3"]).filter(
    (pl.col("addr_p3").str.contains(r"\d")) & (pl.col("addr_p3").str.len_chars() >= 8)
).join(
    tgt_addr_p3.select(["entity_id", "country", "addr_p3"]).filter(
        (pl.col("addr_p3").str.contains(r"\d")) & (pl.col("addr_p3").str.len_chars() >= 8)
    ).rename({"entity_id": "target_id"}),
    on=["country", "addr_p3"],
    how="inner"
).select(["entity_id", "target_id"]).unique()
print(f"  Pairs: {len(c4_pairs):,} in {time.time()-t4:.2f}s")

# Measure individual channel recalls
def eval_channel(pairs: pl.DataFrame, name: str):
    grouped = pairs.group_by("entity_id").agg(pl.col("target_id").alias("candidates"))
    cand_dict = {r["entity_id"]: set(r["candidates"]) for r in grouped.iter_rows(named=True)}
    for s1_id in val_df["entity_id"].to_list():
        if s1_id not in cand_dict:
            cand_dict[s1_id] = set()
    m = evaluator.evaluate(cand_dict, name)
    print(f"  --> {name}: Recall = {m['overall_link_recall_pct']}%, Coverage = {m['entity_coverage_pct']}%, Total Pairs = {m['candidate_volume']['total_candidate_pairs']:,}")
    return m

print("\n--- INDIVIDUAL CHANNEL RECALLS ---")
eval_channel(c1_pairs, "Ch1_Exact_Name")
eval_channel(c2_pairs, "Ch2_Sorted_Name")
eval_channel(c3_pairs, "Ch3_Exact_Address")
eval_channel(c4_pairs, "Ch4_Street_Prefix")

# Now evaluate Union of Channels
print("\n--- COMBINED CHANNELS UNION & BUDGETING ---")
# Union 1: Name Channels (C1 + C2)
u_name = pl.concat([c1_pairs, c2_pairs]).unique()
eval_channel(u_name, "Union_Name_Channels_(C1+C2)")

# Union 2: Name + Address Channels (C1 + C2 + C3 + C4)
u_name_addr = pl.concat([c1_pairs, c2_pairs, c3_pairs, c4_pairs]).unique()
m_all = eval_channel(u_name_addr, "Union_Name_AND_Address_(C1+C2+C3+C4)")

# Apply Top-K Budget (K=30)
u_budget30 = u_name_addr.group_by("entity_id").head(30)
m_b30 = eval_channel(u_budget30, "Budget_Top30_(C1+C2+C3+C4)")

# Save evaluation report
with open("experiments/address_blocking_results.json", "w", encoding="utf-8") as f:
    json.dump([m_all, m_b30], f, indent=2)

print("\nResults saved to experiments/address_blocking_results.json", flush=True)
