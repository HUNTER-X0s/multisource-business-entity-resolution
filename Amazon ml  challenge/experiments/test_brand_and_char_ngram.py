"""
experiments/test_brand_and_char_ngram.py
========================================
Tests Channel 6: Distinctive Brand Token Inverted Index:
  - Extracts the first significant token (len >= 4, not in generic stop words)
  - Filters out high-frequency tokens (frequency > 500 in target corpus)
  - Joins queries with targets on (country, rare_brand_token)
  - Evaluates additive recall gain and candidate volume impact
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath("."))
import polars as pl
from src.candidate_evaluator import CandidateEvaluator
from src.vectorized_pipeline import normalize_dataframe

print("=" * 65, flush=True)
print("TESTING DISTINCTIVE BRAND TOKEN RETRIEVAL CHANNEL", flush=True)
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

# Generic stop words to exclude from brand tokens
GENERIC_WORDS = {
    "center", "services", "service", "enterprises", "enterprise", "solutions",
    "group", "holdings", "holding", "associates", "consultancy", "consulting",
    "management", "industries", "industry", "international", "global", "national",
    "united", "american", "india", "delhi", "mumbai", "texas", "california",
    "medical", "health", "care", "clinic", "hospital", "pharma", "pharmacy",
    "dental", "dentistry", "realty", "properties", "estate", "construction",
    "builders", "technologies", "technology", "tech", "systems", "corp", "inc"
}

# 3. Extract distinctive brand token (first token len >= 4 not in GENERIC_WORDS)
print("Extracting distinctive brand tokens...", flush=True)

# Polars expression: extract first token of stripped_name
val_brand = val_clean.with_columns([
    pl.col("stripped_name").str.split(" ").list.get(0).alias("raw_brand")
]).with_columns([
    pl.when((pl.col("raw_brand").str.len_chars() >= 4) & (~pl.col("raw_brand").is_in(list(GENERIC_WORDS))))
      .then(pl.col("raw_brand"))
      .otherwise(pl.lit(""))
      .alias("brand_token")
])

targets_brand = targets_clean.with_columns([
    pl.col("stripped_name").str.split(" ").list.get(0).alias("raw_brand")
]).with_columns([
    pl.when((pl.col("raw_brand").str.len_chars() >= 4) & (~pl.col("raw_brand").is_in(list(GENERIC_WORDS))))
      .then(pl.col("raw_brand"))
      .otherwise(pl.lit(""))
      .alias("brand_token")
])

# 4. Count target brand token frequency to prune common tokens (> 300 target records)
print("Auditing brand token frequencies in target corpus...", flush=True)
brand_counts = targets_brand.filter(pl.col("brand_token") != "").group_by(["country", "brand_token"]).len()
rare_brands = brand_counts.filter(pl.col("len") <= 300).select(["country", "brand_token"])

print(f"Total distinctive brand tokens: {len(brand_counts):,}")
print(f"Informative rare brand tokens (freq <= 300): {len(rare_brands):,}")

# Filter queries and targets to rare brand tokens only
val_rare = val_brand.select(["entity_id", "country", "brand_token"]).join(rare_brands, on=["country", "brand_token"], how="inner")
tgt_rare = targets_brand.select(["entity_id", "country", "brand_token"]).join(rare_brands, on=["country", "brand_token"], how="inner")

print("Running Rare Brand Token Join...", flush=True)
t_b = time.time()
brand_pairs = val_rare.join(
    tgt_rare.rename({"entity_id": "target_id"}),
    on=["country", "brand_token"],
    how="inner"
).select(["entity_id", "target_id"]).unique().with_columns(pl.lit(5).alias("priority"))

print(f"Rare Brand Token join produced {len(brand_pairs):,} pairs in {time.time()-t_b:.2f}s")

# Measure standalone recall of brand token join
grouped_b = brand_pairs.group_by("entity_id").agg(pl.col("target_id").alias("candidates"))
cand_dict_b = {r["entity_id"]: set(r["candidates"]) for r in grouped_b.iter_rows(named=True)}
for s1_id in val_df["entity_id"].to_list():
    if s1_id not in cand_dict_b:
        cand_dict_b[s1_id] = set()
m_brand = evaluator.evaluate(cand_dict_b, "Standalone_Rare_Brand_Token")
print(f"  Standalone Rare Brand: Recall = {m_brand['overall_link_recall_pct']}%, Coverage = {m_brand['entity_coverage_pct']}%")

# 5. Load Previous 4 Channels
print("\nCombining with Previous 4 Channels (Exact Name, Sorted Name, Exact Addr, Street Prefix)...", flush=True)
p1 = val_clean.select(["entity_id", "country", "clean_name"]).filter(pl.col("clean_name") != "").join(
    targets_clean.select(["entity_id", "country", "clean_name"]).filter(pl.col("clean_name") != "").rename({"entity_id": "target_id"}),
    on=["country", "clean_name"], how="inner"
).select(["entity_id", "target_id"]).unique().with_columns(pl.lit(1).alias("priority"))

p2 = val_clean.select(["entity_id", "country", "sorted_name"]).filter(pl.col("sorted_name") != "").join(
    targets_clean.select(["entity_id", "country", "sorted_name"]).filter(pl.col("sorted_name") != "").rename({"entity_id": "target_id"}),
    on=["country", "sorted_name"], how="inner"
).select(["entity_id", "target_id"]).unique().with_columns(pl.lit(2).alias("priority"))

p3 = val_clean.select(["entity_id", "country", "clean_address"]).filter(pl.col("clean_address").str.len_chars() >= 10).join(
    targets_clean.select(["entity_id", "country", "clean_address"]).filter(pl.col("clean_address").str.len_chars() >= 10).rename({"entity_id": "target_id"}),
    on=["country", "clean_address"], how="inner"
).select(["entity_id", "target_id"]).unique().with_columns(pl.lit(3).alias("priority"))

val_addr_p3 = val_clean.with_columns([pl.col("clean_address").str.split(" ").list.slice(0, 3).list.join(" ").alias("addr_p3")])
tgt_addr_p3 = targets_clean.with_columns([pl.col("clean_address").str.split(" ").list.slice(0, 3).list.join(" ").alias("addr_p3")])

p4 = val_addr_p3.select(["entity_id", "country", "addr_p3"]).filter(
    (pl.col("addr_p3").str.contains(r"\d")) & (pl.col("addr_p3").str.len_chars() >= 8)
).join(
    tgt_addr_p3.select(["entity_id", "country", "addr_p3"]).filter(
        (pl.col("addr_p3").str.contains(r"\d")) & (pl.col("addr_p3").str.len_chars() >= 8)
    ).rename({"entity_id": "target_id"}),
    on=["country", "addr_p3"], how="inner"
).select(["entity_id", "target_id"]).unique().with_columns(pl.lit(4).alias("priority"))

# Combine all 5 channels
all_pairs_raw = pl.concat([p1, p2, p3, p4, brand_pairs])
all_pairs = all_pairs_raw.group_by(["entity_id", "target_id"]).agg(pl.col("priority").min()).sort(["entity_id", "priority"])

print(f"Total Combined Pairs across 5 channels: {len(all_pairs):,}")

# Unbudgeted evaluation
grouped_all = all_pairs.group_by("entity_id").agg(pl.col("target_id").alias("candidates"))
cand_dict_all = {r["entity_id"]: set(r["candidates"]) for r in grouped_all.iter_rows(named=True)}
for s1_id in val_df["entity_id"].to_list():
    if s1_id not in cand_dict_all:
        cand_dict_all[s1_id] = set()
m_total = evaluator.evaluate(cand_dict_all, "All_5_Channels_Plus_Brand")

print("\n================ TOTAL COMBINED RECALL ================")
print(f"Overall Link Recall:      {m_total['overall_link_recall_pct']}%")
print(f"Entity Coverage:          {m_total['entity_coverage_pct']}%")
print(f"S2 Link Recall:           {m_total['s2_link_recall_pct']}%")
print(f"S3 Link Recall:           {m_total['s3_link_recall_pct']}%")
print(f"S2-only Entity Recall:    {m_total['s2_only_entity_recall_pct']}%")
print(f"S3-only Entity Recall:    {m_total['s3_only_entity_recall_pct']}%")
print(f"Both-source Entity Recall:{m_total['both_source_entity_recall_pct']}%")
print(f"Total Candidates:         {m_total['candidate_volume']['total_candidate_pairs']:,}")
print(f"Mean Candidates per S1:   {m_total['candidate_volume']['mean_per_s1']}")
print(f"P95 Candidates per S1:    {m_total['candidate_volume']['p95']}")
print(f"P99 Candidates per S1:    {m_total['candidate_volume']['p99']}")
print(f"Max Candidates per S1:    {m_total['candidate_volume']['max']}")
print("=======================================================")

# Budget evaluations (K=30, 40, 50)
for k in [30, 40, 50]:
    b_df = all_pairs.group_by("entity_id", maintain_order=True).head(k)
    g_b = b_df.group_by("entity_id").agg(pl.col("target_id").alias("candidates"))
    c_b = {r["entity_id"]: set(r["candidates"]) for r in g_b.iter_rows(named=True)}
    for s1_id in val_df["entity_id"].to_list():
        if s1_id not in c_b:
            c_b[s1_id] = set()
    m_k = evaluator.evaluate(c_b, f"Budget_K{k}")
    print(f"Top-K={k}: Link Recall = {m_k['overall_link_recall_pct']}%, Entity Coverage = {m_k['entity_coverage_pct']}%, Total Pairs = {m_k['candidate_volume']['total_candidate_pairs']:,} (avg {m_k['candidate_volume']['mean_per_s1']})")
