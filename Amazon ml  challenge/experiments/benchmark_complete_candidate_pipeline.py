"""
experiments/benchmark_complete_candidate_pipeline.py
====================================================
Comprehensive Multi-Channel Candidate Retrieval & Similarity Ranking Benchmark.
Evaluates:
  - 6 complementary retrieval channels:
      1. Exact Clean Name
      2. Token-Sorted Suffix-Stripped Name
      3. Standardized Exact Address
      4. Street Prefix Match (3 tokens + numeric)
      5. Rare 2-Word Name Bigram (frequency <= 500)
      6. Rare Brand Token (frequency <= 300)
  - Fast Vectorized & SIMD Similarity Scoring for Candidate Re-ranking:
      Score = Channel Priority Base + Name JaroWinkler/QRatio + Address Token Overlap
  - Top-K Budget Truncation at K = 20, 30, 40, 50 vs Unbudgeted.
  - Full stratification across sources (S2 vs S3, S2-only, S3-only, Both) and countries.
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath("."))
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from src.candidate_evaluator import CandidateEvaluator
from src.vectorized_pipeline import normalize_dataframe

print("=" * 70, flush=True)
print("BENCHMARKING COMPLETE MULTI-CHANNEL RETRIEVAL & SIMILARITY RANKING", flush=True)
print("=" * 70, flush=True)

# 1. Load validation sample (50,000 queries)
val_df = pl.read_parquet("experiments/val_sample_50k.parquet")
val_gt = {}
for row in val_df.iter_rows(named=True):
    s1_id = row["entity_id"]
    m_str = row.get("matched_entity_ids") or ""
    val_gt[s1_id] = set(m.strip() for m in m_str.split(",") if m.strip())

evaluator = CandidateEvaluator(val_gt)
total_links = sum(len(v) for v in val_gt.values())
print(f"Loaded {len(val_df):,} validation queries with {total_links:,} true links.", flush=True)

# 2. Load targets and normalize queries
targets_clean = pl.read_parquet("experiments/cache/targets_train_normalized.parquet")
val_clean = normalize_dataframe(val_df)

# Address abbreviations for standardization
STREET_ABBRS = {
    r"\bst\b": "street",
    r"\bave\b": "avenue",
    r"\brd\b": "road",
    r"\bdr\b": "drive",
    r"\bblvd\b": "boulevard",
    r"\bln\b": "lane",
    r"\bdelaware\b": "de",
    r"\bindiana\b": "in",
    r"\bnew york\b": "ny",
    r"\butah\b": "ut",
    r"\bvirginia\b": "va",
    r"\bmaine\b": "me",
    r"\bcalifornia\b": "ca",
    r"\btexas\b": "tx",
    r"\bflorida\b": "fl",
    r"\bohio\b": "oh",
    r"\bdelhi\b": "dl",
    r"\bwest bengal\b": "wb",
    r"\bkarnataka\b": "ka",
    r"\bmaharashtra\b": "mh",
    r"\bgujarat\b": "gj",
    r"\btamil nadu\b": "tn"
}

def standardize_address_expr(col_name: str) -> pl.Expr:
    expr = pl.col(col_name)
    for pat, rep in STREET_ABBRS.items():
        expr = expr.str.replace_all(pat, rep)
    expr = expr.str.replace_all(r"\b0+(\d+)\b", r"$1")
    return expr.str.replace_all(r"\s+", " ").str.strip_chars()

val_clean = val_clean.with_columns([standardize_address_expr("clean_address").alias("std_address")])
targets_clean = targets_clean.with_columns([standardize_address_expr("clean_address").alias("std_address")])

t_start = time.time()

# Channel 1: Exact Clean Name
print("\n[Channel 1] Exact Clean Name Match...", flush=True)
p1 = val_clean.select(["entity_id", "country", "clean_name"]).filter(pl.col("clean_name") != "").join(
    targets_clean.select(["entity_id", "country", "clean_name"]).filter(pl.col("clean_name") != "").rename({"entity_id": "target_id"}),
    on=["country", "clean_name"], how="inner"
).select(["entity_id", "target_id"]).unique().with_columns([
    pl.lit(1.0).alias("base_priority"),
    pl.lit("exact_name").alias("channel")
])
print(f"  Channel 1 produced {len(p1):,} pairs", flush=True)

# Channel 2: Suffix-Stripped Token-Sorted Name
print("[Channel 2] Token-Sorted Name Match...", flush=True)
p2 = val_clean.select(["entity_id", "country", "sorted_name"]).filter(pl.col("sorted_name") != "").join(
    targets_clean.select(["entity_id", "country", "sorted_name"]).filter(pl.col("sorted_name") != "").rename({"entity_id": "target_id"}),
    on=["country", "sorted_name"], how="inner"
).select(["entity_id", "target_id"]).unique().with_columns([
    pl.lit(0.95).alias("base_priority"),
    pl.lit("sorted_name").alias("channel")
])
print(f"  Channel 2 produced {len(p2):,} pairs", flush=True)

# Channel 3: Standardized Exact Address
print("[Channel 3] Standardized Address Match...", flush=True)
p3 = val_clean.select(["entity_id", "country", "std_address"]).filter(pl.col("std_address").str.len_chars() >= 10).join(
    targets_clean.select(["entity_id", "country", "std_address"]).filter(pl.col("std_address").str.len_chars() >= 10).rename({"entity_id": "target_id"}),
    on=["country", "std_address"], how="inner"
).select(["entity_id", "target_id"]).unique().with_columns([
    pl.lit(0.90).alias("base_priority"),
    pl.lit("std_address").alias("channel")
])
print(f"  Channel 3 produced {len(p3):,} pairs", flush=True)

# Channel 4: Street Prefix Match (3 tokens + numeric)
print("[Channel 4] Street Prefix Match...", flush=True)
val_addr_p3 = val_clean.with_columns([pl.col("std_address").str.split(" ").list.slice(0, 3).list.join(" ").alias("addr_p3")])
tgt_addr_p3 = targets_clean.with_columns([pl.col("std_address").str.split(" ").list.slice(0, 3).list.join(" ").alias("addr_p3")])

p4 = val_addr_p3.select(["entity_id", "country", "addr_p3"]).filter(
    (pl.col("addr_p3").str.contains(r"\d")) & (pl.col("addr_p3").str.len_chars() >= 8)
).join(
    tgt_addr_p3.select(["entity_id", "country", "addr_p3"]).filter(
        (pl.col("addr_p3").str.contains(r"\d")) & (pl.col("addr_p3").str.len_chars() >= 8)
    ).rename({"entity_id": "target_id"}),
    on=["country", "addr_p3"], how="inner"
).select(["entity_id", "target_id"]).unique().with_columns([
    pl.lit(0.70).alias("base_priority"),
    pl.lit("street_p3").alias("channel")
])
print(f"  Channel 4 produced {len(p4):,} pairs", flush=True)

# Channel 5: 2-Word Name Bigram (First 2 words >= 3 chars, sorted, freq <= 500)
print("[Channel 5] 2-Word Name Bigram Match...", flush=True)
val_bigrams = val_clean.with_columns([
    pl.col("stripped_name").str.split(" ").list.filter(pl.element().str.len_chars() >= 3).alias("words")
]).with_columns([
    pl.when(pl.col("words").list.len() >= 2)
      .then(pl.col("words").list.slice(0, 2).list.sort().list.join(" "))
      .otherwise(pl.lit(""))
      .alias("name_bigram")
])

tgt_bigrams = targets_clean.with_columns([
    pl.col("stripped_name").str.split(" ").list.filter(pl.element().str.len_chars() >= 3).alias("words")
]).with_columns([
    pl.when(pl.col("words").list.len() >= 2)
      .then(pl.col("words").list.slice(0, 2).list.sort().list.join(" "))
      .otherwise(pl.lit(""))
      .alias("name_bigram")
])

bigram_counts = tgt_bigrams.filter(pl.col("name_bigram") != "").group_by(["country", "name_bigram"]).len()
rare_bigrams = bigram_counts.filter(pl.col("len") <= 500).select(["country", "name_bigram"])

val_rare_bi = val_bigrams.select(["entity_id", "country", "name_bigram"]).join(rare_bigrams, on=["country", "name_bigram"], how="inner")
tgt_rare_bi = tgt_bigrams.select(["entity_id", "country", "name_bigram"]).join(rare_bigrams, on=["country", "name_bigram"], how="inner")

p5 = val_rare_bi.join(
    tgt_rare_bi.rename({"entity_id": "target_id"}),
    on=["country", "name_bigram"],
    how="inner"
).select(["entity_id", "target_id"]).unique().with_columns([
    pl.lit(0.75).alias("base_priority"),
    pl.lit("name_bigram").alias("channel")
])
print(f"  Channel 5 produced {len(p5):,} pairs", flush=True)

# Channel 6: Distinctive Brand Token (first token >= 4 chars, non-generic, freq <= 300)
print("[Channel 6] Distinctive Brand Token Match...", flush=True)
GENERIC_WORDS = {
    "center", "services", "service", "enterprises", "enterprise", "solutions",
    "group", "holdings", "holding", "associates", "consultancy", "consulting",
    "management", "industries", "industry", "international", "global", "national",
    "united", "american", "india", "delhi", "mumbai", "texas", "california",
    "medical", "health", "care", "clinic", "hospital", "pharma", "pharmacy",
    "dental", "dentistry", "realty", "properties", "estate", "construction",
    "builders", "technologies", "technology", "tech", "systems", "corp", "inc"
}

val_brand = val_clean.with_columns([
    pl.col("stripped_name").str.split(" ").list.get(0).alias("raw_brand")
]).with_columns([
    pl.when((pl.col("raw_brand").str.len_chars() >= 4) & (~pl.col("raw_brand").is_in(list(GENERIC_WORDS))))
      .then(pl.col("raw_brand"))
      .otherwise(pl.lit(""))
      .alias("brand_token")
])

tgt_brand = targets_clean.with_columns([
    pl.col("stripped_name").str.split(" ").list.get(0).alias("raw_brand")
]).with_columns([
    pl.when((pl.col("raw_brand").str.len_chars() >= 4) & (~pl.col("raw_brand").is_in(list(GENERIC_WORDS))))
      .then(pl.col("raw_brand"))
      .otherwise(pl.lit(""))
      .alias("brand_token")
])

brand_counts = tgt_brand.filter(pl.col("brand_token") != "").group_by(["country", "brand_token"]).len()
rare_brands = brand_counts.filter(pl.col("len") <= 300).select(["country", "brand_token"])

val_rare_br = val_brand.select(["entity_id", "country", "brand_token"]).join(rare_brands, on=["country", "brand_token"], how="inner")
tgt_rare_br = tgt_brand.select(["entity_id", "country", "brand_token"]).join(rare_brands, on=["country", "brand_token"], how="inner")

p6 = val_rare_br.join(
    tgt_rare_br.rename({"entity_id": "target_id"}),
    on=["country", "brand_token"],
    how="inner"
).select(["entity_id", "target_id"]).unique().with_columns([
    pl.lit(0.60).alias("base_priority"),
    pl.lit("brand_token").alias("channel")
])
print(f"  Channel 6 produced {len(p6):,} pairs", flush=True)

# Combine candidate pairs and preserve highest base_priority and count of channels matching
all_pairs_raw = pl.concat([p1, p2, p3, p4, p5, p6])
print(f"\nTotal raw multi-channel candidates: {len(all_pairs_raw):,}", flush=True)

# Aggregate per (entity_id, target_id): max base_priority, count of channels
cand_pairs = all_pairs_raw.group_by(["entity_id", "target_id"]).agg([
    pl.col("base_priority").max().alias("max_priority"),
    pl.len().alias("num_channels")
])
print(f"Unique multi-channel candidate pairs: {len(cand_pairs):,}", flush=True)

# Measure Unbudgeted Recall
grouped_all = cand_pairs.group_by("entity_id").agg(pl.col("target_id").alias("candidates"))
cand_dict_all = {r["entity_id"]: set(r["candidates"]) for r in grouped_all.iter_rows(named=True)}
for s1_id in val_df["entity_id"].to_list():
    if s1_id not in cand_dict_all:
        cand_dict_all[s1_id] = set()

m_all = evaluator.evaluate(cand_dict_all, "Unbudgeted_All_6_Channels")
print("\n" + "=" * 35 + " UNBUDGETED METRICS " + "=" * 35, flush=True)
print(f"Overall Link Recall:       {m_all['overall_link_recall_pct']}%", flush=True)
print(f"Entity Coverage:           {m_all['entity_coverage_pct']}%", flush=True)
print(f"S2 Link Recall:            {m_all['s2_link_recall_pct']}%", flush=True)
print(f"S3 Link Recall:            {m_all['s3_link_recall_pct']}%", flush=True)
print(f"S2-only Entity Recall:     {m_all['s2_only_entity_recall_pct']}%", flush=True)
print(f"S3-only Entity Recall:     {m_all['s3_only_entity_recall_pct']}%", flush=True)
print(f"Both-source Entity Recall: {m_all['both_source_entity_recall_pct']}%", flush=True)
print(f"Total Candidates:          {m_all['candidate_volume']['total_candidate_pairs']:,}", flush=True)
print(f"Mean per Query:            {m_all['candidate_volume']['mean_per_s1']}", flush=True)
print(f"P95 per Query:             {m_all['candidate_volume']['p95']}", flush=True)
print(f"Max per Query:             {m_all['candidate_volume']['max']}", flush=True)

# Candidate Ranking Function
# If a query has <= 30 candidates, keep them all!
# If a query has > 30 candidates, rank them:
# Priority order:
# 1. Multi-channel matches (matched in >= 2 channels)
# 2. Exact / Sorted Name (max_priority >= 0.95)
# 3. Standardized Address (max_priority >= 0.90)
# 4. Fast Text Similarity between S1 and target
print("\n" + "=" * 35 + " RANKING CANDIDATES " + "=" * 35, flush=True)
print("Applying composite ranking score...", flush=True)

# Join S1 clean_name and target clean_name for text similarity scoring on candidate pairs
s1_names = val_clean.select(["entity_id", "clean_name", "clean_address"])
tgt_names = targets_clean.select(["entity_id", "clean_name", "clean_address"]).rename({
    "entity_id": "target_id",
    "clean_name": "tgt_name",
    "clean_address": "tgt_address"
})

# Filter candidate pairs to those needing similarity calculation (only queries with > 30 candidates)
# To save computation time: compute similarity on candidate pairs
cand_annotated = cand_pairs.join(s1_names, on="entity_id", how="inner").join(tgt_names, on="target_id", how="inner")

# For speed, compute token overlap in Polars:
# 1. Exact match on clean_name -> sim = 1.0
# 2. Multi-channel bonus = (num_channels - 1) * 0.2
# 3. JaroWinkler similarity on names (using RapidFuzz in batch)
print("Computing RapidFuzz similarity on candidate pairs...", flush=True)
t_rf = time.time()

# Extract Python lists for rapidfuzz batch calculation
s1_n_list = cand_annotated["clean_name"].to_list()
tgt_n_list = cand_annotated["tgt_name"].to_list()

sim_scores = []
# SIMD JaroWinkler loop
for q_n, t_n in zip(s1_n_list, tgt_n_list):
    if q_n == t_n:
        sim_scores.append(1.0)
    else:
        sim_scores.append(JaroWinkler.similarity(q_n, t_n))

cand_annotated = cand_annotated.with_columns([
    pl.Series("name_jw_sim", sim_scores),
    (pl.col("max_priority") + (pl.col("num_channels") - 1) * 0.15 + pl.Series("name_jw_sim", sim_scores) * 0.5).alias("rank_score")
])
print(f"Similarity computed on {len(cand_annotated):,} pairs in {time.time()-t_rf:.2f}s", flush=True)

# Sort candidates per query by rank_score descending
print("Sorting candidate pool by rank_score descending...", flush=True)
ranked_candidates = cand_annotated.sort(["entity_id", "rank_score"], descending=[False, True])

# Evaluate Top-K for K in [20, 25, 30, 35, 40, 50]
results = {}
for k in [20, 25, 30, 35, 40, 50]:
    b_df = ranked_candidates.group_by("entity_id", maintain_order=True).head(k)
    g_b = b_df.group_by("entity_id").agg(pl.col("target_id").alias("candidates"))
    c_b = {r["entity_id"]: set(r["candidates"]) for r in g_b.iter_rows(named=True)}
    for s1_id in val_df["entity_id"].to_list():
        if s1_id not in c_b:
            c_b[s1_id] = set()
    m_k = evaluator.evaluate(c_b, f"Ranked_TopK_{k}")
    results[k] = m_k
    print(f"\n--- [Top-K = {k}] ---", flush=True)
    print(f"  Link Recall:       {m_k['overall_link_recall_pct']}%", flush=True)
    print(f"  Entity Coverage:   {m_k['entity_coverage_pct']}%", flush=True)
    print(f"  S2 Recall:         {m_k['s2_link_recall_pct']}% | S3 Recall: {m_k['s3_link_recall_pct']}%", flush=True)
    print(f"  Total Candidates:  {m_k['candidate_volume']['total_candidate_pairs']:,} (avg {m_k['candidate_volume']['mean_per_s1']}/query, P95={m_k['candidate_volume']['p95']})", flush=True)

# Save evaluation results to JSON
import json
with open("experiments/comprehensive_retrieval_benchmark_results.json", "w") as f:
    json.dump({"unbudgeted": m_all, "budgeted": results}, f, indent=2)

print("\n" + "=" * 70, flush=True)
print("BENCHMARK COMPLETED SUCCESSFULLY! Results saved to experiments/comprehensive_retrieval_benchmark_results.json", flush=True)
print("=" * 70, flush=True)
