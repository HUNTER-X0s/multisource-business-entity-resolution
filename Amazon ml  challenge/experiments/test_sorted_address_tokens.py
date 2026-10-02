"""
experiments/test_sorted_address_tokens.py
==========================================
Tests a sorted address token set join to capture:
  - Reordered address tokens (address components in different order)
  - Transliteration pairs (India): English address vs script address with same tokens
  - Shuffled street/city/state order

Key insight from missed links analysis:
  Many India pairs have identical address tokens but in different ORDER.
  E.g., "Flt-1, Fl-Ground, Bl-87, Anandapur, Kolkata, West Bengal"
    vs "Howrah, WB, Flt-1, Fl-ground, Bl-87, Anandapur, Kolkata, Kolkata"

Strategy: Sort the content-bearing address words alphabetically -> reorder invariant key.
Filter to informative length (>= 10 chars sorted key) to avoid false positives.
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath("."))
import polars as pl
from src.candidate_evaluator import CandidateEvaluator
from src.vectorized_pipeline import normalize_dataframe

print("=" * 65, flush=True)
print("TESTING SORTED ADDRESS TOKEN JOIN", flush=True)
print("=" * 65, flush=True)

val_df = pl.read_parquet("experiments/val_sample_50k.parquet")
val_gt = {}
for row in val_df.iter_rows(named=True):
    s1_id = row["entity_id"]
    m_str = row.get("matched_entity_ids") or ""
    val_gt[s1_id] = set(m.strip() for m in m_str.split(",") if m.strip())

evaluator = CandidateEvaluator(val_gt)
targets_clean = pl.read_parquet("experiments/cache/targets_train_normalized.parquet")
val_clean = normalize_dataframe(val_df)

# Address normalization (same as benchmark)
STREET_ABBRS = {
    r"\bst\b": "street", r"\bave\b": "avenue", r"\brd\b": "road", r"\bdr\b": "drive",
    r"\bblvd\b": "boulevard", r"\bln\b": "lane",
    r"\bdelaware\b": "de", r"\bindiana\b": "in", r"\bnew york\b": "ny",
    r"\butah\b": "ut", r"\bvirginia\b": "va", r"\bmaine\b": "me",
    r"\bcalifornia\b": "ca", r"\btexas\b": "tx", r"\bflorida\b": "fl",
    r"\bohio\b": "oh", r"\bdelhi\b": "dl", r"\bwest bengal\b": "wb",
    r"\bkarnataka\b": "ka", r"\bmaharashtra\b": "mh",
    r"\bgujarat\b": "gj", r"\btamil nadu\b": "tn"
}

def standardize_address_expr(col_name: str) -> pl.Expr:
    expr = pl.col(col_name)
    for pat, rep in STREET_ABBRS.items():
        expr = expr.str.replace_all(pat, rep)
    expr = expr.str.replace_all(r"\b0+(\d+)\b", r"$1")
    return expr.str.replace_all(r"\s+", " ").str.strip_chars()

val_clean = val_clean.with_columns([standardize_address_expr("clean_address").alias("std_address")])
targets_clean = targets_clean.with_columns([standardize_address_expr("clean_address").alias("std_address")])

# Build sorted address token key:
# 1. Split std_address into words
# 2. Keep words of len >= 3 (exclude tiny tokens, 'n/a', 'null')
# 3. Filter out generic address words ('floor', 'unit', 'null', 'none', 'na')
ADDR_STOP = {"floor", "unit", "suite", "null", "none", "na", "and", "the"}

def sorted_addr_key_expr(col: str) -> pl.Expr:
    """Sorted bag of significant address tokens (word-order invariant)."""
    return (
        pl.col(col)
        .str.split(" ")
        .list.filter(
            (pl.element().str.len_chars() >= 3) &
            (~pl.element().is_in(list(ADDR_STOP)))
        )
        .list.sort()
        .list.join(" ")
    )

print("Building sorted address keys...", flush=True)
t0 = time.time()
val_sorted_addr = val_clean.with_columns([
    sorted_addr_key_expr("std_address").alias("sorted_addr")
])
tgt_sorted_addr = targets_clean.with_columns([
    sorted_addr_key_expr("std_address").alias("sorted_addr")
])
print(f"  Built in {time.time()-t0:.2f}s", flush=True)

# Filter to informative keys only (>=15 chars prevents single-word false positives)
MIN_SORTED_ADDR_LEN = 15

# Standalone sorted address channel evaluation
print("Running Sorted Address Join...", flush=True)
t1 = time.time()
p_sorted_addr = val_sorted_addr.select(["entity_id", "country", "sorted_addr"]).filter(
    pl.col("sorted_addr").str.len_chars() >= MIN_SORTED_ADDR_LEN
).join(
    tgt_sorted_addr.select(["entity_id", "country", "sorted_addr"]).filter(
        pl.col("sorted_addr").str.len_chars() >= MIN_SORTED_ADDR_LEN
    ).rename({"entity_id": "target_id"}),
    on=["country", "sorted_addr"], how="inner"
).select(["entity_id", "target_id"]).unique()
print(f"  Sorted Address Join: {len(p_sorted_addr):,} pairs in {time.time()-t1:.2f}s", flush=True)

grouped_sa = p_sorted_addr.group_by("entity_id").agg(pl.col("target_id").alias("candidates"))
cand_dict_sa = {r["entity_id"]: set(r["candidates"]) for r in grouped_sa.iter_rows(named=True)}
for s1_id in val_df["entity_id"].to_list():
    if s1_id not in cand_dict_sa:
        cand_dict_sa[s1_id] = set()
m_sa = evaluator.evaluate(cand_dict_sa, "Sorted_Address_Standalone")
print(f"  Sorted Address Standalone: Recall={m_sa['overall_link_recall_pct']}%, Coverage={m_sa['entity_coverage_pct']}%")
print(f"  Mean candidates/query: {m_sa['candidate_volume']['mean_per_s1']}, Max: {m_sa['candidate_volume']['max']}")

# Check precision of sorted address join (collision rate)
# How many S1 queries have >50 candidates from this channel alone?
cand_volume_series = [len(v) for v in cand_dict_sa.values()]
over_50 = sum(1 for c in cand_volume_series if c > 50)
print(f"  Queries with >50 sorted-address candidates: {over_50:,} ({100*over_50/len(val_df):.2f}%)")

# Now build the full 6-channel pipeline + sorted_addr and measure combined recall
print("\nBuilding full pipeline + Sorted Address...", flush=True)

# Channel 1: Exact Clean Name
p1 = val_clean.select(["entity_id", "country", "clean_name"]).filter(pl.col("clean_name") != "").join(
    targets_clean.select(["entity_id", "country", "clean_name"]).filter(pl.col("clean_name") != "").rename({"entity_id": "target_id"}),
    on=["country", "clean_name"], how="inner"
).select(["entity_id", "target_id"]).unique().with_columns(pl.lit(1.0).alias("prio"))

# Channel 2: Token-Sorted Name
p2 = val_clean.select(["entity_id", "country", "sorted_name"]).filter(pl.col("sorted_name") != "").join(
    targets_clean.select(["entity_id", "country", "sorted_name"]).filter(pl.col("sorted_name") != "").rename({"entity_id": "target_id"}),
    on=["country", "sorted_name"], how="inner"
).select(["entity_id", "target_id"]).unique().with_columns(pl.lit(0.95).alias("prio"))

# Channel 3: Standardized Exact Address
p3 = val_sorted_addr.select(["entity_id", "country", "std_address"]).filter(pl.col("std_address").str.len_chars() >= 10).join(
    tgt_sorted_addr.select(["entity_id", "country", "std_address"]).filter(pl.col("std_address").str.len_chars() >= 10).rename({"entity_id": "target_id"}),
    on=["country", "std_address"], how="inner"
).select(["entity_id", "target_id"]).unique().with_columns(pl.lit(0.90).alias("prio"))

# Channel 4: Street Prefix Match
val_addr_p3 = val_sorted_addr.with_columns([pl.col("std_address").str.split(" ").list.slice(0, 3).list.join(" ").alias("addr_p3")])
tgt_addr_p3 = tgt_sorted_addr.with_columns([pl.col("std_address").str.split(" ").list.slice(0, 3).list.join(" ").alias("addr_p3")])
p4 = val_addr_p3.select(["entity_id", "country", "addr_p3"]).filter(
    (pl.col("addr_p3").str.contains(r"\d")) & (pl.col("addr_p3").str.len_chars() >= 8)
).join(
    tgt_addr_p3.select(["entity_id", "country", "addr_p3"]).filter(
        (pl.col("addr_p3").str.contains(r"\d")) & (pl.col("addr_p3").str.len_chars() >= 8)
    ).rename({"entity_id": "target_id"}),
    on=["country", "addr_p3"], how="inner"
).select(["entity_id", "target_id"]).unique().with_columns(pl.lit(0.70).alias("prio"))

# Channel 5: 2-Word Name Bigram
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
p5 = val_rare_bi.join(tgt_rare_bi.rename({"entity_id": "target_id"}), on=["country", "name_bigram"], how="inner"
).select(["entity_id", "target_id"]).unique().with_columns(pl.lit(0.75).alias("prio"))

# Channel 6: Distinctive Brand Token
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
      .then(pl.col("raw_brand")).otherwise(pl.lit("")).alias("brand_token")
])
tgt_brand = targets_clean.with_columns([
    pl.col("stripped_name").str.split(" ").list.get(0).alias("raw_brand")
]).with_columns([
    pl.when((pl.col("raw_brand").str.len_chars() >= 4) & (~pl.col("raw_brand").is_in(list(GENERIC_WORDS))))
      .then(pl.col("raw_brand")).otherwise(pl.lit("")).alias("brand_token")
])
brand_counts = tgt_brand.filter(pl.col("brand_token") != "").group_by(["country", "brand_token"]).len()
rare_brands = brand_counts.filter(pl.col("len") <= 300).select(["country", "brand_token"])
val_rare_br = val_brand.select(["entity_id", "country", "brand_token"]).join(rare_brands, on=["country", "brand_token"], how="inner")
tgt_rare_br = tgt_brand.select(["entity_id", "country", "brand_token"]).join(rare_brands, on=["country", "brand_token"], how="inner")
p6 = val_rare_br.join(tgt_rare_br.rename({"entity_id": "target_id"}), on=["country", "brand_token"], how="inner"
).select(["entity_id", "target_id"]).unique().with_columns(pl.lit(0.60).alias("prio"))

# Channel 7: Sorted Address Token Set Join (new)
p7 = p_sorted_addr.with_columns(pl.lit(0.88).alias("prio"))

# Combine all 7 channels
all_raw = pl.concat([p1, p2, p3, p4, p5, p6, p7])
cand_all = all_raw.group_by(["entity_id", "target_id"]).agg([
    pl.col("prio").max().alias("max_prio"),
    pl.len().alias("n_channels")
])

print(f"All 7 channels unique pairs: {len(cand_all):,}", flush=True)

# Unbudgeted
grouped = cand_all.group_by("entity_id").agg(pl.col("target_id").alias("candidates"))
cand_dict = {r["entity_id"]: set(r["candidates"]) for r in grouped.iter_rows(named=True)}
for s1_id in val_df["entity_id"].to_list():
    if s1_id not in cand_dict:
        cand_dict[s1_id] = set()

m_total = evaluator.evaluate(cand_dict, "7_Channel_Unbudgeted")
print("\n======== 7-CHANNEL UNBUDGETED RESULTS ========")
print(f"  Link Recall:        {m_total['overall_link_recall_pct']}%")
print(f"  Entity Coverage:    {m_total['entity_coverage_pct']}%")
print(f"  S2 Recall:          {m_total['s2_link_recall_pct']}%")
print(f"  S3 Recall:          {m_total['s3_link_recall_pct']}%")
print(f"  Total Candidates:   {m_total['candidate_volume']['total_candidate_pairs']:,}")
print(f"  Mean/Query:         {m_total['candidate_volume']['mean_per_s1']}")
print(f"  P95/Query:          {m_total['candidate_volume']['p95']}")
print(f"  Max/Query:          {m_total['candidate_volume']['max']}")

# Budget evaluations
ranked = cand_all.sort(["entity_id", "max_prio", "n_channels"], descending=[False, True, True])
for k in [30, 40, 50]:
    b_df = ranked.group_by("entity_id", maintain_order=True).head(k)
    g_b = b_df.group_by("entity_id").agg(pl.col("target_id").alias("candidates"))
    c_b = {r["entity_id"]: set(r["candidates"]) for r in g_b.iter_rows(named=True)}
    for s1_id in val_df["entity_id"].to_list():
        if s1_id not in c_b:
            c_b[s1_id] = set()
    m_k = evaluator.evaluate(c_b, f"7ch_TopK_{k}")
    print(f"  K={k}: Recall={m_k['overall_link_recall_pct']}%, Coverage={m_k['entity_coverage_pct']}%, Pairs={m_k['candidate_volume']['total_candidate_pairs']:,}")

print("\nDone.", flush=True)
