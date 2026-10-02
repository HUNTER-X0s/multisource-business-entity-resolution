"""
experiments/test_2token_and_address_normalization.py
===================================================
Tests two critical retrieval boosters identified in forensic error analysis:
  1. Street suffix & state abbreviation normalization (e.g. 'st' -> 'street', 'delaware' -> 'de')
  2. 2-Token Co-occurrence Index (queries sharing >= 2 significant name words with target)
Measures Link Recall, Entity Coverage, and candidate volume on 50K validation queries.
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath("."))
import polars as pl
from src.candidate_evaluator import CandidateEvaluator
from src.vectorized_pipeline import normalize_dataframe

print("=" * 65, flush=True)
print("TESTING 2-TOKEN CO-OCCURRENCE & ADDRESS NORMALIZATION", flush=True)
print("=" * 65, flush=True)

# 1. Load validation queries
val_df = pl.read_parquet("experiments/val_sample_50k.parquet")
val_gt = {}
for row in val_df.iter_rows(named=True):
    s1_id = row["entity_id"]
    m_str = row.get("matched_entity_ids") or ""
    val_gt[s1_id] = set(m.strip() for m in m_str.split(",") if m.strip())

evaluator = CandidateEvaluator(val_gt)
targets_clean = pl.read_parquet("experiments/cache/targets_train_normalized.parquet")
val_clean = normalize_dataframe(val_df)

# Address abbreviations dictionary
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
    # Remove leading zeros from digits (e.g. 00450 -> 450)
    expr = expr.str.replace_all(r"\b0+(\d+)\b", r"$1")
    return expr.str.replace_all(r"\s+", " ").str.strip_chars()

print("Applying address standardization...", flush=True)
val_std = val_clean.with_columns([standardize_address_expr("clean_address").alias("std_address")])
tgt_std = targets_clean.with_columns([standardize_address_expr("clean_address").alias("std_address")])

# Channel A: Standardized Address Join
print("Running Standardized Address Join...", flush=True)
t_a = time.time()
p_addr_std = val_std.select(["entity_id", "country", "std_address"]).filter(pl.col("std_address").str.len_chars() >= 10).join(
    tgt_std.select(["entity_id", "country", "std_address"]).filter(pl.col("std_address").str.len_chars() >= 10).rename({"entity_id": "target_id"}),
    on=["country", "std_address"], how="inner"
).select(["entity_id", "target_id"]).unique().with_columns(pl.lit(3).alias("priority"))
print(f"  Standardized Address Join produced {len(p_addr_std):,} pairs in {time.time()-t_a:.2f}s")

# Channel B: 2-Token Name Pairs (First 2 Significant Words of Name)
# If a name has words [w1, w2, w3], pair (w1, w2) is an ultra-strong retrieval key!
print("Extracting (Word1, Word2) Name Bigram Keys...", flush=True)
t_b = time.time()

# Extract words with len >= 3
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

# Prune high frequency bigrams (> 500 target records)
bigram_counts = tgt_bigrams.filter(pl.col("name_bigram") != "").group_by(["country", "name_bigram"]).len()
rare_bigrams = bigram_counts.filter(pl.col("len") <= 500).select(["country", "name_bigram"])

print(f"Total bigram keys: {len(bigram_counts):,}")
print(f"Informative rare bigrams (freq <= 500): {len(rare_bigrams):,}")

val_rare_bi = val_bigrams.select(["entity_id", "country", "name_bigram"]).join(rare_bigrams, on=["country", "name_bigram"], how="inner")
tgt_rare_bi = tgt_bigrams.select(["entity_id", "country", "name_bigram"]).join(rare_bigrams, on=["country", "name_bigram"], how="inner")

print("Running Rare 2-Word Name Bigram Join...", flush=True)
t_bj = time.time()
p_bigram = val_rare_bi.join(
    tgt_rare_bi.rename({"entity_id": "target_id"}),
    on=["country", "name_bigram"],
    how="inner"
).select(["entity_id", "target_id"]).unique().with_columns(pl.lit(2).alias("priority"))
print(f"  2-Word Bigram Join produced {len(p_bigram):,} pairs in {time.time()-t_bj:.2f}s")

# Load other baseline channels: Exact Name, Sorted Name, Street Prefix
print("Combining all complementary channels...", flush=True)
p1 = val_clean.select(["entity_id", "country", "clean_name"]).filter(pl.col("clean_name") != "").join(
    targets_clean.select(["entity_id", "country", "clean_name"]).filter(pl.col("clean_name") != "").rename({"entity_id": "target_id"}),
    on=["country", "clean_name"], how="inner"
).select(["entity_id", "target_id"]).unique().with_columns(pl.lit(1).alias("priority"))

p2 = val_clean.select(["entity_id", "country", "sorted_name"]).filter(pl.col("sorted_name") != "").join(
    targets_clean.select(["entity_id", "country", "sorted_name"]).filter(pl.col("sorted_name") != "").rename({"entity_id": "target_id"}),
    on=["country", "sorted_name"], how="inner"
).select(["entity_id", "target_id"]).unique().with_columns(pl.lit(1).alias("priority"))

# Street prefix
val_addr_p3 = val_std.with_columns([pl.col("std_address").str.split(" ").list.slice(0, 3).list.join(" ").alias("addr_p3")])
tgt_addr_p3 = tgt_std.with_columns([pl.col("std_address").str.split(" ").list.slice(0, 3).list.join(" ").alias("addr_p3")])

p4 = val_addr_p3.select(["entity_id", "country", "addr_p3"]).filter(
    (pl.col("addr_p3").str.contains(r"\d")) & (pl.col("addr_p3").str.len_chars() >= 8)
).join(
    tgt_addr_p3.select(["entity_id", "country", "addr_p3"]).filter(
        (pl.col("addr_p3").str.contains(r"\d")) & (pl.col("addr_p3").str.len_chars() >= 8)
    ).rename({"entity_id": "target_id"}),
    on=["country", "addr_p3"], how="inner"
).select(["entity_id", "target_id"]).unique().with_columns(pl.lit(4).alias("priority"))

# Combine all channels
all_pairs_raw = pl.concat([p1, p2, p_bigram, p_addr_std, p4])
all_pairs = all_pairs_raw.group_by(["entity_id", "target_id"]).agg(pl.col("priority").min()).sort(["entity_id", "priority"])

print(f"\nTotal Multi-Channel Pairs: {len(all_pairs):,}")

# Unbudgeted evaluation
grouped_all = all_pairs.group_by("entity_id").agg(pl.col("target_id").alias("candidates"))
cand_dict_all = {r["entity_id"]: set(r["candidates"]) for r in grouped_all.iter_rows(named=True)}
for s1_id in val_df["entity_id"].to_list():
    if s1_id not in cand_dict_all:
        cand_dict_all[s1_id] = set()
m_total = evaluator.evaluate(cand_dict_all, "Multi_Channel_Bigram_AddrStd")

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
