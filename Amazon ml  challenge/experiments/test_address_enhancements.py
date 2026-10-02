"""
experiments/test_address_enhancements.py
========================================
Tests address enhancements to capture transliterated / inverted address pairs:
  1. Address prefix stripping (HN, Door No, Flat No, Plot No)
  2. Distinctive address bigram (street name + city/state)
  3. Numeric token + distinctive word join
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath("."))
import polars as pl
from src.candidate_evaluator import CandidateEvaluator
from src.vectorized_pipeline import normalize_dataframe

val_df = pl.read_parquet("experiments/val_sample_50k.parquet")
val_gt = {}
for row in val_df.iter_rows(named=True):
    s1_id = row["entity_id"]
    m_str = row.get("matched_entity_ids") or ""
    val_gt[s1_id] = set(m.strip() for m in m_str.split(",") if m.strip())

evaluator = CandidateEvaluator(val_gt)
targets_clean = pl.read_parquet("experiments/cache/targets_train_normalized.parquet")
val_clean = normalize_dataframe(val_df)

STREET_ABBRS = {
    r"\bst\b": "street", r"\bave\b": "avenue", r"\brd\b": "road", r"\bdr\b": "drive",
    r"\bblvd\b": "boulevard", r"\bln\b": "lane", r"\bdelaware\b": "de", r"\bindiana\b": "in",
    r"\bnew york\b": "ny", r"\butah\b": "ut", r"\bvirginia\b": "va", r"\bmaine\b": "me",
    r"\bcalifornia\b": "ca", r"\btexas\b": "tx", r"\bflorida\b": "fl", r"\bohio\b": "oh",
    r"\bdelhi\b": "dl", r"\bwest bengal\b": "wb", r"\bkarnataka\b": "ka", r"\bmaharashtra\b": "mh",
    r"\bgujarat\b": "gj", r"\btamil nadu\b": "tn"
}

def standardize_address_expr(col_name: str) -> pl.Expr:
    expr = pl.col(col_name)
    # Strip common house/flat prefixes
    expr = expr.str.replace_all(r"\b(door no|hn|h no|hno|flat no|plot no)\b\s*\d*[-/]?\d*", " ")
    for pat, rep in STREET_ABBRS.items():
        expr = expr.str.replace_all(pat, rep)
    expr = expr.str.replace_all(r"\b0+(\d+)\b", r"$1")
    return expr.str.replace_all(r"\s+", " ").str.strip_chars()

val_clean = val_clean.with_columns([standardize_address_expr("clean_address").alias("std_address")])
targets_clean = targets_clean.with_columns([standardize_address_expr("clean_address").alias("std_address")])

# Channel: Street Prefix on stripped std_address
val_addr_p3 = val_clean.with_columns([pl.col("std_address").str.split(" ").list.slice(0, 3).list.join(" ").alias("addr_p3")])
tgt_addr_p3 = targets_clean.with_columns([pl.col("std_address").str.split(" ").list.slice(0, 3).list.join(" ").alias("addr_p3")])

p_addr_p3 = val_addr_p3.select(["entity_id", "country", "addr_p3"]).filter(
    (pl.col("addr_p3").str.len_chars() >= 8)
).join(
    tgt_addr_p3.select(["entity_id", "country", "addr_p3"]).filter(
        (pl.col("addr_p3").str.len_chars() >= 8)
    ).rename({"entity_id": "target_id"}),
    on=["country", "addr_p3"], how="inner"
).select(["entity_id", "target_id"]).unique()

print(f"Address Prefix with HN stripped produced {len(p_addr_p3):,} pairs")

grouped = p_addr_p3.group_by("entity_id").agg(pl.col("target_id").alias("candidates"))
cand_dict = {r["entity_id"]: set(r["candidates"]) for r in grouped.iter_rows(named=True)}
for s1_id in val_df["entity_id"].to_list():
    if s1_id not in cand_dict:
        cand_dict[s1_id] = set()

m = evaluator.evaluate(cand_dict, "Stripped_HN_Address_Prefix")
print(f"Recall: {m['overall_link_recall_pct']}%, Coverage: {m['entity_coverage_pct']}%")
