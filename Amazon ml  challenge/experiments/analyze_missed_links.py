"""
experiments/analyze_missed_links.py
===================================
Forensic error analysis: inspects true links that were missed by exact clean
and token-sorted matching in EXP-1.3 to discover the actual data transformations
and determine the most effective secondary retrieval channels.
"""

import os
import sys

sys.path.insert(0, os.path.abspath("."))
import polars as pl

print("Analyzing missed true links from EXP-1.3...", flush=True)

# 1. Load validation queries
val_df = pl.read_parquet("experiments/val_sample_50k.parquet")
from src.vectorized_pipeline import normalize_dataframe

val_clean = normalize_dataframe(val_df)

# 2. Load cached normalized targets
targets_clean = pl.read_parquet("experiments/cache/targets_train_normalized.parquet")

# 3. Find pairs recovered by EXP-1.3 (Exact + Sorted)
p1 = val_clean.select(["entity_id", "country", "clean_name"]).filter(pl.col("clean_name") != "").join(
    targets_clean.select(["entity_id", "country", "clean_name"]).filter(pl.col("clean_name") != "").rename({"entity_id": "target_id"}),
    on=["country", "clean_name"],
    how="inner"
).select(["entity_id", "target_id"])

p2 = val_clean.select(["entity_id", "country", "sorted_name"]).filter(pl.col("sorted_name") != "").join(
    targets_clean.select(["entity_id", "country", "sorted_name"]).filter(pl.col("sorted_name") != "").rename({"entity_id": "target_id"}),
    on=["country", "sorted_name"],
    how="inner"
).select(["entity_id", "target_id"])

retrieved_set = set(zip(p1["entity_id"].to_list(), p1["target_id"].to_list())).union(
    set(zip(p2["entity_id"].to_list(), p2["target_id"].to_list()))
)

# 4. Find true links that were MISSED
missed_pairs = []
for row in val_df.iter_rows(named=True):
    s1_id = row["entity_id"]
    m_str = row.get("matched_entity_ids") or ""
    if not m_str:
        continue
    for m in m_str.split(","):
        m = m.strip()
        if not m:
            continue
        if (s1_id, m) not in retrieved_set:
            missed_pairs.append((s1_id, m))

print(f"Total true links evaluated: {sum(len(r.get('matched_entity_ids', '').split(',')) for r in val_df.iter_rows(named=True) if r.get('matched_entity_ids'))}")
print(f"Missed true links: {len(missed_pairs):,}")

# Sample 20 missed pairs and look up their details
print("\n=== SAMPLE OF 20 MISSED TRUE LINKS ===")
# Build quick lookup dicts
s1_lookup = {r["entity_id"]: r for r in val_clean.iter_rows(named=True)}

# We need target details for missed targets
sample_targets = set(m for _, m in missed_pairs[:30])
target_sub = targets_clean.filter(pl.col("entity_id").is_in(list(sample_targets)))
target_lookup = {r["entity_id"]: r for r in target_sub.iter_rows(named=True)}

# Write sample of missed pairs to UTF-8 file
out_file = "experiments/missed_links_sample.txt"
with open(out_file, "w", encoding="utf-8") as f:
    f.write("=== SAMPLE OF MISSED TRUE LINKS (EXP-1.3) ===\n\n")
    shown = 0
    for s1_id, t_id in missed_pairs:
        if t_id in target_lookup:
            s1 = s1_lookup[s1_id]
            tgt = target_lookup[t_id]
            f.write(f"[{shown+1}] Pair: ({s1_id} -> {t_id}) | Country: {s1['country']}\n")
            f.write(f"   S1 Name:     {s1['business_name']}\n")
            f.write(f"   Target Name: {tgt['business_name']}\n")
            f.write(f"   S1 Addr:     {s1['business_address']}\n")
            f.write(f"   Target Addr: {tgt['business_address']}\n")
            f.write(f"   S1 Sorted:   {s1['sorted_name']}\n")
            f.write(f"   Tgt Sorted:  {tgt['sorted_name']}\n")
            f.write(f"   S1 Postal:   {s1['postal_code']} | Tgt Postal: {tgt['postal_code']}\n")
            f.write("-" * 65 + "\n")
            shown += 1
            if shown >= 25:
                break

print(f"Sample of {shown} missed pairs written to {out_file}", flush=True)
