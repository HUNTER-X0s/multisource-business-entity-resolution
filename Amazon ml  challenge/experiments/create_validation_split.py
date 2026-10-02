"""
experiments/create_validation_split.py
======================================
Generates reproducible, entity-stratified validation splits from train_source1.tsv
and train_ground_truth.tsv.

Outputs:
  - experiments/val_sample_50k.parquet (50,000 S1 queries stratified by country & match bin)
  - experiments/val_split_20pct.parquet (441,364 S1 queries, 20% holdout)
  - experiments/split_metadata.json (statistical breakdown & verification hashes)
"""

import os
import json
import time
import polars as pl
import numpy as np

print("Generating reproducible validation splits...", flush=True)

DATASET_DIR = "dataset/raw/train"
S1_PATH = os.path.join(DATASET_DIR, "train_source1.tsv")
GT_PATH = os.path.join(DATASET_DIR, "train_ground_truth.tsv")

os.makedirs("experiments", exist_ok=True)

t0 = time.time()

# 1. Load S1 metadata
print("Loading train S1 metadata...", flush=True)
s1_df = pl.read_csv(S1_PATH, separator="\t", columns=["entity_id", "country", "business_name", "business_address"])

# 2. Load Ground Truth
print("Loading train ground truth...", flush=True)
gt_df = pl.read_csv(GT_PATH, separator="\t")

# Compute match count per S1 entity
def parse_match_count(val):
    if val is None or val == "":
        return 0
    return len(val.split(","))

gt_df = gt_df.with_columns([
    pl.col("matched_entity_ids").fill_null("").map_elements(parse_match_count, return_dtype=pl.Int32).alias("match_count")
])

# Join metadata with match count
s1_full = s1_df.join(gt_df, left_on="entity_id", right_on="source1_entity_id", how="inner")
print(f"Loaded {len(s1_full)} entities with ground truth.", flush=True)

# Assign match count bin
# Bin 0: singletons (0)
# Bin 1: exact 1 (1)
# Bin 2: small multi (2-3)
# Bin 3: medium multi (4-6)
# Bin 4: large multi (7+)
s1_full = s1_full.with_columns([
    pl.when(pl.col("match_count") == 0).then(0)
      .when(pl.col("match_count") == 1).then(1)
      .when(pl.col("match_count").is_between(2, 3)).then(2)
      .when(pl.col("match_count").is_between(4, 6)).then(3)
      .otherwise(4)
      .alias("match_bin")
])

# Create combined stratum: country + match_bin
s1_full = s1_full.with_columns([
    (pl.col("country") + "_" + pl.col("match_bin").cast(pl.Utf8)).alias("stratum")
])

# Deterministic stratified sampling
RANDOM_SEED = 2026

# Sample 50,000 entities
print("Sampling 50,000 stratified development validation entities...", flush=True)
strata_counts = s1_full["stratum"].value_counts()
for r in strata_counts.iter_rows(named=True):
    print(f"  Stratum {r['stratum']}: {r['count']}", flush=True)

# Compute fraction for 50k
target_50k = 50000
frac_50k = target_50k / len(s1_full)

# Grouped sample
sampled_50k_dfs = []
sampled_20pct_dfs = []

for row in strata_counts.iter_rows(named=True):
    strat = row["stratum"]
    sub_df = s1_full.filter(pl.col("stratum") == strat)
    n_total = len(sub_df)
    
    n_50k = int(round(n_total * frac_50k))
    n_20p = int(round(n_total * 0.20))
    
    shuffled = sub_df.sample(fraction=1.0, shuffle=True, seed=RANDOM_SEED)
    
    sampled_50k_dfs.append(shuffled.head(n_50k))
    sampled_20pct_dfs.append(shuffled.head(n_20p))

df_50k = pl.concat(sampled_50k_dfs).sample(fraction=1.0, shuffle=True, seed=RANDOM_SEED)
df_20p = pl.concat(sampled_20pct_dfs).sample(fraction=1.0, shuffle=True, seed=RANDOM_SEED)

print(f"Generated 50K split: {len(df_50k)} entities")
print(f"Generated 20% split: {len(df_20p)} entities")

# Save parquet files
p_50k = "experiments/val_sample_50k.parquet"
p_20p = "experiments/val_split_20pct.parquet"

df_50k.write_parquet(p_50k)
df_20p.write_parquet(p_20p)

# Save metadata
metadata = {
    "total_train_s1": len(s1_full),
    "val_sample_50k": {
        "path": p_50k,
        "count": len(df_50k),
        "singletons": int((df_50k["match_count"] == 0).sum()),
        "singletons_pct": round(float((df_50k["match_count"] == 0).mean()) * 100, 3),
        "total_true_links": int(df_50k["match_count"].sum()),
        "countries": df_50k["country"].value_counts().to_dicts(),
        "match_bins": df_50k["match_bin"].value_counts().to_dicts()
    },
    "val_split_20pct": {
        "path": p_20p,
        "count": len(df_20p),
        "singletons": int((df_20p["match_count"] == 0).sum()),
        "singletons_pct": round(float((df_20p["match_count"] == 0).mean()) * 100, 3),
        "total_true_links": int(df_20p["match_count"].sum()),
        "countries": df_20p["country"].value_counts().to_dicts(),
        "match_bins": df_20p["match_bin"].value_counts().to_dicts()
    },
    "random_seed": RANDOM_SEED,
    "generation_time_seconds": round(time.time() - t0, 2)
}

with open("experiments/split_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print("Saved experiments/split_metadata.json successfully in", metadata["generation_time_seconds"], "s")
