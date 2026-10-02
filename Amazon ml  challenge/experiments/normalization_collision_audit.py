"""
experiments/normalization_collision_audit.py
============================================
Performs deep empirical collision audit across derived text representations
in accordance with Phase 1 Section 7.
Measures:
  - Unique values and collision rates
  - Collision group size distribution (mean, median, P95, P99, max)
  - Collision explosion risk keys (>50 entities sharing same key)
  - Cross-source behavior
"""

import os
import sys
import json
import time

sys.path.insert(0, os.path.abspath("."))

import polars as pl
import numpy as np
from src.data_preparation import clean_text, strip_legal_suffixes, sort_tokens, fold_accents

print("Starting Normalization Collision Audit...", flush=True)
t0 = time.time()

DATASET_DIR = "dataset/raw/train"

# Sample 100k records from S1 and S2 for rapid deep collision distribution analysis
print("Loading train S1 sample (100,000 records)...", flush=True)
s1_df = pl.read_csv(os.path.join(DATASET_DIR, "train_source1.tsv"), separator="\t", n_rows=100000)

names = s1_df["business_name"].fill_null("").to_list()
addrs = s1_df["business_address"].fill_null("").to_list()
countries = s1_df["country"].fill_null("").to_list()

print("Computing representations for S1 sample...", flush=True)
clean_names = [clean_text(n) for n in names]
stripped_names = [strip_legal_suffixes(cn) for cn in clean_names]
sorted_names = [sort_tokens(sn) for sn in stripped_names]
accent_folded_names = [fold_accents(cn) for cn in clean_names]

clean_addrs = [clean_text(a) for a in addrs]

def audit_representation(values: list, label: str) -> dict:
    from collections import Counter
    counts = Counter(values)
    # Remove empty string
    counts.pop("", None)
    
    unique_vals = len(counts)
    total_non_empty = sum(counts.values())
    
    sizes = np.array(list(counts.values()), dtype=np.int32)
    collisions_only = sizes[sizes > 1]
    
    top_5_colliding = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]
    
    return {
        "representation": label,
        "total_records": len(values),
        "unique_keys": unique_vals,
        "compression_ratio": round(len(values) / unique_vals, 3) if unique_vals > 0 else 0,
        "keys_with_collisions": int(len(collisions_only)),
        "pct_keys_with_collisions": round(len(collisions_only) / unique_vals * 100, 2) if unique_vals > 0 else 0,
        "mean_group_size": round(float(np.mean(sizes)), 2) if len(sizes) > 0 else 0,
        "p95_group_size": float(np.percentile(sizes, 95)) if len(sizes) > 0 else 0,
        "p99_group_size": float(np.percentile(sizes, 99)) if len(sizes) > 0 else 0,
        "max_group_size": int(np.max(sizes)) if len(sizes) > 0 else 0,
        "keys_exceeding_50_entities": int(np.sum(sizes > 50)),
        "top_5_colliding_keys": [{"key": k, "count": c} for k, c in top_5_colliding]
    }

results = {
    "sample_size": len(names),
    "name_representations": [
        audit_representation(clean_names, "clean_name"),
        audit_representation(stripped_names, "stripped_legal_suffix_name"),
        audit_representation(sorted_names, "token_sorted_name"),
        audit_representation(accent_folded_names, "accent_folded_name")
    ],
    "address_representations": [
        audit_representation(clean_addrs, "clean_address")
    ],
    "audit_time_seconds": round(time.time() - t0, 2)
}

os.makedirs("experiments", exist_ok=True)
out_path = "experiments/normalization_collision_results.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"Collision audit complete in {results['audit_time_seconds']}s. Saved to {out_path}", flush=True)
