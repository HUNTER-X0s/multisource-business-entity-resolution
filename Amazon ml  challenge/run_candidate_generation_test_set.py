"""
run_candidate_generation_test_set.py
=====================================
Production runner: generates candidate_pairs.tsv for the full test set.

Inputs (competition test data):
  dataset/raw/test/test_source1.tsv  — 1,732,544 S1 queries
  dataset/raw/test/test_source2.tsv  — 4,887,273 S2 targets
  dataset/raw/test/test_source3.tsv  — 5,082,316 S3 targets
  Total targets: 9,969,589

Output:
  output/candidate_pairs.tsv  — competition-format candidate file

Usage:
  python run_candidate_generation_test_set.py [--top-k 40]
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath("."))
import polars as pl

from src.candidate_generation import generate_candidates_and_save

def main():
    parser = argparse.ArgumentParser(description="Generate candidate pairs for full test set.")
    parser.add_argument("--top-k", type=int, default=40, help="Max candidates per S1 query (default 40).")
    parser.add_argument("--bigram-freq", type=int, default=500, help="Bigram frequency cutoff (default 500).")
    parser.add_argument("--brand-freq", type=int, default=300, help="Brand token frequency cutoff (default 300).")
    parser.add_argument("--output", type=str, default="output/candidate_pairs.tsv", help="Output path.")
    args = parser.parse_args()

    print("=" * 70, flush=True)
    print("AMAZON ML CHALLENGE 2026 — CANDIDATE GENERATION (FULL TEST SET)", flush=True)
    print("=" * 70, flush=True)
    print(f"Top-K: {args.top_k}", flush=True)
    print(f"Bigram freq cutoff: {args.bigram_freq}", flush=True)
    print(f"Brand token freq cutoff: {args.brand_freq}", flush=True)
    print(f"Output: {args.output}", flush=True)
    print()

    t0 = time.time()

    # Load S1 queries
    print("Loading test_source1.tsv ...", flush=True)
    s1 = pl.read_csv(
        "dataset/raw/test/test_source1.tsv",
        separator="\t",
        infer_schema_length=0,
        null_values=["N/A", "n/a", "null", "NULL", "None"],
    )
    print(f"  Loaded {len(s1):,} S1 queries in {time.time()-t0:.1f}s", flush=True)

    # Load S2 targets
    t1 = time.time()
    print("Loading test_source2.tsv ...", flush=True)
    s2 = pl.read_csv(
        "dataset/raw/test/test_source2.tsv",
        separator="\t",
        infer_schema_length=0,
        null_values=["N/A", "n/a", "null", "NULL", "None"],
    )
    print(f"  Loaded {len(s2):,} S2 targets in {time.time()-t1:.1f}s", flush=True)

    # Load S3 targets
    t2 = time.time()
    print("Loading test_source3.tsv ...", flush=True)
    s3 = pl.read_csv(
        "dataset/raw/test/test_source3.tsv",
        separator="\t",
        infer_schema_length=0,
        null_values=["N/A", "n/a", "null", "NULL", "None"],
    )
    print(f"  Loaded {len(s3):,} S3 targets in {time.time()-t2:.1f}s", flush=True)

    # Combine S2 and S3 into single target pool
    targets = pl.concat([s2, s3])
    print(f"\nTotal target pool: {len(targets):,} records (S2 + S3)", flush=True)

    # Ensure required columns exist
    required_cols = {"entity_id", "business_name", "business_address", "country"}
    for df_name, df in [("S1", s1), ("targets", targets)]:
        missing = required_cols - set(df.columns)
        if missing:
            print(f"ERROR: {df_name} missing columns: {missing}", flush=True)
            sys.exit(1)

    # Add missing columns if present as null
    for df_ref, name in [(s1, "S1"), (targets, "Targets")]:
        for col in required_cols:
            if col not in df_ref.columns:
                print(f"WARNING: Adding missing column '{col}' as empty string to {name}", flush=True)

    # Normalize null strings
    for col in ["business_name", "business_address", "country"]:
        s1 = s1.with_columns(pl.col(col).fill_null(""))
        targets = targets.with_columns(pl.col(col).fill_null(""))

    print(f"\nStarting 7-channel candidate generation...", flush=True)
    print(f"  Queries:  {len(s1):,}", flush=True)
    print(f"  Targets:  {len(targets):,}", flush=True)
    print()

    generate_candidates_and_save(
        queries_df=s1,
        targets_df=targets,
        output_path=args.output,
        top_k=args.top_k,
        bigram_max_freq=args.bigram_freq,
        brand_max_freq=args.brand_freq,
        verbose=True,
    )

    total_time = time.time() - t0
    print(f"\n{'=' * 70}", flush=True)
    print(f"COMPLETE. Total wall time: {total_time:.1f}s ({total_time/60:.1f} min)", flush=True)
    print(f"Output written to: {args.output}", flush=True)
    print(f"{'=' * 70}", flush=True)

    # Quick output sanity check
    print("\nRunning output sanity check...", flush=True)
    out_df = pl.read_csv(args.output, separator="\t", infer_schema_length=0, null_values=[])
    print(f"  Lines in output: {len(out_df):,}")
    print(f"  Expected:        {len(s1):,} (one per S1 query)")
    if len(out_df) != len(s1):
        print(f"  WARNING: Row count mismatch! Got {len(out_df):,}, expected {len(s1):,}")
    else:
        print(f"  Row count: OK")

    # Check header
    assert list(out_df.columns) == ["source1_entity_id", "candidate_entity_ids"], \
        f"Header mismatch: {out_df.columns}"
    print(f"  Header: OK")

    # Compute candidate stats
    non_empty = out_df.filter(
        pl.col("candidate_entity_ids").is_not_null() &
        (pl.col("candidate_entity_ids") != "")
    )
    print(f"  Queries with candidates: {len(non_empty):,} ({100*len(non_empty)/len(out_df):.2f}%)")
    print(f"  Queries with 0 candidates: {len(out_df) - len(non_empty):,}")

    print("\nSanity check PASSED. Submission file is ready.", flush=True)


if __name__ == "__main__":
    main()
