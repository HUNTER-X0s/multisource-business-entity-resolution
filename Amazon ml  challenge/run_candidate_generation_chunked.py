"""
run_candidate_generation_chunked.py
=====================================
Memory-safe production runner for generating candidate_pairs.tsv on the full
test set using chunked query processing.

Strategy:
  - Normalize ALL targets once and hold in RAM (~4GB normalized Polars frame)
  - Process S1 queries in batches of CHUNK_SIZE (default 100,000)
  - Per batch: run all 7 channels against full target pool, rank, truncate to K
  - Write results incrementally to output file

This avoids the OOM that occurs when holding 270M+ raw candidate pairs in RAM.

Memory profile at full scale:
  - Normalized targets:  ~4.5 GB  (9.97M rows × ~450 bytes/row in Arrow)
  - Per-batch candidates: ~0.5-1.5 GB  (100K queries × ~40 candidates × ~120 bytes)
  - Peak total:           ~6 GB  (fits in 16GB RAM)

Usage:
  python run_candidate_generation_chunked.py [--top-k 40] [--chunk 100000]
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath("."))
import polars as pl

from src.candidate_generation import (
    normalize_entities,
    _compute_rare_bigrams,
    _compute_rare_brand_tokens,
    _ch1_exact_name,
    _ch2_sorted_name,
    _ch3_exact_address,
    _ch4_sorted_address,
    _ch5_name_bigram,
    _ch6_street_prefix,
    _ch7_brand_token,
)


def generate_candidates_chunked(
    s1: pl.DataFrame,
    targets_norm: pl.DataFrame,
    rare_bigrams: pl.DataFrame,
    rare_brands: pl.DataFrame,
    top_k: int,
    chunk_size: int,
    output_path: str,
    verbose: bool = True,
) -> dict:
    """
    Processes S1 queries in chunks. For each chunk:
      1. Normalize the chunk queries.
      2. Run all 7 channels against the pre-normalized target pool.
      3. Merge channels: keep max priority + channel count per pair.
      4. Rank by (max_prio desc, n_channels desc), truncate to top_k.
      5. Format as TSV and write to output file.
    Returns statistics dict.
    """
    n_total = len(s1)
    n_chunks = (n_total + chunk_size - 1) // chunk_size
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    total_pairs_written = 0
    zero_cand_count = 0
    chunk_times = []

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")

        for chunk_idx in range(n_chunks):
            t_chunk = time.time()
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, n_total)
            chunk_q_raw = s1.slice(start, end - start)
            chunk_q = normalize_entities(chunk_q_raw)

            # Run all 7 channels against full target pool
            # Note: channel functions already include their prio column
            ch_pairs = []
            ch_pairs.append(_ch1_exact_name(chunk_q, targets_norm))
            ch_pairs.append(_ch2_sorted_name(chunk_q, targets_norm))
            ch_pairs.append(_ch3_exact_address(chunk_q, targets_norm))
            ch_pairs.append(_ch4_sorted_address(chunk_q, targets_norm))
            ch_pairs.append(_ch5_name_bigram(chunk_q, targets_norm, rare_bigrams))
            ch_pairs.append(_ch6_street_prefix(chunk_q, targets_norm))
            ch_pairs.append(_ch7_brand_token(chunk_q, targets_norm, rare_brands))

            # Merge and rank
            all_raw = pl.concat([p.select(["entity_id", "target_id", "prio"]) for p in ch_pairs])
            candidate_pool = (
                all_raw
                .group_by(["entity_id", "target_id"])
                .agg([
                    pl.col("prio").max().alias("max_prio"),
                    pl.len().alias("n_channels"),
                ])
                .sort(["entity_id", "max_prio", "n_channels"], descending=[False, True, True])
                .group_by("entity_id", maintain_order=True)
                .head(top_k)
            )

            # Group by entity_id for TSV output
            grouped = (
                candidate_pool
                .group_by("entity_id")
                .agg(pl.col("target_id").alias("cand_list"))
            )
            cand_map = {r["entity_id"]: r["cand_list"] for r in grouped.iter_rows(named=True)}

            # Write all S1 IDs in this chunk (including zero-candidate)
            for eid in chunk_q_raw["entity_id"].to_list():
                cands = cand_map.get(eid, [])
                if not cands:
                    zero_cand_count += 1
                cand_str = ",".join(cands)
                f.write(f"{eid}\t{cand_str}\n")
                total_pairs_written += len(cands)

            elapsed = time.time() - t_chunk
            chunk_times.append(elapsed)
            avg_time = sum(chunk_times) / len(chunk_times)
            remaining = n_chunks - chunk_idx - 1
            eta = remaining * avg_time

            if verbose:
                print(
                    f"  Chunk {chunk_idx+1:4d}/{n_chunks} "
                    f"| rows {start:,}-{end:,} "
                    f"| pairs {len(candidate_pool):,} "
                    f"| {elapsed:.1f}s "
                    f"| ETA {eta/60:.1f}min",
                    flush=True,
                )

    return {
        "total_pairs_written": total_pairs_written,
        "zero_candidate_queries": zero_cand_count,
        "n_chunks": n_chunks,
        "avg_chunk_time_s": sum(chunk_times) / len(chunk_times) if chunk_times else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--chunk", type=int, default=100000,
                        help="Number of S1 queries per processing chunk (default 100,000)")
    parser.add_argument("--bigram-freq", type=int, default=500)
    parser.add_argument("--brand-freq", type=int, default=300)
    parser.add_argument("--output", type=str, default="output/candidate_pairs.tsv")
    args = parser.parse_args()

    print("=" * 70, flush=True)
    print("CANDIDATE GENERATION — CHUNKED FULL TEST SET", flush=True)
    print("=" * 70, flush=True)
    print(f"  Top-K:      {args.top_k}", flush=True)
    print(f"  Chunk size: {args.chunk:,}", flush=True)
    print(f"  Bigram freq cutoff: {args.bigram_freq}", flush=True)
    print(f"  Brand freq cutoff:  {args.brand_freq}", flush=True)
    print(f"  Output:     {args.output}", flush=True)
    print()

    t0 = time.time()

    # --- Load all data ---
    print("Loading S1 queries...", flush=True)
    s1 = pl.read_csv(
        "dataset/raw/test/test_source1.tsv",
        separator="\t", infer_schema_length=0,
        null_values=["N/A", "n/a", "null", "NULL", "None"],
    ).with_columns([
        pl.col("business_name").fill_null(""),
        pl.col("business_address").fill_null(""),
        pl.col("country").fill_null(""),
    ])
    print(f"  {len(s1):,} S1 queries loaded in {time.time()-t0:.1f}s", flush=True)

    print("Loading S2 + S3 targets...", flush=True)
    t1 = time.time()
    s2 = pl.read_csv(
        "dataset/raw/test/test_source2.tsv",
        separator="\t", infer_schema_length=0,
        null_values=["N/A", "n/a", "null", "NULL", "None"],
    )
    s3 = pl.read_csv(
        "dataset/raw/test/test_source3.tsv",
        separator="\t", infer_schema_length=0,
        null_values=["N/A", "n/a", "null", "NULL", "None"],
    )
    targets_raw = pl.concat([s2, s3]).with_columns([
        pl.col("business_name").fill_null(""),
        pl.col("business_address").fill_null(""),
        pl.col("country").fill_null(""),
    ])
    print(f"  {len(targets_raw):,} total targets loaded in {time.time()-t1:.1f}s", flush=True)

    # --- Normalize targets once ---
    print("\nNormalizing target pool (this is done ONCE)...", flush=True)
    t2 = time.time()
    targets_norm = normalize_entities(targets_raw)
    del targets_raw  # free raw memory
    print(f"  Targets normalized in {time.time()-t2:.1f}s", flush=True)

    # --- Build frequency filters on target corpus ---
    print("Building frequency filters...", flush=True)
    t3 = time.time()
    rare_bigrams = _compute_rare_bigrams(targets_norm, max_freq=args.bigram_freq)
    rare_brands = _compute_rare_brand_tokens(targets_norm, max_freq=args.brand_freq)
    print(f"  Filters built in {time.time()-t3:.1f}s", flush=True)
    print(f"  Rare bigrams: {len(rare_bigrams):,}  |  Rare brands: {len(rare_brands):,}", flush=True)

    # --- Chunked candidate generation ---
    print(f"\nStarting chunked processing ({args.chunk:,} queries/chunk)...", flush=True)
    t4 = time.time()
    stats = generate_candidates_chunked(
        s1=s1,
        targets_norm=targets_norm,
        rare_bigrams=rare_bigrams,
        rare_brands=rare_brands,
        top_k=args.top_k,
        chunk_size=args.chunk,
        output_path=args.output,
        verbose=True,
    )
    gen_time = time.time() - t4

    # --- Summary ---
    total_time = time.time() - t0
    print(f"\n{'=' * 70}", flush=True)
    print(f"CANDIDATE GENERATION COMPLETE", flush=True)
    print(f"  Total pairs written:     {stats['total_pairs_written']:,}", flush=True)
    print(f"  Zero-candidate queries:  {stats['zero_candidate_queries']:,}", flush=True)
    print(f"  Chunks processed:        {stats['n_chunks']}", flush=True)
    print(f"  Avg time/chunk:          {stats['avg_chunk_time_s']:.1f}s", flush=True)
    print(f"  Generation time:         {gen_time/60:.1f} min", flush=True)
    print(f"  Total wall time:         {total_time/60:.1f} min", flush=True)
    print(f"  Output:                  {args.output}", flush=True)
    print(f"{'=' * 70}", flush=True)

    # --- Sanity check ---
    print("\nRunning output sanity check...", flush=True)
    out_df = pl.read_csv(args.output, separator="\t", infer_schema_length=0, null_values=[])
    n_out = len(out_df)
    n_exp = len(s1)
    print(f"  Output rows: {n_out:,}  |  Expected: {n_exp:,}  |  {'OK' if n_out == n_exp else 'MISMATCH!'}")
    assert list(out_df.columns) == ["source1_entity_id", "candidate_entity_ids"], \
        f"Header mismatch: {out_df.columns}"
    print(f"  Header: OK")

    non_empty = out_df.filter(
        pl.col("candidate_entity_ids").is_not_null() &
        (pl.col("candidate_entity_ids").str.len_chars() > 0)
    )
    print(f"  Queries with >=1 candidate: {len(non_empty):,} ({100*len(non_empty)/n_out:.2f}%)")
    print(f"\nDone. Submission file ready: {args.output}", flush=True)


if __name__ == "__main__":
    main()
