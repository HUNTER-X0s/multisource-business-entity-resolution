"""
Quick smoke test: run chunked generation on first 200K S1 queries,
measure memory and speed, then verify output format.
"""
import os, sys, time
sys.path.insert(0, os.path.abspath("."))
import polars as pl
from src.candidate_generation import (
    normalize_entities, _compute_rare_bigrams, _compute_rare_brand_tokens,
    _ch1_exact_name, _ch2_sorted_name, _ch3_exact_address, _ch4_sorted_address,
    _ch5_name_bigram, _ch6_street_prefix, _ch7_brand_token
)

N_TEST = 200_000
CHUNK = 100_000
TOP_K = 40

print(f"Smoke test: {N_TEST:,} S1 queries, chunk={CHUNK:,}, K={TOP_K}")

# Load small S1 slice
s1 = pl.read_csv("dataset/raw/test/test_source1.tsv", separator="\t",
    infer_schema_length=0, null_values=["N/A","n/a","null","NULL","None"]
).head(N_TEST).with_columns([
    pl.col("business_name").fill_null(""),
    pl.col("business_address").fill_null(""),
    pl.col("country").fill_null(""),
])

# Full targets
print("Loading targets...", flush=True)
t0 = time.time()
s2 = pl.read_csv("dataset/raw/test/test_source2.tsv", separator="\t",
    infer_schema_length=0, null_values=["N/A","n/a","null","NULL","None"])
s3 = pl.read_csv("dataset/raw/test/test_source3.tsv", separator="\t",
    infer_schema_length=0, null_values=["N/A","n/a","null","NULL","None"])
targets_raw = pl.concat([s2, s3]).with_columns([
    pl.col("business_name").fill_null(""),
    pl.col("business_address").fill_null(""),
    pl.col("country").fill_null(""),
])
print(f"  Loaded {len(targets_raw):,} targets in {time.time()-t0:.1f}s")

print("Normalizing targets...", flush=True)
t1 = time.time()
targets_norm = normalize_entities(targets_raw)
del targets_raw
print(f"  Normalized in {time.time()-t1:.1f}s")

print("Building frequency filters...", flush=True)
t2 = time.time()
rare_bigrams = _compute_rare_bigrams(targets_norm, max_freq=500)
rare_brands = _compute_rare_brand_tokens(targets_norm, max_freq=300)
print(f"  Done in {time.time()-t2:.1f}s. Bigrams: {len(rare_bigrams):,}, Brands: {len(rare_brands):,}")

# Run 2 chunks
out_lines = ["source1_entity_id\tcandidate_entity_ids"]
n_chunks = (N_TEST + CHUNK - 1) // CHUNK
total_pairs = 0
for chunk_idx in range(n_chunks):
    t_ch = time.time()
    start = chunk_idx * CHUNK
    end = min(start + CHUNK, N_TEST)
    chunk_q_raw = s1.slice(start, end-start)
    chunk_q = normalize_entities(chunk_q_raw)

    ch_pairs = [
        _ch1_exact_name(chunk_q, targets_norm),
        _ch2_sorted_name(chunk_q, targets_norm),
        _ch3_exact_address(chunk_q, targets_norm),
        _ch4_sorted_address(chunk_q, targets_norm),
        _ch5_name_bigram(chunk_q, targets_norm, rare_bigrams),
        _ch6_street_prefix(chunk_q, targets_norm),
        _ch7_brand_token(chunk_q, targets_norm, rare_brands),
    ]
    all_raw = pl.concat([p.select(["entity_id", "target_id", "prio"]) for p in ch_pairs])
    candidate_pool = (
        all_raw.group_by(["entity_id", "target_id"])
        .agg([pl.col("prio").max().alias("max_prio"), pl.len().alias("n_channels")])
        .sort(["entity_id", "max_prio", "n_channels"], descending=[False, True, True])
        .group_by("entity_id", maintain_order=True).head(TOP_K)
    )
    grouped = candidate_pool.group_by("entity_id").agg(pl.col("target_id").alias("cand_list"))
    cand_map = {r["entity_id"]: r["cand_list"] for r in grouped.iter_rows(named=True)}
    for eid in chunk_q_raw["entity_id"].to_list():
        cands = cand_map.get(eid, [])
        out_lines.append(f"{eid}\t{','.join(cands)}")
        total_pairs += len(cands)

    elapsed = time.time()-t_ch
    print(f"  Chunk {chunk_idx+1}/{n_chunks}: {len(candidate_pool):,} candidate pairs in {elapsed:.1f}s "
          f"(~{elapsed * (1_732_544//CHUNK)/60:.0f}min ETA for full run)", flush=True)

print(f"\nSmoke test complete.")
print(f"  Total pairs: {total_pairs:,}  ({total_pairs/N_TEST:.1f} avg/query)")
print(f"  Output rows: {len(out_lines)-1:,}  Expected: {N_TEST:,}  {'OK' if len(out_lines)-1 == N_TEST else 'MISMATCH'}")
# Verify no duplicates
s1_ids = [l.split('\t')[0] for l in out_lines[1:]]
assert len(set(s1_ids)) == N_TEST, f"Duplicate S1 IDs found!"
print("  No duplicate S1 IDs: OK")
print("  Smoke test PASSED")
