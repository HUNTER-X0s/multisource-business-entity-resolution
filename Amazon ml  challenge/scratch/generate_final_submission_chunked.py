"""
scratch/generate_final_submission_chunked.py
============================================
Memory-safe End-to-End Competition Submission Generator.
Shards test queries into batches of SHARD_SIZE to avoid OOM when running
10-channel retrieval against the 9.97M test target corpus.

Strategy:
  - Process queries in chunks of SHARD_SIZE (200k)
  - For each chunk, run 10-channel retrieval → write shard parquet
  - Concatenate all shards at the end → feature extraction → inference
"""
import os
import sys
import time
import json
import pickle
import shutil
import subprocess
from pathlib import Path
from collections import defaultdict

import numpy as np
import polars as pl
import pandas as pd
from rapidfuzz import fuzz as rfuzz
from rapidfuzz.distance import JaroWinkler

ROOT = Path("z:/Amazon ML")
CACHE_DIR = ROOT / "experiments" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = ROOT / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SHARD_DIR = OUT_DIR / "retrieval_shards"
SHARD_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = ROOT / "experiments" / "phase2" / "models"
RESULTS_JSON = ROOT / "experiments" / "phase2" / "meta_ensemble_results.json"

TEST_S1_FILE = ROOT / "dataset/raw/test/test_source1.tsv"
TEST_S2_FILE = ROOT / "dataset/raw/test/test_source2.tsv"
TEST_S3_FILE = ROOT / "dataset/raw/test/test_source3.tsv"

TEST_Q_CACHE = CACHE_DIR / "test_queries_norm.parquet"
TEST_T_CACHE = CACHE_DIR / "test_targets_norm.parquet"

SHARD_SIZE = 200_000  # queries per shard

sys.path.append(str(ROOT / "scratch"))
from benchmark_10ch_retrieval import normalize_text, run_10_channels, GENERIC_BRAND_WORDS
from train_gpu_meta_ensemble import char_ngram_jac, tok_jac, tok_overlap

def flush(msg, end="\n"):
    print(msg, end=end, flush=True)


# ── Thin channel runner that works on a query shard ──────────────────────────
def run_10_channels_shard(q_shard: pl.DataFrame, t: pl.DataFrame) -> pl.DataFrame:
    """Run 10-channel retrieval for a single query shard against full target."""
    channels = []

    def ch(name, prio, q_df, t_df, on_cols):
        pairs = (
            q_df.select(["entity_id"] + on_cols)
            .join(t_df.select(["entity_id"] + on_cols).rename({"entity_id": "target_id"}), on=on_cols, how="inner")
            .select(["entity_id", "target_id"]).unique()
            .with_columns(pl.lit(prio).alias("prio"), pl.lit(name).alias("ch"))
        )
        return pairs

    channels.append(ch("Ch1 Exact Name", 1.00,
        q_shard.filter(pl.col("clean_name") != ""),
        t.filter(pl.col("clean_name") != ""),
        ["country", "clean_name"]))

    channels.append(ch("Ch2 Sorted Name", 0.98,
        q_shard.filter(pl.col("sorted_name") != ""),
        t.filter(pl.col("sorted_name") != ""),
        ["country", "sorted_name"]))

    channels.append(ch("Ch3 Spaceless Name", 0.95,
        q_shard.filter(pl.col("spaceless_name").str.len_chars() >= 6),
        t.filter(pl.col("spaceless_name").str.len_chars() >= 6),
        ["country", "spaceless_name"]))

    channels.append(ch("Ch4 Exact Address", 0.90,
        q_shard.filter(pl.col("std_address").str.len_chars() >= 10),
        t.filter(pl.col("std_address").str.len_chars() >= 10),
        ["country", "std_address"]))

    channels.append(ch("Ch5 Sorted Address", 0.88,
        q_shard.filter(pl.col("sorted_addr").str.len_chars() >= 15),
        t.filter(pl.col("sorted_addr").str.len_chars() >= 15),
        ["country", "sorted_addr"]))

    # Ch6 – Rare Name Bigram (compute rarity on full target corpus)
    bi_counts = t.filter(pl.col("name_bigram") != "").group_by(["country", "name_bigram"]).len()
    rare_bi = bi_counts.filter(pl.col("len") <= 500).select(["country", "name_bigram"])
    q_bi = q_shard.filter(pl.col("name_bigram") != "").join(rare_bi, on=["country", "name_bigram"], how="inner")
    t_bi = t.filter(pl.col("name_bigram") != "").join(rare_bi, on=["country", "name_bigram"], how="inner")
    channels.append(ch("Ch6 Rare Name Bigram", 0.85, q_bi, t_bi, ["country", "name_bigram"]))

    # Ch7 – Rare Brand Token
    br_counts = t.filter(pl.col("brand_token") != "").group_by(["country", "brand_token"]).len()
    rare_br = br_counts.filter(pl.col("len") <= 300).select(["country", "brand_token"])
    q_br = q_shard.filter(pl.col("brand_token") != "").join(rare_br, on=["country", "brand_token"], how="inner")
    t_br = t.filter(pl.col("brand_token") != "").join(rare_br, on=["country", "brand_token"], how="inner")
    channels.append(ch("Ch7 Rare Brand Token", 0.83, q_br, t_br, ["country", "brand_token"]))

    channels.append(ch("Ch8 HouseNo + First Word", 0.80,
        q_shard.filter((pl.col("house_no").str.len_chars() >= 2) & (pl.col("first_word").str.len_chars() >= 4)),
        t.filter((pl.col("house_no").str.len_chars() >= 2) & (pl.col("first_word").str.len_chars() >= 4)),
        ["country", "house_no", "first_word"]))

    channels.append(ch("Ch9 Postal + First Word", 0.78,
        q_shard.filter((pl.col("postal_code").str.len_chars() >= 5) & (pl.col("first_word").str.len_chars() >= 4)),
        t.filter((pl.col("postal_code").str.len_chars() >= 5) & (pl.col("first_word").str.len_chars() >= 4)),
        ["country", "postal_code", "first_word"]))

    channels.append(ch("Ch10 HouseNo + Street Prefix", 0.75,
        q_shard.filter((pl.col("house_no").str.len_chars() >= 2) & (pl.col("street_prefix").str.len_chars() >= 6)),
        t.filter((pl.col("house_no").str.len_chars() >= 2) & (pl.col("street_prefix").str.len_chars() >= 6)),
        ["country", "house_no", "street_prefix"]))

    pooled = (
        pl.concat(channels)
        .group_by(["entity_id", "target_id"])
        .agg([pl.col("prio").max().alias("max_prio"), pl.len().alias("n_channels")])
    )
    return pooled


# ── Feature extraction (vectorized per row, reused from train script) ─────────
def extract_features(cands_df: pl.DataFrame, q_dict: dict, t_dict: dict) -> pd.DataFrame:
    rows_list = list(cands_df.select(["entity_id", "target_id", "max_prio", "n_channels"]).iter_rows(named=True))
    total = len(rows_list)
    t0 = time.time()
    feat_records = []

    for i, row in enumerate(rows_list):
        if i % 500_000 == 0 and i > 0:
            flush(f"    [{i:,}/{total:,}] {time.time()-t0:.1f}s")

        qid = str(row["entity_id"])
        tid = str(row["target_id"])
        q = q_dict.get(qid, {})
        t = t_dict.get(tid, {})

        qn = q.get("clean_name", "") or ""
        tn = t.get("clean_name", "") or ""
        qns = q.get("sorted_name", "") or ""
        tns = t.get("sorted_name", "") or ""
        qns2 = q.get("stripped_name", "") or ""
        tns2 = t.get("stripped_name", "") or ""
        qsp = q.get("spaceless_name", "") or ""
        tsp = t.get("spaceless_name", "") or ""
        qa = q.get("std_address", "") or ""
        ta = t.get("std_address", "") or ""
        qas = q.get("sorted_addr", "") or ""
        tas = t.get("sorted_addr", "") or ""
        qhno = q.get("house_no", "") or ""
        thno = t.get("house_no", "") or ""
        qpin = q.get("postal_code", "") or ""
        tpin = t.get("postal_code", "") or ""
        qfw = q.get("first_word", "") or ""
        tfw = t.get("first_word", "") or ""
        cty = (q.get("country", "") or "").lower()
        is_s3 = 1 if tid.startswith("S3-") else 0

        rec = {
            "entity_id": qid, "target_id": tid,
            "max_prio": float(row["max_prio"]), "n_channels": int(row["n_channels"]),
            "exact_clean_name": int(qn == tn and qn != ""),
            "exact_sorted_name": int(qns == tns and qns != ""),
            "exact_stripped_name": int(qns2 == tns2 and qns2 != ""),
            "exact_spaceless": int(len(qsp) >= 5 and qsp == tsp),
            "spaceless_subset": int(len(qsp) >= 5 and len(tsp) >= 5 and (qsp in tsp or tsp in qsp) and qsp != tsp),
            "exact_clean_addr": int(qa == ta and len(qa) >= 10),
            "exact_sorted_addr": int(qas == tas and len(qas) >= 12),
            "hno_match": int(qhno == thno and len(qhno) >= 2),
            "pin_match": int(qpin == tpin and len(qpin) >= 5),
            "fw_match": int(qfw == tfw and len(qfw) >= 4),
            "name_jw": JaroWinkler.similarity(qn, tn),
            "name_tok_sort": rfuzz.token_sort_ratio(qn, tn) / 100.0,
            "name_tok_set": rfuzz.token_set_ratio(qn, tn) / 100.0,
            "name_partial": rfuzz.partial_ratio(qn, tn) / 100.0,
            "name_char2_jac": char_ngram_jac(qn, tn, 2),
            "name_char3_jac": char_ngram_jac(qn, tn, 3),
            "name_char4_jac": char_ngram_jac(qn, tn, 4),
            "name_tok_jac": tok_jac(qn, tn),
            "name_tok_ovlp": tok_overlap(qn, tn),
            "addr_jw": JaroWinkler.similarity(qa, ta) if qa and ta else 0.0,
            "addr_tok_sort": rfuzz.token_sort_ratio(qa, ta) / 100.0 if qa and ta else 0.0,
            "addr_tok_set": rfuzz.token_set_ratio(qa, ta) / 100.0 if qa and ta else 0.0,
            "addr_char3_jac": char_ngram_jac(qa, ta, 3),
            "addr_tok_jac": tok_jac(qas, tas),
            "addr_null_tgt": int(not ta),
            "name_len_diff": abs(len(qn) - len(tn)),
            "name_len_ratio": min(len(qn), len(tn)) / max(len(qn), len(tn), 1),
            "addr_len_diff": abs(len(qa) - len(ta)),
            "name_wc_diff": abs(len(qn.split()) - len(tn.split())),
            "name_wc_ratio": min(len(qn.split()), len(tn.split())) / max(len(qn.split()), len(tn.split()), 1),
            "name_prefix3": int(qn[:3] == tn[:3] if len(qn) >= 3 and len(tn) >= 3 else 0),
            "name_prefix5": int(qn[:5] == tn[:5] if len(qn) >= 5 and len(tn) >= 5 else 0),
            "hno_contradiction": int(qhno != thno and bool(qhno) and bool(thno)),
            "pin_contradiction": int(qpin != tpin and bool(qpin) and bool(tpin)),
            "pin_prefix3": int(qpin[:3] == tpin[:3] if len(qpin) >= 3 and len(tpin) >= 3 else 0),
            "pin_both_present": int(bool(qpin) and bool(tpin)),
            "name_x_addr": JaroWinkler.similarity(qn, tn) * (rfuzz.token_sort_ratio(qa, ta) / 100.0 if qa and ta else 0.0),
            "name_x_pin": JaroWinkler.similarity(qn, tn) * float(qpin == tpin and len(qpin) >= 5),
            "sorted_x_addr": (rfuzz.token_sort_ratio(qns, tns) / 100.0) * (rfuzz.token_set_ratio(qa, ta) / 100.0 if qa and ta else 0.0),
            "is_source3": is_s3,
            "india_flag": int("india" in cty),
            "france_flag": int("france" in cty),
        }
        feat_records.append(rec)

    return pd.DataFrame(feat_records)


def main():
    flush("=" * 70)
    flush("AMAZON ML CHALLENGE 2026 — CHUNKED FINAL TEST SUBMISSION")
    flush("=" * 70)

    # 1. Load Models & Parameters
    flush("[1] Loading trained models...")
    with open(RESULTS_JSON) as f:
        meta_res = json.load(f)
    best_theta = meta_res["best_theta"]
    best_cap = meta_res.get("best_cap", False)
    feat_cols = meta_res["features"]
    flush(f"  Threshold={best_theta} | Cap={best_cap} | Features={len(feat_cols)}")

    model_lgb = pickle.load(open(MODEL_DIR / "meta_lgb.pkl", "rb"))
    model_xgb = pickle.load(open(MODEL_DIR / "meta_xgb.pkl", "rb"))
    model_cb  = pickle.load(open(MODEL_DIR / "meta_cb.pkl",  "rb"))
    meta_model = pickle.load(open(MODEL_DIR / "meta_stacker.pkl", "rb"))
    flush("  All models loaded.")

    # 2. Load or build normalized test queries
    if TEST_Q_CACHE.exists():
        flush(f"[2] Loading cached test queries from {TEST_Q_CACHE}...")
        test_q_norm = pl.read_parquet(TEST_Q_CACHE)
    else:
        flush("[2] Normalizing test queries...")
        t0 = time.time()
        test_s1 = pl.read_csv(TEST_S1_FILE, separator="\t", quote_char=None)
        test_q_norm = normalize_text(test_s1)
        test_q_norm.write_parquet(TEST_Q_CACHE)
        flush(f"  Done in {time.time()-t0:.1f}s. Cached.")
    flush(f"  Test queries: {len(test_q_norm):,}")

    # 3. Load or build normalized test targets
    if TEST_T_CACHE.exists():
        flush(f"[3] Loading cached test targets from {TEST_T_CACHE}...")
        test_t_norm = pl.read_parquet(TEST_T_CACHE)
    else:
        flush("[3] Normalizing test targets (S2 + S3)...")
        t0 = time.time()
        s2 = pl.read_csv(TEST_S2_FILE, separator="\t", quote_char=None)
        s3 = pl.read_csv(TEST_S3_FILE, separator="\t", quote_char=None)
        test_t_norm = normalize_text(pl.concat([s2, s3]))
        test_t_norm.write_parquet(TEST_T_CACHE)
        flush(f"  Done in {time.time()-t0:.1f}s. Cached.")
    flush(f"  Test targets: {len(test_t_norm):,}")

    # 4. Build lookup dicts for feature extraction
    flush("\n[4] Building feature extraction lookup dicts...")
    FEAT_COLS_NEEDED = [
        "entity_id", "clean_name", "stripped_name", "sorted_name", "spaceless_name",
        "std_address", "sorted_addr", "house_no", "postal_code", "country", "first_word"
    ]
    q_dict = {str(r["entity_id"]): r for r in test_q_norm.select(FEAT_COLS_NEEDED).iter_rows(named=True)}
    t_dict = {str(r["entity_id"]): r for r in test_t_norm.select(FEAT_COLS_NEEDED).iter_rows(named=True)}
    flush(f"  Q dict: {len(q_dict):,} | T dict: {len(t_dict):,}")

    # 5. Sharded retrieval — process SHARD_SIZE queries at a time
    flush(f"\n[5] Sharded 10-channel retrieval (shard_size={SHARD_SIZE:,})...")
    query_ids = test_q_norm["entity_id"].to_list()
    n_shards = (len(query_ids) + SHARD_SIZE - 1) // SHARD_SIZE
    flush(f"  Total queries: {len(query_ids):,} -> {n_shards} shards")

    shard_files = []
    t_ret_total = time.time()

    for shard_idx in range(n_shards):
        shard_path = SHARD_DIR / f"shard_{shard_idx:04d}.parquet"

        if shard_path.exists():
            flush(f"  Shard {shard_idx+1}/{n_shards}: CACHED ({shard_path.name})")
            shard_files.append(shard_path)
            continue

        t_shard = time.time()
        start = shard_idx * SHARD_SIZE
        end   = min(start + SHARD_SIZE, len(query_ids))
        shard_ids = query_ids[start:end]

        q_shard = test_q_norm.filter(pl.col("entity_id").is_in(shard_ids))
        flush(f"  Shard {shard_idx+1}/{n_shards}: {len(q_shard):,} queries...", end="")

        shard_cands = run_10_channels_shard(q_shard, test_t_norm)
        shard_cands.write_parquet(shard_path)

        flush(f" → {len(shard_cands):,} pairs | {time.time()-t_shard:.1f}s")
        shard_files.append(shard_path)
        del shard_cands, q_shard

    flush(f"\n  All shards done in {time.time()-t_ret_total:.1f}s total.")

    # 6. Concatenate shards
    flush("\n[6] Concatenating all retrieval shards...")
    t0 = time.time()
    all_cands = pl.concat([pl.read_parquet(f) for f in shard_files])
    flush(f"  Total candidate pairs before dedup: {len(all_cands):,}")
    # Re-aggregate in case same (q,t) pair was retrieved by two different channel types across shards
    # (should be rare since shards are disjoint by query, but just to be safe)
    flush(f"  Combined unique pairs: {len(all_cands):,} in {time.time()-t0:.1f}s")

    # 7. Export candidate_pairs.tsv
    flush("\n[7] Exporting output/candidate_pairs.tsv...")
    t0 = time.time()
    cand_sorted = all_cands.sort(["entity_id", "max_prio", "n_channels"], descending=[False, True, True])
    cand_budgeted = cand_sorted.group_by("entity_id", maintain_order=True).head(40)
    cand_map = (
        cand_budgeted.group_by("entity_id")
        .agg(pl.col("target_id").list.join(",").alias("candidate_entity_ids"))
        .rename({"entity_id": "source1_entity_id"})
    )
    all_s1_ids = pl.DataFrame({"source1_entity_id": test_q_norm["entity_id"]})
    cand_tsv_df = all_s1_ids.join(cand_map, on="source1_entity_id", how="left").with_columns(
        pl.col("candidate_entity_ids").fill_null("")
    )
    cand_tsv_path = OUT_DIR / "candidate_pairs.tsv"
    cand_tsv_df.write_csv(cand_tsv_path, separator="\t")
    flush(f"  candidate_pairs.tsv written ({len(cand_tsv_df):,} rows) in {time.time()-t0:.1f}s")

    # 8. Feature extraction
    flush(f"\n[8] Extracting features for {len(all_cands):,} candidate pairs...")
    t0 = time.time()
    test_feat_df = extract_features(all_cands, q_dict, t_dict)
    flush(f"  Feature extraction done in {time.time()-t0:.1f}s")

    # 9. GPU Meta-Ensemble Inference
    flush("\n[9] Running Meta-Ensemble Inference...")
    t0 = time.time()
    X_test = test_feat_df[feat_cols].values.astype(np.float32)

    p_lgb = model_lgb.predict_proba(X_test)[:, 1]
    flush(f"  LightGBM done ({time.time()-t0:.1f}s)")
    t1 = time.time()
    p_xgb = model_xgb.predict_proba(X_test)[:, 1]
    flush(f"  XGBoost-GPU done ({time.time()-t1:.1f}s)")
    t1 = time.time()
    p_cb = model_cb.predict_proba(X_test)[:, 1]
    flush(f"  CatBoost-GPU done ({time.time()-t1:.1f}s)")

    meta_X_test = np.column_stack([p_lgb, p_xgb, p_cb])
    p_meta = meta_model.predict_proba(meta_X_test)[:, 1]
    flush(f"  Meta-stacker blending done. Total inference: {time.time()-t0:.1f}s")

    # 10. Decision Rule & Write matching_results.tsv
    flush("\n[10] Applying decision rule (threshold={:.2f})...".format(best_theta))
    q_arr = test_feat_df["entity_id"].values
    t_arr = test_feat_df["target_id"].values

    pred_map = defaultdict(list)
    for p, q, t in zip(p_meta, q_arr, t_arr):
        pred_map[q].append((p, t))

    matched_records = []
    all_s1_set = set(str(x) for x in test_q_norm["entity_id"].to_list())

    for qid in sorted(all_s1_set):
        preds_q = pred_map.get(qid, [])
        sorted_cands = sorted(preds_q, key=lambda x: x[0], reverse=True)

        matched = []
        if best_cap:
            s2_found = s3_found = 0
            for prob, tid in sorted_cands:
                if prob >= best_theta:
                    if tid.startswith("S3-") and s3_found < 1:
                        matched.append(tid); s3_found += 1
                    elif not tid.startswith("S3-") and s2_found < 1:
                        matched.append(tid); s2_found += 1
        else:
            matched = [tid for (prob, tid) in sorted_cands if prob >= best_theta]

        matched_records.append((qid, ",".join(sorted(matched)) if matched else ""))

    match_df = pl.DataFrame({
        "source1_entity_id": [r[0] for r in matched_records],
        "matched_entity_ids": [r[1] for r in matched_records]
    })
    match_tsv_path = OUT_DIR / "matching_results.tsv"
    match_df.write_csv(match_tsv_path, separator="\t")
    flush(f"\n  matching_results.tsv written ({len(match_df):,} rows)")
    flush(f"  Queries with at least one match: {(match_df['matched_entity_ids'] != '').sum():,}")

    # 11. Official Validation
    flush("\n[11] Running official competition validator...")
    val_script = ROOT / "dataset" / "student_resource" / "utils" / "validate_submission.py"
    if val_script.exists():
        cmd = [
            sys.executable, str(val_script),
            "--matching",   str(match_tsv_path),
            "--candidate",  str(cand_tsv_path),
            "--test-dir",   str(ROOT / "dataset" / "raw" / "test"),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        flush(res.stdout)
        if res.stderr:
            flush("STDERR: " + res.stderr[:500])
        flush(f"Validator exit code: {res.returncode}")
    else:
        flush("  validate_submission.py not found — skipping.")

    # Cleanup shards (optional — comment out to cache for re-runs)
    # shutil.rmtree(SHARD_DIR)

    flush("\n" + "=" * 70)
    flush("SUBMISSION PIPELINE COMPLETE!")
    flush(f"  candidate_pairs.tsv  → {cand_tsv_path}")
    flush(f"  matching_results.tsv → {match_tsv_path}")
    flush("=" * 70)


if __name__ == "__main__":
    main()
