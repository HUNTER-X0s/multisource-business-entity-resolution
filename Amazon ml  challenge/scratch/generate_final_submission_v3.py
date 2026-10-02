"""
scratch/generate_final_submission_v3.py
========================================
Memory-safe submission generator — shard-at-a-time approach.

Key insight: queries are DISJOINT across shards, so:
  1. Budget top-40 within each shard (never more than 8M rows per shard)
  2. Extract features + run GPU inference per shard
  3. Collect predictions across shards
  4. Write TSV outputs at the very end

Peak RAM = max(shard size in RAM) << 266M combined
"""
import os
import sys
import time
import json
import pickle
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
SHARD_DIR = OUT_DIR / "retrieval_shards_budgeted"
SHARD_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = ROOT / "experiments" / "phase2" / "models"
RESULTS_JSON = ROOT / "experiments" / "phase2" / "meta_ensemble_results.json"

TEST_S1_FILE = ROOT / "dataset/raw/test/test_source1.tsv"
TEST_S2_FILE = ROOT / "dataset/raw/test/test_source2.tsv"
TEST_S3_FILE = ROOT / "dataset/raw/test/test_source3.tsv"

TEST_Q_CACHE = CACHE_DIR / "test_queries_norm.parquet"
TEST_T_CACHE = CACHE_DIR / "test_targets_norm.parquet"

SHARD_SIZE = 200_000   # queries per shard
TOP_K      = 40        # candidates per query budget

sys.path.append(str(ROOT / "scratch"))
from benchmark_10ch_retrieval import normalize_text, GENERIC_BRAND_WORDS
from train_gpu_meta_ensemble import char_ngram_jac, tok_jac, tok_overlap

def flush(msg, end="\n"):
    print(msg, end=end, flush=True)


# ── Per-shard 10-channel retrieval (returns budgeted top-K) ──────────────────
def retrieve_and_budget(q_shard: pl.DataFrame, t: pl.DataFrame, top_k: int) -> pl.DataFrame:
    channels = []

    def ch(name, prio, q_df, t_df, on_cols):
        return (
            q_df.select(["entity_id"] + on_cols)
            .join(t_df.select(["entity_id"] + on_cols).rename({"entity_id": "target_id"}), on=on_cols, how="inner")
            .select(["entity_id", "target_id"]).unique()
            .with_columns(pl.lit(prio).cast(pl.Float32).alias("prio"), pl.lit(name).alias("ch"))
        )

    channels.append(ch("Ch1", 1.00,
        q_shard.filter(pl.col("clean_name") != ""),
        t.filter(pl.col("clean_name") != ""),
        ["country", "clean_name"]))

    channels.append(ch("Ch2", 0.98,
        q_shard.filter(pl.col("sorted_name") != ""),
        t.filter(pl.col("sorted_name") != ""),
        ["country", "sorted_name"]))

    channels.append(ch("Ch3", 0.95,
        q_shard.filter(pl.col("spaceless_name").str.len_chars() >= 6),
        t.filter(pl.col("spaceless_name").str.len_chars() >= 6),
        ["country", "spaceless_name"]))

    channels.append(ch("Ch4", 0.90,
        q_shard.filter(pl.col("std_address").str.len_chars() >= 10),
        t.filter(pl.col("std_address").str.len_chars() >= 10),
        ["country", "std_address"]))

    channels.append(ch("Ch5", 0.88,
        q_shard.filter(pl.col("sorted_addr").str.len_chars() >= 15),
        t.filter(pl.col("sorted_addr").str.len_chars() >= 15),
        ["country", "sorted_addr"]))

    bi_counts = t.filter(pl.col("name_bigram") != "").group_by(["country", "name_bigram"]).len()
    rare_bi = bi_counts.filter(pl.col("len") <= 500).select(["country", "name_bigram"])
    q_bi = q_shard.filter(pl.col("name_bigram") != "").join(rare_bi, on=["country", "name_bigram"], how="inner")
    t_bi = t.filter(pl.col("name_bigram") != "").join(rare_bi, on=["country", "name_bigram"], how="inner")
    channels.append(ch("Ch6", 0.85, q_bi, t_bi, ["country", "name_bigram"]))

    br_counts = t.filter(pl.col("brand_token") != "").group_by(["country", "brand_token"]).len()
    rare_br = br_counts.filter(pl.col("len") <= 300).select(["country", "brand_token"])
    q_br = q_shard.filter(pl.col("brand_token") != "").join(rare_br, on=["country", "brand_token"], how="inner")
    t_br = t.filter(pl.col("brand_token") != "").join(rare_br, on=["country", "brand_token"], how="inner")
    channels.append(ch("Ch7", 0.83, q_br, t_br, ["country", "brand_token"]))

    channels.append(ch("Ch8", 0.80,
        q_shard.filter((pl.col("house_no").str.len_chars() >= 2) & (pl.col("first_word").str.len_chars() >= 4)),
        t.filter((pl.col("house_no").str.len_chars() >= 2) & (pl.col("first_word").str.len_chars() >= 4)),
        ["country", "house_no", "first_word"]))

    channels.append(ch("Ch9", 0.78,
        q_shard.filter((pl.col("postal_code").str.len_chars() >= 5) & (pl.col("first_word").str.len_chars() >= 4)),
        t.filter((pl.col("postal_code").str.len_chars() >= 5) & (pl.col("first_word").str.len_chars() >= 4)),
        ["country", "postal_code", "first_word"]))

    channels.append(ch("Ch10", 0.75,
        q_shard.filter((pl.col("house_no").str.len_chars() >= 2) & (pl.col("street_prefix").str.len_chars() >= 6)),
        t.filter((pl.col("house_no").str.len_chars() >= 2) & (pl.col("street_prefix").str.len_chars() >= 6)),
        ["country", "house_no", "street_prefix"]))

    # Aggregate channels, budget top-K per query
    pooled = (
        pl.concat(channels)
        .group_by(["entity_id", "target_id"])
        .agg([pl.col("prio").max().alias("max_prio"), pl.len().alias("n_channels")])
        .sort(["entity_id", "max_prio", "n_channels"], descending=[False, True, True])
        .group_by("entity_id", maintain_order=True).head(top_k)
    )
    return pooled


# ── Feature extraction ────────────────────────────────────────────────────────
def extract_features(cands_df: pl.DataFrame, q_dict: dict, t_dict: dict) -> pd.DataFrame:
    rows_list = list(cands_df.select(["entity_id", "target_id", "max_prio", "n_channels"]).iter_rows(named=True))
    feat_records = []
    for row in rows_list:
        qid = str(row["entity_id"])
        tid = str(row["target_id"])
        q = q_dict.get(qid, {})
        t = t_dict.get(tid, {})

        qn  = q.get("clean_name", "") or ""
        tn  = t.get("clean_name", "") or ""
        qns = q.get("sorted_name", "") or ""
        tns = t.get("sorted_name", "") or ""
        qns2= q.get("stripped_name","") or ""
        tns2= t.get("stripped_name","") or ""
        qsp = q.get("spaceless_name","") or ""
        tsp = t.get("spaceless_name","") or ""
        qa  = q.get("std_address", "") or ""
        ta  = t.get("std_address", "") or ""
        qas = q.get("sorted_addr", "") or ""
        tas = t.get("sorted_addr", "") or ""
        qhno= q.get("house_no", "") or ""
        thno= t.get("house_no", "") or ""
        qpin= q.get("postal_code","") or ""
        tpin= t.get("postal_code","") or ""
        qfw = q.get("first_word","") or ""
        tfw = t.get("first_word","") or ""
        cty = (q.get("country","") or "").lower()
        is_s3 = 1 if tid.startswith("S3-") else 0

        feat_records.append({
            "entity_id": qid, "target_id": tid,
            "max_prio": float(row["max_prio"]),
            "n_channels": int(row["n_channels"]),
            "exact_clean_name":   int(qn == tn and qn != ""),
            "exact_sorted_name":  int(qns == tns and qns != ""),
            "exact_stripped_name":int(qns2 == tns2 and qns2 != ""),
            "exact_spaceless":    int(len(qsp) >= 5 and qsp == tsp),
            "spaceless_subset":   int(len(qsp) >= 5 and len(tsp) >= 5 and (qsp in tsp or tsp in qsp) and qsp != tsp),
            "exact_clean_addr":   int(qa == ta and len(qa) >= 10),
            "exact_sorted_addr":  int(qas == tas and len(qas) >= 12),
            "hno_match":          int(qhno == thno and len(qhno) >= 2),
            "pin_match":          int(qpin == tpin and len(qpin) >= 5),
            "fw_match":           int(qfw == tfw and len(qfw) >= 4),
            "name_jw":            JaroWinkler.similarity(qn, tn),
            "name_tok_sort":      rfuzz.token_sort_ratio(qn, tn) / 100.0,
            "name_tok_set":       rfuzz.token_set_ratio(qn, tn) / 100.0,
            "name_partial":       rfuzz.partial_ratio(qn, tn) / 100.0,
            "name_char2_jac":     char_ngram_jac(qn, tn, 2),
            "name_char3_jac":     char_ngram_jac(qn, tn, 3),
            "name_char4_jac":     char_ngram_jac(qn, tn, 4),
            "name_tok_jac":       tok_jac(qn, tn),
            "name_tok_ovlp":      tok_overlap(qn, tn),
            "addr_jw":            JaroWinkler.similarity(qa, ta) if qa and ta else 0.0,
            "addr_tok_sort":      rfuzz.token_sort_ratio(qa, ta) / 100.0 if qa and ta else 0.0,
            "addr_tok_set":       rfuzz.token_set_ratio(qa, ta) / 100.0 if qa and ta else 0.0,
            "addr_char3_jac":     char_ngram_jac(qa, ta, 3),
            "addr_tok_jac":       tok_jac(qas, tas),
            "addr_null_tgt":      int(not ta),
            "name_len_diff":      abs(len(qn) - len(tn)),
            "name_len_ratio":     min(len(qn), len(tn)) / max(len(qn), len(tn), 1),
            "addr_len_diff":      abs(len(qa) - len(ta)),
            "name_wc_diff":       abs(len(qn.split()) - len(tn.split())),
            "name_wc_ratio":      min(len(qn.split()), len(tn.split())) / max(len(qn.split()), len(tn.split()), 1),
            "name_prefix3":       int(qn[:3] == tn[:3] if len(qn) >= 3 and len(tn) >= 3 else 0),
            "name_prefix5":       int(qn[:5] == tn[:5] if len(qn) >= 5 and len(tn) >= 5 else 0),
            "hno_contradiction":  int(qhno != thno and bool(qhno) and bool(thno)),
            "pin_contradiction":  int(qpin != tpin and bool(qpin) and bool(tpin)),
            "pin_prefix3":        int(qpin[:3] == tpin[:3] if len(qpin) >= 3 and len(tpin) >= 3 else 0),
            "pin_both_present":   int(bool(qpin) and bool(tpin)),
            "name_x_addr":        JaroWinkler.similarity(qn, tn) * (rfuzz.token_sort_ratio(qa, ta)/100.0 if qa and ta else 0.0),
            "name_x_pin":         JaroWinkler.similarity(qn, tn) * float(qpin == tpin and len(qpin) >= 5),
            "sorted_x_addr":      (rfuzz.token_sort_ratio(qns, tns)/100.0) * (rfuzz.token_set_ratio(qa, ta)/100.0 if qa and ta else 0.0),
            "is_source3":         is_s3,
            "india_flag":         int("india" in cty),
            "france_flag":        int("france" in cty),
        })
    return pd.DataFrame(feat_records)


def main():
    flush("=" * 70)
    flush("AMAZON ML CHALLENGE 2026 — FINAL SUBMISSION (v3 shard-per-shard)")
    flush("=" * 70)

    # 1. Load models
    flush("[1] Loading trained models...")
    with open(RESULTS_JSON) as f:
        meta_res = json.load(f)
    best_theta = meta_res["best_theta"]
    best_cap   = meta_res.get("best_cap", False)
    feat_cols  = meta_res["features"]
    flush(f"  Threshold={best_theta} | Cap={best_cap} | Features={len(feat_cols)}")

    model_lgb  = pickle.load(open(MODEL_DIR / "meta_lgb.pkl",     "rb"))
    model_xgb  = pickle.load(open(MODEL_DIR / "meta_xgb.pkl",     "rb"))
    model_cb   = pickle.load(open(MODEL_DIR / "meta_cb.pkl",      "rb"))
    meta_model = pickle.load(open(MODEL_DIR / "meta_stacker.pkl", "rb"))
    flush("  All models loaded.")

    # 2. Load cached normalized data
    flush(f"[2] Loading cached test queries...")
    test_q_norm = pl.read_parquet(TEST_Q_CACHE)
    flush(f"  Test queries: {len(test_q_norm):,}")

    flush(f"[3] Loading cached test targets...")
    test_t_norm = pl.read_parquet(TEST_T_CACHE)
    flush(f"  Test targets: {len(test_t_norm):,}")

    # 3. Build feature dicts once (kept in RAM, ~3GB total, fits)
    flush("\n[4] Building feature lookup dicts...")
    FEAT_COLS_NEEDED = [
        "entity_id", "clean_name", "stripped_name", "sorted_name", "spaceless_name",
        "std_address", "sorted_addr", "house_no", "postal_code", "country", "first_word"
    ]
    q_dict = {str(r["entity_id"]): r for r in test_q_norm.select(FEAT_COLS_NEEDED).iter_rows(named=True)}
    t_dict = {str(r["entity_id"]): r for r in test_t_norm.select(FEAT_COLS_NEEDED).iter_rows(named=True)}
    flush(f"  Q={len(q_dict):,} | T={len(t_dict):,}")

    # 4. Per-shard: retrieve → budget → features → inference → collect predictions
    query_ids = test_q_norm["entity_id"].to_list()
    n_shards  = (len(query_ids) + SHARD_SIZE - 1) // SHARD_SIZE
    flush(f"\n[5] Shard-at-a-time pipeline: {len(query_ids):,} queries -> {n_shards} shards of {SHARD_SIZE:,}")
    flush(f"    Top-K={TOP_K} per query | Threshold={best_theta}")

    # Accumulate results across shards
    all_candidate_rows = []   # (query_id, target_id) for candidate_pairs.tsv
    all_match_rows     = []   # (query_id, matched_str) for matching_results.tsv
    t_pipeline_start = time.time()

    for shard_idx in range(n_shards):
        t_shard = time.time()
        start = shard_idx * SHARD_SIZE
        end   = min(start + SHARD_SIZE, len(query_ids))
        shard_ids = query_ids[start:end]
        q_shard = test_q_norm.filter(pl.col("entity_id").is_in(shard_ids))

        flush(f"\n  --- Shard {shard_idx+1}/{n_shards} ({len(q_shard):,} queries) ---")

        # 5a. Retrieval + budget
        shard_cands = retrieve_and_budget(q_shard, test_t_norm, TOP_K)
        flush(f"    Retrieval+budget: {len(shard_cands):,} pairs ({time.time()-t_shard:.1f}s)")

        # Collect candidate pairs for TSV
        for row in shard_cands.select(["entity_id", "target_id"]).iter_rows():
            all_candidate_rows.append(row)

        # 5b. Feature extraction
        t_feat = time.time()
        feat_df = extract_features(shard_cands, q_dict, t_dict)
        flush(f"    Feature extraction: {len(feat_df):,} rows ({time.time()-t_feat:.1f}s)")
        del shard_cands

        # 5c. GPU Inference
        t_inf = time.time()
        X = feat_df[feat_cols].values.astype(np.float32)
        p_lgb = model_lgb.predict_proba(X)[:, 1]
        p_xgb = model_xgb.predict_proba(X)[:, 1]
        p_cb  = model_cb.predict_proba(X)[:, 1]
        p_meta = meta_model.predict_proba(np.column_stack([p_lgb, p_xgb, p_cb]))[:, 1]
        flush(f"    Inference: {time.time()-t_inf:.1f}s")

        # 5d. Decision rule per query
        pred_map = defaultdict(list)
        for p, qid, tid in zip(p_meta, feat_df["entity_id"].values, feat_df["target_id"].values):
            pred_map[qid].append((float(p), tid))
        del feat_df, X, p_lgb, p_xgb, p_cb, p_meta

        for qid in shard_ids:
            qid_str = str(qid)
            preds = sorted(pred_map.get(qid_str, []), key=lambda x: x[0], reverse=True)
            matched = []
            if best_cap:
                s2c = s3c = 0
                for prob, tid in preds:
                    if prob >= best_theta:
                        if tid.startswith("S3-") and s3c < 1:
                            matched.append(tid); s3c += 1
                        elif not tid.startswith("S3-") and s2c < 1:
                            matched.append(tid); s2c += 1
            else:
                matched = [tid for prob, tid in preds if prob >= best_theta]
            all_match_rows.append((qid_str, ",".join(sorted(matched))))

        flush(f"    Shard {shard_idx+1} complete in {time.time()-t_shard:.1f}s | matches so far: {sum(1 for _,m in all_match_rows if m):,}")

    flush(f"\n  Full pipeline done in {time.time()-t_pipeline_start:.1f}s")

    # 6. Write candidate_pairs.tsv
    flush("\n[6] Writing output/candidate_pairs.tsv...")
    t0 = time.time()
    cand_map_agg = defaultdict(list)
    for qid, tid in all_candidate_rows:
        cand_map_agg[str(qid)].append(str(tid))

    all_s1_ids = [str(x) for x in test_q_norm["entity_id"].to_list()]
    cand_rows_out = [(qid, ",".join(cand_map_agg.get(qid, []))) for qid in all_s1_ids]

    cand_df = pl.DataFrame({
        "source1_entity_id":     [r[0] for r in cand_rows_out],
        "candidate_entity_ids":  [r[1] for r in cand_rows_out],
    })
    cand_tsv_path = OUT_DIR / "candidate_pairs.tsv"
    cand_df.write_csv(cand_tsv_path, separator="\t")
    flush(f"  candidate_pairs.tsv written ({len(cand_df):,} rows) in {time.time()-t0:.1f}s")

    # 7. Write matching_results.tsv
    flush("\n[7] Writing output/matching_results.tsv...")
    match_df = pl.DataFrame({
        "source1_entity_id": [r[0] for r in all_match_rows],
        "matched_entity_ids": [r[1] for r in all_match_rows],
    })
    match_tsv_path = OUT_DIR / "matching_results.tsv"
    match_df.write_csv(match_tsv_path, separator="\t")
    matched_count = (match_df["matched_entity_ids"] != "").sum()
    flush(f"  matching_results.tsv written ({len(match_df):,} rows)")
    flush(f"  Queries with at least one match: {matched_count:,} / {len(match_df):,} ({100*matched_count/len(match_df):.1f}%)")

    # 8. Official Validation
    flush("\n[8] Running official competition validator...")
    val_script = ROOT / "dataset" / "student_resource" / "utils" / "validate_submission.py"
    if val_script.exists():
        cmd = [sys.executable, str(val_script),
               "--matching",  str(match_tsv_path),
               "--candidate", str(cand_tsv_path),
               "--test-dir",  str(ROOT / "dataset" / "raw" / "test")]
        res = subprocess.run(cmd, capture_output=True, text=True)
        flush(res.stdout)
        if res.stderr:
            flush("STDERR: " + res.stderr[:800])
        flush(f"Validator exit code: {res.returncode}")
    else:
        flush("  validate_submission.py not found — skipping.")

    flush("\n" + "=" * 70)
    flush("SUBMISSION COMPLETE!")
    flush(f"  candidate_pairs.tsv  -> {cand_tsv_path}")
    flush(f"  matching_results.tsv -> {match_tsv_path}")
    flush("=" * 70)


if __name__ == "__main__":
    main()
