"""
scratch/generate_final_submission.py
====================================
End-to-End Competition Submission Generator:
1. Normalizes test queries and 9.97M test targets.
2. Runs 10-channel deterministic Polars retrieval.
3. Exports competition-compliant output/candidate_pairs.tsv.
4. Extracts 40+ tabular features on test candidate pairs.
5. Predicts with trained GPU Meta-Ensemble (LightGBM + XGBoost-GPU + CatBoost-GPU + Meta-Stacker).
6. Applies precision-optimal decision rule (per-source capping & tuned threshold).
7. Exports competition-compliant output/matching_results.tsv.
8. Runs official validation script (validate_submission.py).
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

MODEL_DIR = ROOT / "experiments" / "phase2" / "models"
RESULTS_JSON = ROOT / "experiments" / "phase2" / "meta_ensemble_results.json"

TEST_S1_FILE = ROOT / "dataset/raw/test/test_source1.tsv"
TEST_S2_FILE = ROOT / "dataset/raw/test/test_source2.tsv"
TEST_S3_FILE = ROOT / "dataset/raw/test/test_source3.tsv"

TEST_Q_CACHE = CACHE_DIR / "test_queries_norm.parquet"
TEST_T_CACHE = CACHE_DIR / "test_targets_norm.parquet"

sys.path.append(str(ROOT / "scratch"))
from benchmark_10ch_retrieval import normalize_text, run_10_channels
from train_gpu_meta_ensemble import char_ngram_jac, tok_jac, tok_overlap

def flush(msg, end="\n"):
    print(msg, end=end, flush=True)

def main():
    flush("=" * 70)
    flush("AMAZON ML CHALLENGE 2026 — FINAL TEST SUBMISSION GENERATOR")
    flush("=" * 70)

    # 1. Load Trained Models & Parameters
    flush("[1] Loading trained models and decision parameters...")
    if not RESULTS_JSON.exists():
        flush(f"ERROR: {RESULTS_JSON} not found. Train meta-ensemble first!")
        return

    with open(RESULTS_JSON, "r") as f:
        meta_res = json.load(f)
    best_theta = meta_res["best_theta"]
    best_cap = meta_res.get("best_cap", True)
    feat_cols = meta_res["features"]
    flush(f"  Optimal Threshold: {best_theta} | Per-source Cap: {best_cap} | Features: {len(feat_cols)}")

    model_lgb = pickle.load(open(MODEL_DIR / "meta_lgb.pkl", "rb"))
    model_xgb = pickle.load(open(MODEL_DIR / "meta_xgb.pkl", "rb"))
    model_cb  = pickle.load(open(MODEL_DIR / "meta_cb.pkl", "rb"))
    meta_model = pickle.load(open(MODEL_DIR / "meta_stacker.pkl", "rb"))
    flush("  All models loaded successfully.")

    # 2. Normalize Test Queries
    if TEST_Q_CACHE.exists():
        flush(f"[2] Loading cached normalized test queries from {TEST_Q_CACHE}...")
        test_q_norm = pl.read_parquet(TEST_Q_CACHE)
    else:
        flush("[2] Loading and normalizing test queries...")
        t0 = time.time()
        test_s1 = pl.read_csv(TEST_S1_FILE, separator="\t", quote_char=None)
        test_q_norm = normalize_text(test_s1)
        test_q_norm.write_parquet(TEST_Q_CACHE)
        flush(f"  Test queries normalized in {time.time()-t0:.1f}s and cached.")

    # 3. Normalize Test Targets
    if TEST_T_CACHE.exists():
        flush(f"[3] Loading cached normalized test targets from {TEST_T_CACHE}...")
        test_t_norm = pl.read_parquet(TEST_T_CACHE)
    else:
        flush("[3] Loading and normalizing test targets (S2 + S3)...")
        t0 = time.time()
        s2 = pl.read_csv(TEST_S2_FILE, separator="\t", quote_char=None)
        s3 = pl.read_csv(TEST_S3_FILE, separator="\t", quote_char=None)
        test_targets = pl.concat([s2, s3])
        test_t_norm = normalize_text(test_targets)
        test_t_norm.write_parquet(TEST_T_CACHE)
        flush(f"  Test targets normalized in {time.time()-t0:.1f}s and cached.")

    # 4. Run 10-Channel Retrieval
    flush("\n[4] Running 10-channel deterministic candidate retrieval on test set...")
    t_ret = time.time()
    test_cands = run_10_channels(test_q_norm, test_t_norm)
    flush(f"  Retrieval finished in {time.time()-t_ret:.1f}s. Unique pairs: {len(test_cands):,}")

    # 5. Export output/candidate_pairs.tsv
    flush("\n[5] Exporting output/candidate_pairs.tsv...")
    t_exp = time.time()
    cand_sorted = test_cands.sort(["entity_id", "max_prio", "n_channels"], descending=[False, True, True])
    # Budget top-40 candidates per query for candidate_pairs.tsv
    cand_budgeted = cand_sorted.group_by("entity_id", maintain_order=True).head(40)
    
    # Group candidates by query
    cand_map = (
        cand_budgeted.group_by("entity_id")
        .agg(pl.col("target_id").list.join(",").alias("candidate_entity_ids"))
        .rename({"entity_id": "source1_entity_id"})
    )
    
    # Ensure all test queries are present (even if 0 candidates)
    all_s1_ids = pl.DataFrame({"source1_entity_id": test_q_norm["entity_id"]})
    cand_tsv_df = all_s1_ids.join(cand_map, on="source1_entity_id", how="left").with_columns(
        pl.col("candidate_entity_ids").fill_null("")
    )
    cand_tsv_path = OUT_DIR / "candidate_pairs.tsv"
    cand_tsv_df.write_csv(cand_tsv_path, separator="\t")
    flush(f"  candidate_pairs.tsv written ({len(cand_tsv_df):,} rows) in {time.time()-t_exp:.1f}s.")

    # 6. Extract Tabular Features for Candidate Pairs
    flush("\n[6] Building feature matrix for test candidate pairs...")
    t_feat = time.time()
    
    q_dict = {str(r["entity_id"]): r for r in test_q_norm.select([
        "entity_id", "clean_name", "stripped_name", "sorted_name", "spaceless_name",
        "std_address", "sorted_addr", "house_no", "postal_code", "country", "first_word"
    ]).iter_rows(named=True)}

    target_ids_needed = set(test_cands["target_id"].to_list())
    flush(f"  Subsetting {len(target_ids_needed):,} unique target entities...")
    t_sub = test_t_norm.filter(pl.col("entity_id").is_in(list(target_ids_needed)))
    t_dict = {str(r["entity_id"]): r for r in t_sub.select([
        "entity_id", "clean_name", "stripped_name", "sorted_name", "spaceless_name",
        "std_address", "sorted_addr", "house_no", "postal_code", "country", "first_word"
    ]).iter_rows(named=True)}

    rows_list = list(test_cands.select(["entity_id", "target_id", "max_prio", "n_channels"]).iter_rows(named=True))
    total_cands = len(rows_list)
    flush(f"  Extracting features for {total_cands:,} candidate pairs...")

    feat_records = []
    for i, row in enumerate(rows_list):
        if i % 500000 == 0 and i > 0:
            flush(f"    [{i:,}/{total_cands:,}] {time.time()-t_feat:.1f}s")
        
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

    test_feat_df = pd.DataFrame(feat_records)
    flush(f"  Feature extraction complete in {time.time()-t_feat:.1f}s.")

    # 7. GPU Meta-Ensemble Inference
    flush("\n[7] Running Meta-Ensemble Inference...")
    t_inf = time.time()
    X_test = test_feat_df[feat_cols].values.astype(np.float32)

    p_lgb = model_lgb.predict_proba(X_test)[:, 1]
    flush(f"  LightGBM inference done ({time.time()-t_inf:.1f}s)")
    
    t_xgb = time.time()
    p_xgb = model_xgb.predict_proba(X_test)[:, 1]
    flush(f"  XGBoost-GPU inference done ({time.time()-t_xgb:.1f}s)")
    
    t_cb = time.time()
    p_cb = model_cb.predict_proba(X_test)[:, 1]
    flush(f"  CatBoost-GPU inference done ({time.time()-t_cb:.1f}s)")
    
    meta_X_test = np.column_stack([p_lgb, p_xgb, p_cb])
    p_meta = meta_model.predict_proba(meta_X_test)[:, 1]
    flush(f"  Meta-stacker blending complete. Total inference time: {time.time()-t_inf:.1f}s.")

    # 8. Decision Rule & Output Matching Results
    flush("\n[8] Applying precision-optimal decision rule...")
    q_arr = test_feat_df["entity_id"].values
    t_arr = test_feat_df["target_id"].values
    
    pred_map = defaultdict(list)
    for p, q, t in zip(p_meta, q_arr, t_arr):
        pred_map[q].append((p, t))

    matched_records = []
    all_s1_set = set(test_q_norm["entity_id"].to_list())

    for qid in sorted(all_s1_set):
        preds_q = pred_map.get(qid, [])
        sorted_cands = sorted(preds_q, key=lambda x: x[0], reverse=True)
        
        matched = []
        if best_cap:
            s2_found = 0
            s3_found = 0
            for p, t in sorted_cands:
                if p >= best_theta:
                    is_s3 = t.startswith("S3-")
                    if is_s3 and s3_found < 1:
                        matched.append(t)
                        s3_found += 1
                    elif not is_s3 and s2_found < 1:
                        matched.append(t)
                        s2_found += 1
        else:
            matched = [t for (p, t) in sorted_cands if p >= best_theta]
            
        matched_str = ",".join(sorted(matched)) if matched else ""
        matched_records.append((qid, matched_str))

    match_df = pl.DataFrame({
        "source1_entity_id": [r[0] for r in matched_records],
        "matched_entity_ids": [r[1] for r in matched_records]
    })
    
    match_tsv_path = OUT_DIR / "matching_results.tsv"
    match_df.write_csv(match_tsv_path, separator="\t")
    flush(f"\n[8] Successfully written matching_results.tsv to {match_tsv_path} ({len(match_df):,} rows)")

    # 9. Official Validation
    flush("\n[9] Running official competition validator...")
    val_script = ROOT / "dataset" / "student_resource" / "utils" / "validate_submission.py"
    if val_script.exists():
        cmd = [
            sys.executable, str(val_script),
            "--matching", str(match_tsv_path),
            "--candidate", str(cand_tsv_path),
            "--test-dir", str(ROOT / "dataset" / "raw" / "test"),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        flush(res.stdout)
        if res.stderr:
            flush("Validator STDERR:")
            flush(res.stderr)
        flush(f"Validation Exit Code: {res.returncode}")

    flush("\n" + "=" * 70)
    flush("SUBMISSION PIPELINE COMPLETE!")
    flush("=" * 70)

if __name__ == "__main__":
    main()
