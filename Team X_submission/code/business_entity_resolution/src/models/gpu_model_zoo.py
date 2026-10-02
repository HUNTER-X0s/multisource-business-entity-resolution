"""
experiments/phase2/models/gpu_model_zoo.py
==========================================
EXP-12, EXP-13, EXP-14: GPU-Accelerated Model Zoo & Heterogeneous Ensemble.

Executes:
  1. EXP-12: XGBoost Pointwise on GPU (device="cuda", tree_method="hist")
  2. EXP-13: CatBoost Pointwise on GPU (task_type="GPU", devices="0")
  3. LightGBM Pointwise on CPU (n_jobs=-1, leaf-wise baseline)
  4. Real-time GPU telemetry measurement (nvidia-smi VRAM, CUDA fit times)
  5. EXP-14: Heterogeneous Tri-Model Ensemble (Leaf-Wise + Depth-Wise + Oblivious Trees)
"""

import sys
import os
import json
import time
import subprocess
from typing import Dict, Set, Tuple
import numpy as np
import polars as pl
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.abspath("."))
from src.evaluate import macro_f05_from_dicts
try:
    from src.validation_framework import load_validation_dataset, evaluate_matching_predictions
except ImportError:
    try:
        from validation_framework import load_validation_dataset, evaluate_matching_predictions
    except ImportError:
        load_validation_dataset, evaluate_matching_predictions = None, None

SEED = 2026
np.random.seed(SEED)

FEATURE_COLS = [
    "exact_clean_name", "exact_stripped_name", "exact_sorted_name",
    "name_jaro_winkler", "name_token_sort", "name_token_set",
    "name_len_diff", "name_len_ratio",
    "exact_clean_address", "addr_is_null_target", "addr_jaro_winkler",
    "addr_token_jaccard", "addr_token_overlap", "numeric_token_jaccard",
    "house_number_match", "contradiction_house_no",
    "postal_code_match", "postal_both_present", "contradiction_postal",
    "is_source3", "channel_max_priority", "channel_count",
    "name_x_addr", "name_x_postal",
]


def get_gpu_memory_used_mb() -> float:
    """Reads current GPU memory used in MB via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
            encoding="utf-8"
        )
        return float(out.strip().split("\n")[0])
    except Exception:
        return 0.0


def main():
    print("=" * 75, flush=True)
    print("PHASE 2B: GPU MODEL ZOO & HETEROGENEOUS ENSEMBLE BENCHMARK", flush=True)
    print("=" * 75, flush=True)
    t_global_start = time.time()

    print(f"CUDA Available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"Initial GPU Memory: {get_gpu_memory_used_mb():.1f} MB", flush=True)

    # 1. Load data
    val_path = "experiments/val_sample_50k.parquet"
    print(f"\n[1/5] Loading ground truth from {val_path}...", flush=True)
    val_df, gt_dict = load_validation_dataset(val_path)

    feat_path = "experiments/phase2/val_50k_features.parquet"
    print(f"[2/5] Loading 24 candidate features from {feat_path}...", flush=True)
    feats = pl.read_parquet(feat_path)
    total_pairs = len(feats)
    print(f"      Loaded {total_pairs:,} candidate pairs.", flush=True)

    qid_col = "entity_id"
    cid_col = "target_id"

    # Deterministic query split (Method B: sorted unique queries for exact reproducibility)
    unique_queries = np.array(sorted(feats[qid_col].unique().to_list()))
    rng = np.random.RandomState(SEED)
    rng.shuffle(unique_queries)

    n_train = int(len(unique_queries) * 0.7)
    train_queries = set(unique_queries[:n_train])
    holdout_queries = set(unique_queries[n_train:])
    print(f"      Split: {len(train_queries):,} train queries | {len(holdout_queries):,} holdout queries.", flush=True)

    feats_pd = feats.to_pandas()
    train_mask = feats_pd[qid_col].isin(train_queries)
    test_mask = feats_pd[qid_col].isin(holdout_queries)

    train_df = feats_pd[train_mask].copy()
    test_df = feats_pd[test_mask].copy()

    def label_row(row):
        return int(row[cid_col] in gt_dict.get(row[qid_col], set()))

    print("      Labeling candidate pairs...", flush=True)
    train_df["label"] = train_df.apply(label_row, axis=1)
    test_df["label"] = test_df.apply(label_row, axis=1)

    X_train = train_df[FEATURE_COLS].values.astype(np.float32)
    y_train = train_df["label"].values.astype(np.int32)
    X_test = test_df[FEATURE_COLS].values.astype(np.float32)
    y_test = test_df["label"].values.astype(np.int32)

    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    print(f"      Train Matrix: {X_train.shape} ({n_pos:,} pos, {n_neg:,} neg | ratio 1:{n_neg/n_pos:.1f})", flush=True)
    print(f"      Test Matrix:  {X_test.shape} ({(y_test==1).sum():,} pos, {(y_test==0).sum():,} neg)", flush=True)

    holdout_gt = {sid: gt_dict[sid] for sid in holdout_queries if sid in gt_dict}
    holdout_s1 = val_df.filter(pl.col("entity_id").is_in(list(holdout_queries)))

    results = {}

    # ─────────────────────────────────────────────────────────────────────────
    # MODEL 1: LightGBM Pointwise (CPU Baseline)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70, flush=True)
    print("[MODEL 1/3] Training LightGBM Pointwise (CPU Multithreaded OpenMP)...", flush=True)
    print("=" * 70, flush=True)
    t0 = time.time()
    lgb_model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        objective="binary",
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_model.fit(X_train, y_train)
    lgb_fit_time = time.time() - t0
    lgb_probs = lgb_model.predict_proba(X_test)[:, 1]
    lgb_roc = roc_auc_score(y_test, lgb_probs)
    lgb_pr = average_precision_score(y_test, lgb_probs)
    print(f"  LightGBM Fit Time: {lgb_fit_time:.2f}s | ROC-AUC: {lgb_roc:.5f} | PR-AUC: {lgb_pr:.5f}", flush=True)

    # ─────────────────────────────────────────────────────────────────────────
    # MODEL 2: EXP-12 — XGBoost Pointwise on GPU (CUDA)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70, flush=True)
    print("[MODEL 2/3: EXP-12] Training XGBoost Pointwise on GPU (device='cuda')...", flush=True)
    print("=" * 70, flush=True)
    mem_before_xgb = get_gpu_memory_used_mb()
    t0 = time.time()
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        tree_method="hist",
        device="cuda",
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=SEED,
    )
    xgb_model.fit(X_train, y_train)
    xgb_fit_time = time.time() - t0
    mem_during_xgb = get_gpu_memory_used_mb()
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    xgb_roc = roc_auc_score(y_test, xgb_probs)
    xgb_pr = average_precision_score(y_test, xgb_probs)
    print(f"  XGBoost GPU Fit Time: {xgb_fit_time:.2f}s | ROC-AUC: {xgb_roc:.5f} | PR-AUC: {xgb_pr:.5f}", flush=True)
    print(f"  GPU VRAM: Before={mem_before_xgb:.1f} MB -> Active={mem_during_xgb:.1f} MB (Delta: +{mem_during_xgb-mem_before_xgb:.1f} MB)", flush=True)

    # ─────────────────────────────────────────────────────────────────────────
    # MODEL 3: EXP-13 — CatBoost Pointwise on GPU (CUDA)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70, flush=True)
    print("[MODEL 3/3: EXP-13] Training CatBoost Pointwise on GPU (task_type='GPU')...", flush=True)
    print("=" * 70, flush=True)
    mem_before_cb = get_gpu_memory_used_mb()
    t0 = time.time()
    cb_model = cb.CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        task_type="GPU",
        loss_function="Logloss",
        random_seed=SEED,
        verbose=0,
    )
    cb_model.fit(X_train, y_train)
    cb_fit_time = time.time() - t0
    mem_during_cb = get_gpu_memory_used_mb()
    cb_probs = cb_model.predict_proba(X_test)[:, 1]
    cb_roc = roc_auc_score(y_test, cb_probs)
    cb_pr = average_precision_score(y_test, cb_probs)
    print(f"  CatBoost GPU Fit Time: {cb_fit_time:.2f}s | ROC-AUC: {cb_roc:.5f} | PR-AUC: {cb_pr:.5f}", flush=True)
    print(f"  GPU VRAM: Before={mem_before_cb:.1f} MB -> Active={mem_during_cb:.1f} MB (Delta: +{mem_during_cb-mem_before_cb:.1f} MB)", flush=True)

    # Correlation Matrix between Models
    print("\n" + "=" * 70, flush=True)
    print("MODEL PREDICTION CORRELATION MATRIX (DIVERSITY AUDIT):", flush=True)
    print("=" * 70, flush=True)
    corr_lx = np.corrcoef(lgb_probs, xgb_probs)[0, 1]
    corr_lc = np.corrcoef(lgb_probs, cb_probs)[0, 1]
    corr_xc = np.corrcoef(xgb_probs, cb_probs)[0, 1]
    print(f"  r(LightGBM, XGBoost GPU):  {corr_lx:.5f}", flush=True)
    print(f"  r(LightGBM, CatBoost GPU): {corr_lc:.5f}", flush=True)
    print(f"  r(XGBoost,  CatBoost GPU): {corr_xc:.5f}", flush=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Standalone Model Evaluations across Thresholds
    # ─────────────────────────────────────────────────────────────────────────
    test_eval_df = test_df[[qid_col, cid_col]].copy()
    thresholds = [0.45, 0.50, 0.55, 0.60, 0.65]

    def evaluate_model_thresholds(name, probs):
        test_eval_df["score"] = probs
        best_t, best_f = 0.55, 0.0
        best_ev = {}
        for t in thresholds:
            matched = test_eval_df[test_eval_df["score"] >= t]
            preds = {}
            for qid, grp in matched.groupby(qid_col):
                preds[qid] = set(grp[cid_col].tolist())
            for qid in holdout_queries:
                if qid not in preds:
                    preds[qid] = set()
            ev = macro_f05_from_dicts(gt_dict=holdout_gt, pred_dict=preds, s1_ids=list(holdout_queries))
            f05 = ev["macro_f05"]
            if f05 > best_f:
                best_f, best_t = f05, t
                best_ev = ev
        print(f"  {name:<25}: Optimal theta*={best_t:.2f} -> Macro F0.5 = {best_f:.5f} (Prec={best_ev['mean_precision']:.4f}, Rec={best_ev['mean_recall']:.4f})", flush=True)
        return best_t, best_f, best_ev

    print("\n--- STANDALONE BENCHMARKS ---", flush=True)
    t_lgb, f_lgb, ev_lgb = evaluate_model_thresholds("LightGBM (CPU)", lgb_probs)
    t_xgb, f_xgb, ev_xgb = evaluate_model_thresholds("EXP-12: XGBoost (GPU)", xgb_probs)
    t_cb, f_cb, ev_cb   = evaluate_model_thresholds("EXP-13: CatBoost (GPU)", cb_probs)

    # ─────────────────────────────────────────────────────────────────────────
    # EXP-14: Heterogeneous Tri-Model Ensemble Grid Search
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70, flush=True)
    print("[EXP-14] HETEROGENEOUS TRI-MODEL ENSEMBLE OPTIMIZATION", flush=True)
    print("=" * 70, flush=True)

    weights_to_test = [
        # (w_lgb, w_xgb, w_cb)
        (0.50, 0.50, 0.00),  # Baseline EXP-11 50/50 blend
        (0.34, 0.33, 0.33),  # Equal 3-way blend
        (0.40, 0.40, 0.20),  # Heavy tree, light CatBoost
        (0.45, 0.45, 0.10),
        (0.30, 0.40, 0.30),
        (0.40, 0.30, 0.30),
        (0.25, 0.50, 0.25),
        (0.20, 0.60, 0.20),
    ]

    best_ensemble = {"f05": 0.0, "weights": None, "theta": 0.55}

    for w_l, w_x, w_c in weights_to_test:
        p_ens = w_l * lgb_probs + w_x * xgb_probs + w_c * cb_probs
        test_eval_df["score"] = p_ens
        for t in [0.50, 0.55, 0.60, 0.65]:
            matched = test_eval_df[test_eval_df["score"] >= t]
            preds = {}
            for qid, grp in matched.groupby(qid_col):
                preds[qid] = set(grp[cid_col].tolist())
            for qid in holdout_queries:
                if qid not in preds:
                    preds[qid] = set()
            ev = macro_f05_from_dicts(gt_dict=holdout_gt, pred_dict=preds, s1_ids=list(holdout_queries))
            f05 = ev["macro_f05"]
            if f05 > best_ensemble["f05"]:
                best_ensemble = {
                    "f05": f05,
                    "weights": (w_l, w_x, w_c),
                    "theta": t,
                    "precision": ev["mean_precision"],
                    "recall": ev["mean_recall"],
                }
            label = f"W: LGB={w_l:.2f}, XGB={w_x:.2f}, CB={w_c:.2f} | theta={t:.2f}"
            marker = " *** BEST ***" if f05 == best_ensemble["f05"] else ""
            if t in [0.55, 0.60]:
                print(f"  {label:<45} -> F0.5 = {f05:.5f}{marker}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("OPTIMAL HETEROGENEOUS TRI-MODEL ENSEMBLE:", flush=True)
    w_best = best_ensemble["weights"]
    print(f"  Weights: LightGBM={w_best[0]:.2f}, XGBoost-GPU={w_best[1]:.2f}, CatBoost-GPU={w_best[2]:.2f}", flush=True)
    print(f"  Optimal theta*: {best_ensemble['theta']:.2f}", flush=True)
    print(f"  Macro F0.5:     {best_ensemble['f05']:.5f}", flush=True)
    print("=" * 70, flush=True)

    # Official Full Slice Evaluation of Best Tri-Model Ensemble
    p_best = w_best[0] * lgb_probs + w_best[1] * xgb_probs + w_best[2] * cb_probs
    test_eval_df["score"] = p_best
    best_preds = {}
    for qid, grp in test_eval_df[test_eval_df["score"] >= best_ensemble["theta"]].groupby(qid_col):
        best_preds[qid] = set(grp[cid_col].tolist())
    for qid in holdout_queries:
        if qid not in best_preds:
            best_preds[qid] = set()

    print("\n--- OFFICIAL FULL EVALUATION OF BEST TRI-MODEL ENSEMBLE ---", flush=True)
    final_eval = evaluate_matching_predictions(best_preds, holdout_gt, s1_df=holdout_s1, verbose=True)

    # Save output artifacts
    elapsed = time.time() - t_global_start
    output = {
        "experiment": "EXP-12_13_14",
        "description": "GPU Model Zoo: XGBoost GPU + CatBoost GPU + LightGBM Tri-Model Ensemble",
        "gpu_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "models": {
            "lightgbm_cpu": {"fit_time_sec": lgb_fit_time, "roc_auc": lgb_roc, "pr_auc": lgb_pr, "f05": f_lgb, "theta": t_lgb},
            "xgboost_gpu": {"fit_time_sec": xgb_fit_time, "roc_auc": xgb_roc, "pr_auc": xgb_pr, "f05": f_xgb, "theta": t_xgb, "vram_mb": mem_during_xgb},
            "catboost_gpu": {"fit_time_sec": cb_fit_time, "roc_auc": cb_roc, "pr_auc": cb_pr, "f05": f_cb, "theta": t_cb, "vram_mb": mem_during_cb},
        },
        "correlations": {
            "r_lgb_xgb": corr_lx,
            "r_lgb_cb": corr_lc,
            "r_xgb_cb": corr_xc,
        },
        "optimal_ensemble": {
            "weights": {"lightgbm": w_best[0], "xgboost_gpu": w_best[1], "catboost_gpu": w_best[2]},
            "optimal_theta": best_ensemble["theta"],
            "macro_f05": best_ensemble["f05"],
        },
        "final_metrics": {k: float(v) if isinstance(v, (float, int, np.floating)) else str(v)
                          for k, v in final_eval.items() if not isinstance(v, dict)},
        "elapsed_seconds": elapsed,
    }

    out_file = "experiments/phase2/exp12_14_gpu_zoo_results.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults successfully saved to {out_file} in {elapsed:.2f}s", flush=True)


if __name__ == "__main__":
    main()
