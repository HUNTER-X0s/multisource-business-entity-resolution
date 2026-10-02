"""
experiments/phase2/models/ensemble_pointwise.py
================================================
EXP-11: Pointwise Ensemble — LightGBM + XGBoost Binary Classifiers.

Hypothesis:
  EXP-03 established that Pointwise Binary Classification strongly outperforms
  Learning-to-Rank (F0.5 = 0.7786 vs 0.7711).
  However, LightGBM uses leaf-wise (best-first) tree growth, which can overfit
  specific deep boundary leaves.
  XGBoost uses depth-wise tree growth with exact/hist split regularization.
  A convex combination of their calibrated predicted probabilities:
    P_ens = alpha * P_lgbm + (1 - alpha) * P_xgb
  will reduce prediction variance, improve calibration on match_bin_1, and
  safeguard singleton accuracy.

Features: Compact, verified 24-feature matrix (EXP-03 standard).
"""

import sys
import os
import json
import time
from typing import Dict, Set
import numpy as np
import polars as pl
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.abspath("."))
from src.evaluate import macro_f05_from_dicts
from experiments.phase2.validation_framework import load_validation_dataset, evaluate_matching_predictions

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


def run_ensemble():
    print("=" * 70, flush=True)
    print("EXP-11: POINTWISE ENSEMBLE (LIGHTGBM + XGBOOST BINARY CLASSIFIERS)", flush=True)
    print("=" * 70, flush=True)
    t_start = time.time()

    val_path = "experiments/val_sample_50k.parquet"
    print(f"Loading ground truth from {val_path}...", flush=True)
    val_df, gt_dict = load_validation_dataset(val_path)

    feat_path = "experiments/phase2/val_50k_features.parquet"
    print(f"Loading candidate features from {feat_path}...", flush=True)
    feats = pl.read_parquet(feat_path)
    print(f"Loaded {len(feats):,} candidate pairs.", flush=True)

    qid_col = "entity_id"
    cid_col = "target_id"

    # Reproducible 70/30 Query-Level Train/Holdout Split
    unique_queries = np.array(sorted(feats[qid_col].unique().to_list()))
    rng = np.random.RandomState(SEED)
    rng.shuffle(unique_queries)

    n_train = int(len(unique_queries) * 0.7)
    train_queries = set(unique_queries[:n_train])
    holdout_queries = set(unique_queries[n_train:])
    print(f"Query Split: {len(train_queries):,} train | {len(holdout_queries):,} eval", flush=True)

    feats_pd = feats.to_pandas()
    train_mask = feats_pd[qid_col].isin(train_queries)
    test_mask = feats_pd[qid_col].isin(holdout_queries)

    train_df = feats_pd[train_mask].copy()
    test_df = feats_pd[test_mask].copy()

    # Ground-truth binary labeling
    def label_row(row):
        return int(row[cid_col] in gt_dict.get(row[qid_col], set()))

    print("Labeling pairs...", flush=True)
    train_df["label"] = train_df.apply(label_row, axis=1)
    test_df["label"] = test_df.apply(label_row, axis=1)

    X_train = train_df[FEATURE_COLS].values.astype(np.float32)
    y_train = train_df["label"].values
    X_test = test_df[FEATURE_COLS].values.astype(np.float32)
    y_test = test_df["label"].values

    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    scale_pos = n_neg / max(n_pos, 1)
    print(f"Train set: {len(train_df):,} pairs ({n_pos:,} pos, {n_neg:,} neg, ratio 1:{scale_pos:.1f})", flush=True)

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Train LightGBM Pointwise (EXP-03 Champion Configuration)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[1/3] Training LightGBM Pointwise (n_est=300, num_leaves=31, lr=0.05)...", flush=True)
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
    lgb_proba = lgb_model.predict_proba(X_test)[:, 1]
    lgb_roc = roc_auc_score(y_test, lgb_proba)
    lgb_pr = average_precision_score(y_test, lgb_proba)
    print(f"      LightGBM trained in {time.time()-t0:.2f}s | ROC-AUC: {lgb_roc:.5f} | PR-AUC: {lgb_pr:.5f}", flush=True)

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Train XGBoost Pointwise (Hist Gradient Boosting, Depth-Wise)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[2/3] Training XGBoost Pointwise (tree_method='hist', max_depth=6, lr=0.05, n_est=300)...", flush=True)
    t0 = time.time()
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        tree_method="hist",
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=-1,
    )
    xgb_model.fit(X_train, y_train)
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    xgb_roc = roc_auc_score(y_test, xgb_proba)
    xgb_pr = average_precision_score(y_test, xgb_proba)
    print(f"      XGBoost trained in {time.time()-t0:.2f}s | ROC-AUC: {xgb_roc:.5f} | PR-AUC: {xgb_pr:.5f}", flush=True)

    # Correlation between model predictions
    corr = np.corrcoef(lgb_proba, xgb_proba)[0, 1]
    print(f"\nModel Prediction Pearson Correlation: {corr:.4f}", flush=True)

    holdout_gt = {sid: gt_dict[sid] for sid in holdout_queries if sid in gt_dict}
    holdout_s1 = val_df.filter(pl.col("entity_id").is_in(list(holdout_queries)))

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Grid Search over Ensembling Weight alpha and Decision Threshold theta
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[3/3] Evaluating Ensembles across alpha in [0.0, 1.0] and theta in [0.45, 0.70]...", flush=True)
    alphas = [1.0, 0.75, 0.60, 0.50, 0.40, 0.25, 0.0]  # 1.0 = Pure LGBM, 0.0 = Pure XGB
    thresholds = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

    best_score = {"macro_f05": 0.0, "alpha": 1.0, "theta": 0.55}
    grid_summary = []

    # Fast evaluation helper using dictionary lookups
    test_eval_df = test_df[[qid_col, cid_col]].copy()

    for alpha in alphas:
        p_ens = alpha * lgb_proba + (1.0 - alpha) * xgb_proba
        test_eval_df["score"] = p_ens

        for theta in thresholds:
            matched_df = test_eval_df[test_eval_df["score"] >= theta]
            preds = {}
            for qid, grp in matched_df.groupby(qid_col):
                preds[qid] = set(grp[cid_col].tolist())
            for qid in holdout_queries:
                if qid not in preds:
                    preds[qid] = set()

            res = macro_f05_from_dicts(gt_dict=holdout_gt, pred_dict=preds, s1_ids=list(holdout_queries))
            f05 = res["macro_f05"]
            prec = res["mean_precision"]
            rec = res["mean_recall"]

            grid_summary.append({
                "alpha_lgb": alpha,
                "theta": theta,
                "macro_f05": f05,
                "mean_precision": prec,
                "mean_recall": rec,
            })

            label = f"alpha={alpha:.2f} (LGB={alpha:.0%}/XGB={1-alpha:.0%}) | theta={theta:.2f}"
            marker = " *** BEST ***" if f05 > best_score["macro_f05"] else ""
            if theta in [0.55, 0.60] or marker:
                print(f"  {label:<45} -> F0.5 = {f05:.5f} | Prec = {prec:.4f} | Rec = {rec:.4f}{marker}", flush=True)

            if f05 > best_score["macro_f05"]:
                best_score = {
                    "macro_f05": f05,
                    "alpha": alpha,
                    "theta": theta,
                    "precision": prec,
                    "recall": rec,
                }

    print("\n" + "=" * 70, flush=True)
    print(f"OPTIMAL ENSEMBLE CONFIGURATION:", flush=True)
    print(f"  Best alpha (LGB weight): {best_score['alpha']:.2f} (XGB weight: {1-best_score['alpha']:.2f})", flush=True)
    print(f"  Best theta*:            {best_score['theta']:.2f}", flush=True)
    print(f"  Macro F0.5:             {best_score['macro_f05']:.5f}", flush=True)
    print(f"  EXP-03 Champion:        0.77856", flush=True)
    delta = best_score["macro_f05"] - 0.77856
    print(f"  Delta vs Champion:      {delta:+.5f}", flush=True)
    print("=" * 70, flush=True)

    # Full official slice evaluation of optimal ensemble
    best_p = best_score["alpha"] * lgb_proba + (1.0 - best_score["alpha"]) * xgb_proba
    test_eval_df["score"] = best_p
    best_preds = {}
    for qid, grp in test_eval_df[test_eval_df["score"] >= best_score["theta"]].groupby(qid_col):
        best_preds[qid] = set(grp[cid_col].tolist())
    for qid in holdout_queries:
        if qid not in best_preds:
            best_preds[qid] = set()

    print("\n--- OFFICIAL FULL EVALUATION OF BEST CONFIGURATION ---", flush=True)
    final_eval = evaluate_matching_predictions(best_preds, holdout_gt, s1_df=holdout_s1, verbose=True)

    elapsed = time.time() - t_start
    output = {
        "experiment": "EXP-11",
        "model": "Pointwise Ensemble (LightGBM + XGBoost)",
        "lgb_roc_auc": float(lgb_roc),
        "lgb_pr_auc": float(lgb_pr),
        "xgb_roc_auc": float(xgb_roc),
        "xgb_pr_auc": float(xgb_pr),
        "prediction_correlation": float(corr),
        "optimal_alpha_lgb": best_score["alpha"],
        "optimal_theta": best_score["theta"],
        "macro_f05": best_score["macro_f05"],
        "exp03_champion_f05": 0.77856,
        "delta_vs_champion": delta,
        "grid_summary": grid_summary,
        "final_metrics": {k: float(v) if isinstance(v, (float, int, np.floating)) else str(v)
                          for k, v in final_eval.items() if not isinstance(v, dict)},
        "elapsed_seconds": elapsed,
    }

    out_path = "experiments/phase2/exp11_ensemble_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path} in {elapsed:.2f}s", flush=True)


if __name__ == "__main__":
    run_ensemble()
