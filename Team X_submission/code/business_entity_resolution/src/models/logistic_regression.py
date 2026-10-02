"""
experiments/phase2/models/logistic_regression.py
================================================
EXP-02: L2-Regularized Logistic Regression on 24 Pairwise Features.
Evaluates linear feature combinatorics, calibrates probability outputs,
tunes decision threshold for Macro F0.5, and inspects learned feature weights.
"""

import sys
import os
import json
import time
from typing import Dict, Set, List
import polars as pl
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.abspath("."))
try:
    from src.validation_framework import load_validation_dataset, evaluate_matching_predictions
except ImportError:
    try:
        from validation_framework import load_validation_dataset, evaluate_matching_predictions
    except ImportError:
        load_validation_dataset, evaluate_matching_predictions = None, None


def run_logistic_regression(features_path: str = "experiments/phase2/val_50k_features.parquet"):
    print("=" * 70, flush=True)
    print("EXP-02: L2-REGULARIZED LOGISTIC REGRESSION ON 24 PAIRWISE FEATURES", flush=True)
    print("=" * 70, flush=True)

    t0 = time.time()
    # 1. Load ground truth and feature matrix
    val_path = "experiments/val_sample_50k.parquet"
    print(f"Loading ground truth from {val_path}...", flush=True)
    val_df, gt_dict = load_validation_dataset(val_path)

    print(f"Loading candidate features from {features_path}...", flush=True)
    feats_df = pl.read_parquet(features_path)
    print(f"Loaded {len(feats_df):,} candidate pairs.", flush=True)

    # 2. Entity-Grouped Train / Dev Split (70% train queries, 30% test holdout)
    all_queries = val_df["entity_id"].unique().to_list()
    np.random.seed(2026)
    np.random.shuffle(all_queries)
    n_train_q = int(len(all_queries) * 0.70)
    train_queries = set(all_queries[:n_train_q])
    holdout_queries = set(all_queries[n_train_q:])

    print(f"Query Split: {len(train_queries):,} train queries, {len(holdout_queries):,} evaluation queries.")

    # 3. Separate train and test feature matrices
    feature_cols = [
        col for col in feats_df.columns
        if col not in ["entity_id", "target_id", "max_prio", "n_channels", "label"]
    ]
    print(f"Using {len(feature_cols)} features: {feature_cols}")

    # Convert to pandas/numpy for scikit-learn
    train_df = feats_df.filter(pl.col("entity_id").is_in(list(train_queries)))
    test_df = feats_df.filter(pl.col("entity_id").is_in(list(holdout_queries)))

    X_train = train_df.select(feature_cols).to_numpy()
    y_train = train_df["label"].to_numpy()

    X_test = test_df.select(feature_cols).to_numpy()
    y_test = test_df["label"].to_numpy()

    print(f"X_train: {X_train.shape} (Positives: {y_train.sum():,})")
    print(f"X_test:  {X_test.shape} (Positives: {y_test.sum():,})")

    # 4. Standard Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. Fit Logistic Regression
    print("\nFitting Logistic Regression (C=1.0, max_iter=500)...", flush=True)
    lr = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs", random_state=2026)
    lr.fit(X_train_scaled, y_train)

    # 6. Model Diagnostics
    train_probs = lr.predict_proba(X_train_scaled)[:, 1]
    test_probs = lr.predict_proba(X_test_scaled)[:, 1]

    roc_auc = roc_auc_score(y_test, test_probs)
    pr_auc = average_precision_score(y_test, test_probs)
    print(f"Holdout Diagnostics: ROC-AUC = {roc_auc:.4f}, PR-AUC = {pr_auc:.4f}")

    # Learned Feature Coefficients
    print("\nTop Learned Feature Coefficients (Weights):")
    coef_pairs = sorted(zip(feature_cols, lr.coef_[0]), key=lambda x: abs(x[1]), reverse=True)
    for name, w in coef_pairs:
        direction = "(+ Evidence)" if w > 0 else "(- Contradiction)"
        print(f"  {name:<26}: {w:+.4f}  {direction}")

    # 7. Threshold Search for Macro F0.5 on Holdout Queries
    print("\n--- THRESHOLD GRID SEARCH FOR MACRO F0.5 ON EVALUATION QUERIES ---", flush=True)
    holdout_gt = {sid: gt_dict[sid] for sid in holdout_queries if sid in gt_dict}
    test_q_arr = test_df["entity_id"].to_numpy()
    test_t_arr = test_df["target_id"].to_numpy()

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    best_f05 = -1.0
    best_theta = None
    best_preds = None
    best_eval = None
    grid_records = []

    for theta in thresholds:
        mask = test_probs >= theta
        filt_q = test_q_arr[mask]
        filt_t = test_t_arr[mask]

        pred_dict: Dict[str, Set[str]] = {sid: set() for sid in holdout_queries}
        for q, t in zip(filt_q, filt_t):
            pred_dict[q].add(t)

        ev = evaluate_matching_predictions(pred_dict, holdout_gt, s1_df=None, verbose=False)
        f05 = ev["macro_f05"]
        p = ev["mean_precision"]
        r = ev["mean_recall"]
        sing_acc = ev["singleton_accuracy"]
        tot_preds = ev["link_metrics"]["total_pred_links"]

        grid_records.append({
            "theta": theta,
            "macro_f05": f05,
            "precision": p,
            "recall": r,
            "singleton_acc": sing_acc,
            "pred_links": tot_preds
        })
        print(f"  theta = {theta:.2f} | Macro F0.5 = {f05:.4f} | Prec = {p:.4f} | Rec = {r:.4f} | SingAcc = {sing_acc:.4f} | Pred Links = {tot_preds:,}", flush=True)

        if f05 > best_f05:
            best_f05 = f05
            best_theta = theta
            best_preds = pred_dict
            best_eval = ev

    print(f"\nOptimal Decision Threshold: theta* = {best_theta:.2f} with Macro F0.5 = {best_f05:.5f}")

    # Full slice evaluation on holdout
    holdout_s1_df = val_df.filter(pl.col("entity_id").is_in(list(holdout_queries)))
    final_eval = evaluate_matching_predictions(best_preds, holdout_gt, s1_df=holdout_s1_df, verbose=True)

    # Save artifact
    result_artifact = {
        "experiment_id": "EXP-02-FEAT-LOGREG",
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "coefficients": {k: round(float(v), 4) for k, v in coef_pairs},
        "optimal_theta": best_theta,
        "grid_search": grid_records,
        "final_evaluation": final_eval,
        "runtime_seconds": round(time.time() - t0, 2)
    }

    out_file = "experiments/phase2/exp02_logreg_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result_artifact, f, indent=2)

    print(f"Results saved to {out_file} in {result_artifact['runtime_seconds']}s")
    return result_artifact

if __name__ == "__main__":
    run_logistic_regression()
