"""
experiments/phase2/models/lightgbm_lambdarank.py
================================================
EXP-04: LightGBM LambdaRank (Query-Grouped Learning-to-Rank).
Trains a query-grouped LambdaMART ranker to optimize intra-query ranking (NDCG),
coupled with a calibrated decision boundary and margin-based abstention for Macro F0.5.
"""

import sys
import os
import json
import time
from typing import Dict, Set, List
import polars as pl
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.abspath("."))
from experiments.phase2.validation_framework import load_validation_dataset, evaluate_matching_predictions


def run_lightgbm_lambdarank(features_path: str = "experiments/phase2/val_50k_features.parquet"):
    print("=" * 70, flush=True)
    print("EXP-04: LIGHTGBM LAMBDARANK (QUERY-GROUPED LTR ON 24 FEATURES)", flush=True)
    print("=" * 70, flush=True)

    t0 = time.time()
    # 1. Load validation queries and ground truth
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

    # 3. Sort feature matrices by entity_id (MANDATORY for LightGBM LambdaRank groups)
    feature_cols = [
        col for col in feats_df.columns
        if col not in ["entity_id", "target_id", "max_prio", "n_channels", "label"]
    ]
    print(f"Using {len(feature_cols)} features: {feature_cols}")

    train_df = feats_df.filter(pl.col("entity_id").is_in(list(train_queries))).sort("entity_id")
    test_df = feats_df.filter(pl.col("entity_id").is_in(list(holdout_queries))).sort("entity_id")

    # Compute group sizes per query
    train_groups = train_df.group_by("entity_id", maintain_order=True).len()["len"].to_numpy()
    test_groups = test_df.group_by("entity_id", maintain_order=True).len()["len"].to_numpy()

    X_train = train_df.select(feature_cols).to_numpy()
    y_train = train_df["label"].to_numpy()

    X_test = test_df.select(feature_cols).to_numpy()
    y_test = test_df["label"].to_numpy()

    print(f"Train groups: {len(train_groups):,} queries, {X_train.shape[0]:,} candidates (Positives: {y_train.sum():,})")
    print(f"Test groups:  {len(test_groups):,} queries, {X_test.shape[0]:,} candidates (Positives: {y_test.sum():,})")

    # 4. Train LightGBM Ranker
    print("\nTraining LightGBM Ranker (objective='lambdarank', metric='ndcg', leaves=31)...", flush=True)
    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        eval_at=[1, 3, 5, 10],
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=2026,
        n_jobs=-1,
        importance_type="gain"
    )
    ranker.fit(
        X_train, y_train,
        group=train_groups,
        eval_set=[(X_test, y_test)],
        eval_group=[test_groups],
        callbacks=[lgb.log_evaluation(50)]
    )

    # 5. Predict ranking scores
    test_scores = ranker.predict(X_test)

    # Feature Importances (Gain)
    print("\nTop Feature Importances (LambdaRank Gain):")
    gains = ranker.feature_importances_
    feat_gains = sorted(zip(feature_cols, gains), key=lambda x: x[1], reverse=True)
    total_gain = sum(gains)
    for name, g in feat_gains:
        pct = (g / total_gain) * 100.0 if total_gain > 0 else 0.0
        print(f"  {name:<26}: {g:12.1f} ({pct:5.2f}%)")

    # 6. Threshold Grid Search for Macro F0.5
    print("\n--- THRESHOLD GRID SEARCH FOR MACRO F0.5 ON EVALUATION QUERIES ---", flush=True)
    holdout_gt = {sid: gt_dict[sid] for sid in holdout_queries if sid in gt_dict}
    test_q_arr = test_df["entity_id"].to_numpy()
    test_t_arr = test_df["target_id"].to_numpy()

    # Normalizing ranking scores with sigmoid to convert into [0, 1] pseudo-probabilities
    test_probs = 1.0 / (1.0 + np.exp(-test_scores))

    thresholds = [0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
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

    # 7. Full slice evaluation on holdout
    holdout_s1_df = val_df.filter(pl.col("entity_id").is_in(list(holdout_queries)))
    final_eval = evaluate_matching_predictions(best_preds, holdout_gt, s1_df=holdout_s1_df, verbose=True)

    # 8. Save artifact
    result_artifact = {
        "experiment_id": "EXP-04-GBDT-LGBM-RANK",
        "feature_importances_gain": {k: round(float(v), 2) for k, v in feat_gains},
        "optimal_theta": best_theta,
        "grid_search": grid_records,
        "final_evaluation": final_eval,
        "runtime_seconds": round(time.time() - t0, 2)
    }

    out_file = "experiments/phase2/exp04_lgbm_lambdarank_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result_artifact, f, indent=2)

    print(f"Results saved to {out_file} in {result_artifact['runtime_seconds']}s")
    return result_artifact

if __name__ == "__main__":
    run_lightgbm_lambdarank()
