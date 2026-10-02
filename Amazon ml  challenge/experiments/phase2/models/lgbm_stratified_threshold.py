"""
experiments/phase2/models/lgbm_stratified_threshold.py
=======================================================
EXP-07: Country-Stratified Threshold Optimization on EXP-03 scores.

Hypothesis: A single global threshold is suboptimal because India (F0.5=0.691)
and US (F0.5=0.837) have different score distributions and matching difficulty.
A lower threshold for India recovers more true matches at acceptable FP cost.

Strategy:
  1. Re-load EXP-03 model (or re-train identical model)
  2. Split holdout queries by country (US, India)
  3. Grid-search theta_US and theta_IN independently
  4. Apply country-specific thresholds at inference
  5. Report global Macro F0.5 with stratified thresholds vs. single global theta
"""

import sys
import os
import json
import time
import re
from typing import Dict, Set
import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.abspath("."))
from src.evaluate import macro_f05_from_dicts
from experiments.phase2.validation_framework import load_validation_dataset, evaluate_matching_predictions

SEED = 2026
np.random.seed(SEED)


def run_stratified_threshold(features_path="experiments/phase2/val_50k_features.parquet"):
    import pandas as pd

    print("=" * 70, flush=True)
    print("EXP-07: COUNTRY-STRATIFIED THRESHOLD OPTIMIZATION (EXP-03 FEATURES)", flush=True)
    print("=" * 70, flush=True)
    t_start = time.time()

    val_path = "experiments/val_sample_50k.parquet"
    print(f"Loading ground truth from {val_path}...", flush=True)
    s1_df, gt_dict = load_validation_dataset(val_path)

    print(f"Loading candidate features from {features_path}...", flush=True)
    feats_pl = pl.read_parquet(features_path)
    print(f"Loaded {len(feats_pl):,} candidate pairs.", flush=True)

    if "query_country" not in feats_pl.columns and "country" in s1_df.columns:
        qmeta = s1_df.select(["entity_id", "country"]).unique("entity_id")
        qid_name = "query_id" if "query_id" in feats_pl.columns else "entity_id"
        feats_pl = feats_pl.join(
            qmeta.rename({"entity_id": qid_name, "country": "query_country"}),
            on=qid_name, how="left"
        )

    feats_pd = feats_pl.to_pandas()

    all_queries = s1_df["entity_id"].unique().to_list()
    np.random.seed(SEED)
    np.random.shuffle(all_queries)
    n_train_q = int(len(all_queries) * 0.70)
    train_queries   = set(all_queries[:n_train_q])
    holdout_queries = set(all_queries[n_train_q:])
    print(f"Query Split: {len(train_queries):,} train | {len(holdout_queries):,} eval", flush=True)

    qid_col = "query_id" if "query_id" in feats_pd.columns else "entity_id"
    cid_col = "target_id" if "target_id" in feats_pd.columns else "candidate_id"

    feature_cols = [
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
    feature_cols = [c for c in feature_cols if c in feats_pd.columns]
    print(f"Using {len(feature_cols)} EXP-03 features", flush=True)

    train_mask = feats_pd[qid_col].isin(train_queries)
    test_mask  = feats_pd[qid_col].isin(holdout_queries)
    train_df   = feats_pd[train_mask].copy()
    test_df    = feats_pd[test_mask].copy()

    print("Labeling pairs...", flush=True)
    def label_row(row):
        return int(row[cid_col] in gt_dict.get(row[qid_col], set()))
    train_df["label"] = train_df.apply(label_row, axis=1)
    test_df["label"]  = test_df.apply(label_row, axis=1)

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df["label"].values
    X_test  = test_df[feature_cols].values.astype(np.float32)
    y_test  = test_df["label"].values

    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    print(f"\nTraining EXP-03-identical LightGBM (n_est=300, leaves=31, lr=0.05)...", flush=True)
    model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        objective="binary", random_state=SEED, n_jobs=-1, verbose=-1,
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    roc   = roc_auc_score(y_test, proba)
    pr    = average_precision_score(y_test, proba)
    print(f"Holdout Diagnostics: ROC-AUC = {roc:.4f}, PR-AUC = {pr:.4f}", flush=True)

    test_df = test_df.copy()
    test_df["score"] = proba

    # Country labels for holdout queries
    qid_country = {}
    if "query_country" in feats_pd.columns:
        for _, row in feats_pd[[qid_col, "query_country"]].drop_duplicates(qid_col).iterrows():
            qid_country[row[qid_col]] = str(row["query_country"]).upper()
    else:
        # Try from s1_df
        for row in s1_df.select(["entity_id","country"]).to_pandas().itertuples():
            qid_country[row.entity_id] = str(row.country).upper() if row.country else "US"

    holdout_gt = {sid: gt_dict[sid] for sid in holdout_queries if sid in gt_dict}
    holdout_s1 = s1_df.filter(pl.col("entity_id").is_in(list(holdout_queries)))

    # ── Baseline: global threshold (EXP-03 optimal was theta=0.55)
    print("\n--- BASELINE: GLOBAL THRESHOLD (EXP-03 optimal theta=0.55) ---")
    global_theta = 0.55
    preds_global = {}
    for qid, grp in test_df[test_df["score"] >= global_theta].groupby(qid_col):
        preds_global[qid] = set(grp[cid_col].tolist())
    for qid in holdout_queries:
        if qid not in preds_global:
            preds_global[qid] = set()
    ev_global = evaluate_matching_predictions(preds_global, holdout_gt, s1_df=holdout_s1, verbose=True)
    print(f"  Global Macro F0.5 @ theta=0.55: {ev_global['macro_f05']:.5f}")

    # Add country flag to test_df
    test_df["is_india"] = test_df[qid_col].map(lambda q: qid_country.get(q, "US").startswith("IN"))
    
    us_queries = set(q for q in holdout_queries if not qid_country.get(q, "US").startswith("IN"))
    in_queries = set(q for q in holdout_queries if qid_country.get(q, "US").startswith("IN"))
    print(f"Holdout query split: {len(us_queries):,} US queries, {len(in_queries):,} India queries", flush=True)

    us_gt = {q: holdout_gt[q] for q in us_queries if q in holdout_gt}
    in_gt = {q: holdout_gt[q] for q in in_queries if q in holdout_gt}

    # Pre-filter by country
    df_us = test_df[~test_df["is_india"]][[qid_col, cid_col, "score"]]
    df_in = test_df[test_df["is_india"]][[qid_col, cid_col, "score"]]

    # ── Stratified: grid search theta_US and theta_IN independently (they are mathematically decoupled)
    print("\n--- STRATIFIED THRESHOLD SEARCH (DECOUPLED) ---", flush=True)
    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

    us_grid = []
    best_theta_us, best_f05_us = 0.55, 0.0
    for theta in thresholds:
        preds = {}
        for qid, grp in df_us[df_us["score"] >= theta].groupby(qid_col):
            preds[qid] = set(grp[cid_col].tolist())
        for qid in us_queries:
            if qid not in preds:
                preds[qid] = set()
        res = macro_f05_from_dicts(gt_dict=us_gt, pred_dict=preds, s1_ids=list(us_queries))
        f05 = res["macro_f05"]
        us_grid.append({"theta": theta, "macro_f05": f05, "precision": res["mean_precision"], "recall": res["mean_recall"]})
        print(f"  US theta = {theta:.2f} | Macro F0.5 = {f05:.5f} | Prec = {res['mean_precision']:.4f} | Rec = {res['mean_recall']:.4f}", flush=True)
        if f05 > best_f05_us:
            best_f05_us, best_theta_us = f05, theta

    in_grid = []
    best_theta_in, best_f05_in = 0.55, 0.0
    for theta in thresholds:
        preds = {}
        for qid, grp in df_in[df_in["score"] >= theta].groupby(qid_col):
            preds[qid] = set(grp[cid_col].tolist())
        for qid in in_queries:
            if qid not in preds:
                preds[qid] = set()
        res = macro_f05_from_dicts(gt_dict=in_gt, pred_dict=preds, s1_ids=list(in_queries))
        f05 = res["macro_f05"]
        in_grid.append({"theta": theta, "macro_f05": f05, "precision": res["mean_precision"], "recall": res["mean_recall"]})
        print(f"  India theta = {theta:.2f} | Macro F0.5 = {f05:.5f} | Prec = {res['mean_precision']:.4f} | Rec = {res['mean_recall']:.4f}", flush=True)
        if f05 > best_f05_in:
            best_f05_in, best_theta_in = f05, theta

    # Combine best thresholds
    best_combo = {
        "theta_us": best_theta_us,
        "theta_in": best_theta_in,
        "us_f05": best_f05_us,
        "in_f05": best_f05_in,
    }

    best_preds = {}
    for qid, grp in df_us[df_us["score"] >= best_theta_us].groupby(qid_col):
        best_preds[qid] = set(grp[cid_col].tolist())
    for qid in us_queries:
        if qid not in best_preds:
            best_preds[qid] = set()

    for qid, grp in df_in[df_in["score"] >= best_theta_in].groupby(qid_col):
        best_preds[qid] = set(grp[cid_col].tolist())
    for qid in in_queries:
        if qid not in best_preds:
            best_preds[qid] = set()

    final_eval = evaluate_matching_predictions(best_preds, holdout_gt, s1_df=holdout_s1, verbose=True)
    best_combo["f05"] = final_eval["macro_f05"]

    print(f"\nBest combination: theta_US={best_combo['theta_us']:.2f}, theta_IN={best_combo['theta_in']:.2f}", flush=True)
    print(f"  Stratified Macro F0.5: {best_combo['f05']:.5f}", flush=True)
    print(f"  vs. Global theta=0.55: {ev_global['macro_f05']:.5f}", flush=True)
    delta = best_combo["f05"] - ev_global["macro_f05"]
    print(f"  Delta: {delta:+.5f}", flush=True)

    elapsed = time.time() - t_start
    output = {
        "experiment": "EXP-07",
        "model": "LightGBM Pointwise + Country-Stratified Thresholds",
        "baseline_global_theta": global_theta,
        "baseline_global_f05": float(ev_global["macro_f05"]),
        "best_theta_us": best_combo["theta_us"],
        "best_theta_in": best_combo["theta_in"],
        "stratified_f05": best_combo["f05"],
        "delta_vs_global": delta,
        "holdout_roc_auc": float(roc),
        "holdout_pr_auc": float(pr),
        "us_grid_results": us_grid,
        "in_grid_results": in_grid,
        "final_metrics": {k: float(v) if isinstance(v, (float, int, np.floating)) else str(v)
                          for k, v in final_eval.items() if not isinstance(v, dict)},
        "elapsed_seconds": elapsed,
    }
    out_path = "experiments/phase2/exp07_stratified_threshold_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path} in {elapsed:.2f}s")


if __name__ == "__main__":
    run_stratified_threshold()
