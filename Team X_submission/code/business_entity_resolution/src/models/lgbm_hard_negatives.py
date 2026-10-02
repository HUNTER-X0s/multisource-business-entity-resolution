"""
experiments/phase2/models/lgbm_hard_negatives.py
=================================================
EXP-09: LightGBM Pointwise + Hard Negative Mining.

Hypothesis: The model over-predicts matches for entity pairs that share
the same business name (same-name collisions). Explicitly including
hard negatives from same-name groups during training will sharpen
the decision boundary and improve precision without hurting recall.

Strategy:
  1. Re-use EXP-03 model configuration (24 features, n_est=300)
  2. During training, sample ADDITIONAL hard negatives:
     - Pairs with name_jaro_winkler > 0.90 but different address (non-match)
     - Pairs with exact_clean_name match but addr_token_jaccard < 0.2 (non-match)
  3. Up-sample hard negatives at 2x normal negative sampling rate
  4. Evaluate against the same 15,000 holdout queries
"""

import sys
import os
import json
import time
import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.abspath("."))
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


def run_hard_negatives(features_path="experiments/phase2/val_50k_features.parquet"):
    import pandas as pd

    print("=" * 70, flush=True)
    print("EXP-09: LIGHTGBM + HARD NEGATIVE MINING", flush=True)
    print("=" * 70, flush=True)
    t_start = time.time()

    val_path = "experiments/val_sample_50k.parquet"
    s1_df, gt_dict = load_validation_dataset(val_path)

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

    qid_col = "query_id" if "query_id" in feats_pd.columns else "entity_id"
    cid_col = "target_id" if "target_id" in feats_pd.columns else "candidate_id"
    feature_cols = [c for c in FEATURE_COLS if c in feats_pd.columns]

    train_mask = feats_pd[qid_col].isin(train_queries)
    test_mask  = feats_pd[qid_col].isin(holdout_queries)
    train_df   = feats_pd[train_mask].copy()
    test_df    = feats_pd[test_mask].copy()

    print("Labeling pairs...", flush=True)
    def label_row(row):
        return int(row[cid_col] in gt_dict.get(row[qid_col], set()))
    train_df["label"] = train_df.apply(label_row, axis=1)
    test_df["label"]  = test_df.apply(label_row, axis=1)

    y_train = train_df["label"].values
    y_test  = test_df["label"].values

    # ── Identify hard negatives: high name similarity but address contradiction
    hard_neg_mask = (
        (train_df["name_jaro_winkler"] > 0.90) &
        (train_df["addr_token_jaccard"].fillna(0) < 0.20) &
        (train_df["addr_is_null_target"].fillna(0) == 0) &
        (y_train == 0)
    )
    hard_neg_count = hard_neg_mask.sum()
    print(f"\nHard negatives identified: {hard_neg_count:,} / {(y_train == 0).sum():,} total negatives", flush=True)
    print(f"  Hard neg rate: {100*hard_neg_count/(y_train==0).sum():.1f}% of training negatives", flush=True)

    # ── Build sample weights: hard negatives get 2x weight
    sample_weight = np.ones(len(train_df), dtype=np.float32)
    sample_weight[hard_neg_mask & (y_train == 0)] = 2.0
    sample_weight[y_train == 1] = (y_train == 0).sum() / max((y_train == 1).sum(), 1)  # pos weight

    print(f"\nSample weight distribution:", flush=True)
    print(f"  Positives weight: {sample_weight[y_train==1][0]:.1f}", flush=True)
    print(f"  Hard negatives weight: 2.0 ({hard_neg_count:,} samples)", flush=True)
    print(f"  Normal negatives weight: 1.0", flush=True)

    X_train = train_df[feature_cols].values.astype(np.float32)
    X_test  = test_df[feature_cols].values.astype(np.float32)

    print(f"\nTraining LightGBM with hard-negative sample weighting (n_est=300, leaves=31, lr=0.05)...", flush=True)
    model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        objective="binary", random_state=SEED, n_jobs=-1, verbose=-1,
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)

    proba = model.predict_proba(X_test)[:, 1]
    roc   = roc_auc_score(y_test, proba)
    pr    = average_precision_score(y_test, proba)
    print(f"Holdout Diagnostics: ROC-AUC = {roc:.4f}, PR-AUC = {pr:.4f}", flush=True)

    holdout_gt = {sid: gt_dict[sid] for sid in holdout_queries if sid in gt_dict}
    holdout_s1 = s1_df.filter(pl.col("entity_id").is_in(list(holdout_queries)))

    test_df = test_df.copy()
    test_df["score"] = proba

    print("\n--- THRESHOLD GRID SEARCH ---")
    best_theta, best_f05 = 0.5, 0.0
    thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    grid_results = []

    for theta in thresholds:
        preds = {}
        for qid, grp in test_df[test_df["score"] >= theta].groupby(qid_col):
            preds[qid] = set(grp[cid_col].tolist())
        for qid in holdout_queries:
            if qid not in preds:
                preds[qid] = set()
        ev   = evaluate_matching_predictions(preds, holdout_gt, s1_df=holdout_s1, verbose=False)
        f05  = ev["macro_f05"]
        prec = ev["mean_precision"]
        rec  = ev["mean_recall"]
        sing = ev.get("singleton_accuracy", 0.0)
        npred = sum(len(v) for v in preds.values())
        print(f"  theta = {theta:.2f} | F0.5 = {f05:.4f} | Prec = {prec:.4f} | Rec = {rec:.4f} | SingAcc = {sing:.4f} | Links = {npred:,}")
        grid_results.append({"theta": theta, "macro_f05": f05, "mean_precision": prec,
                              "mean_recall": rec, "singleton_accuracy": sing, "n_pred_links": npred})
        if f05 > best_f05:
            best_f05, best_theta = f05, theta

    print(f"\nOptimal theta* = {best_theta:.2f} | Macro F0.5 = {best_f05:.5f}", flush=True)
    print(f"EXP-03 champion: 0.77856 | Delta: {best_f05-0.77856:+.5f}", flush=True)

    best_preds = {}
    for qid, grp in test_df[test_df["score"] >= best_theta].groupby(qid_col):
        best_preds[qid] = set(grp[cid_col].tolist())
    for qid in holdout_queries:
        if qid not in best_preds:
            best_preds[qid] = set()

    final_eval = evaluate_matching_predictions(best_preds, holdout_gt, s1_df=holdout_s1, verbose=True)

    elapsed = time.time() - t_start
    output = {
        "experiment": "EXP-09",
        "model": "LightGBM Hard Negative Mining",
        "n_hard_negatives": int(hard_neg_count),
        "hard_neg_fraction": float(hard_neg_count / max((y_train == 0).sum(), 1)),
        "n_train_queries": len(train_queries),
        "n_eval_queries":  len(holdout_queries),
        "holdout_roc_auc": float(roc),
        "holdout_pr_auc":  float(pr),
        "optimal_threshold": best_theta,
        "macro_f05": best_f05,
        "exp03_champion_f05": 0.77856,
        "delta_vs_champion": best_f05 - 0.77856,
        "threshold_grid": grid_results,
        "final_metrics": {k: float(v) if isinstance(v, (float, int, np.floating)) else str(v)
                          for k, v in final_eval.items() if not isinstance(v, dict)},
        "elapsed_seconds": elapsed,
    }
    out_path = "experiments/phase2/exp09_hard_negatives_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path} in {elapsed:.2f}s")


if __name__ == "__main__":
    run_hard_negatives()
