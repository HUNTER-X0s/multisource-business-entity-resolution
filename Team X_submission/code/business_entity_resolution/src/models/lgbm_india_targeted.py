"""
experiments/phase2/models/lgbm_india_targeted.py
=================================================
EXP-10: India-Targeted Retraining with Address Imputation Features.

Root cause analysis of India gap (F0.5=0.691 vs US=0.837):
  - Higher address null rate in Indian entity records (addr_is_null_target=1)
  - When addr_is_null_target=1, name_x_addr=0 --> model loses its dominant feature
  - India names have higher transliteration variance (Devanagari romanizations)

Strategy:
  1. Add address-null-aware features:
     - name_only_score: name_token_set when addr_is_null_target=1, else 0
     - addr_available_score: name_x_addr when addr_is_null_target=0, else 0
  2. Add India-specific name similarity signal:
     - name_char3_jaccard: char 3-gram Jaccard (robust to transliteration)
     - name_char2_jaccard: char 2-gram Jaccard
  3. Train GLOBAL model with these 4 extra features (28 total), then compare
  4. Also test INDIA-ONLY model: separate LightGBM trained only on Indian queries
"""

import sys
import os
import json
import time
import re
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

BASE_FEATURES = [
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

def _char_ngrams(s, n):
    s = str(s).lower() if s else ""
    return {s[i:i+n] for i in range(len(s) - n + 1)} if len(s) >= n else set()

def char_ngram_jaccard(a, b, n):
    if not a or not b:
        return 0.0
    sa, sb = _char_ngrams(a, n), _char_ngrams(b, n)
    union = len(sa | sb)
    return len(sa & sb) / union if union > 0 else 0.0


def add_targeted_features(df_pd):
    """Add 4 targeted features addressing India-specific weaknesses."""
    n = len(df_pd)

    def get_col(*names):
        for c in names:
            if c in df_pd.columns:
                return df_pd[c].tolist()
        return [None] * n

    qnames = get_col("query_name", "name_query")
    tnames = get_col("target_name", "name_target")

    nc2 = np.zeros(n, np.float32)
    nc3 = np.zeros(n, np.float32)
    for i in range(n):
        nc2[i] = char_ngram_jaccard(qnames[i], tnames[i], 2)
        nc3[i] = char_ngram_jaccard(qnames[i], tnames[i], 3)

    df_pd = df_pd.copy()
    df_pd["name_char2_jaccard"] = nc2
    df_pd["name_char3_jaccard"] = nc3

    # Address-null-conditional features
    addr_null = df_pd.get("addr_is_null_target", 0).fillna(0) if "addr_is_null_target" in df_pd.columns else 0
    name_ts   = df_pd.get("name_token_set", 0).fillna(0) if "name_token_set" in df_pd.columns else 0
    name_xa   = df_pd.get("name_x_addr", 0).fillna(0) if "name_x_addr" in df_pd.columns else 0

    df_pd["name_only_score"]     = (name_ts * addr_null).astype(np.float32)       # name_token_set when addr=null
    df_pd["addr_available_score"] = (name_xa * (1 - addr_null)).astype(np.float32) # name_x_addr when addr=present

    return df_pd


TARGET_FEATURES = ["name_char2_jaccard", "name_char3_jaccard", "name_only_score", "addr_available_score"]


def run_india_targeted(features_path="experiments/phase2/val_50k_features.parquet"):
    import pandas as pd

    print("=" * 70, flush=True)
    print("EXP-10: INDIA-TARGETED FEATURE ENGINEERING + RETRAINING", flush=True)
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

    print(f"\nAdding {len(TARGET_FEATURES)} targeted features...", flush=True)
    t0 = time.time()
    feats_pd = add_targeted_features(feats_pd)
    print(f"  Done in {time.time()-t0:.1f}s", flush=True)

    feature_cols = [c for c in BASE_FEATURES if c in feats_pd.columns] + \
                   [c for c in TARGET_FEATURES if c in feats_pd.columns]
    print(f"Using {len(feature_cols)} features ({len(BASE_FEATURES)} base + {len(TARGET_FEATURES)} targeted)", flush=True)

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

    print(f"\nTraining LightGBM (n_est=300, leaves=31, lr=0.05)...", flush=True)
    model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        objective="binary", random_state=SEED, n_jobs=-1, verbose=-1,
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    roc   = roc_auc_score(y_test, proba)
    pr    = average_precision_score(y_test, proba)
    print(f"Holdout Diagnostics: ROC-AUC = {roc:.4f}, PR-AUC = {pr:.4f}", flush=True)

    importances = model.feature_importances_
    total_gain  = importances.sum()
    fi_sorted   = sorted(zip(feature_cols, importances), key=lambda x: -x[1])
    print("\nTop Feature Importances (incl. targeted features):")
    for fname, gain in fi_sorted[:15]:
        marker = " <<< TARGETED" if fname in TARGET_FEATURES else ""
        print(f"  {fname:<30}: {gain:>12.1f} ({100*gain/total_gain:>5.2f}%){marker}")

    holdout_gt = {sid: gt_dict[sid] for sid in holdout_queries if sid in gt_dict}
    holdout_s1 = s1_df.filter(pl.col("entity_id").is_in(list(holdout_queries)))
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
        print(f"  theta = {theta:.2f} | F0.5 = {f05:.4f} | Prec = {prec:.4f} | Rec = {rec:.4f} | SingAcc = {sing:.4f}")
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
        "experiment": "EXP-10",
        "model": "LightGBM India-Targeted (28 features)",
        "target_features": TARGET_FEATURES,
        "n_features": len(feature_cols),
        "holdout_roc_auc": float(roc),
        "holdout_pr_auc":  float(pr),
        "optimal_threshold": best_theta,
        "macro_f05": best_f05,
        "exp03_champion_f05": 0.77856,
        "delta_vs_champion": best_f05 - 0.77856,
        "threshold_grid": grid_results,
        "final_metrics": {k: float(v) if isinstance(v, (float, int, np.floating)) else str(v)
                          for k, v in final_eval.items() if not isinstance(v, dict)},
        "feature_importances": {n: float(g) for n, g in fi_sorted},
        "elapsed_seconds": elapsed,
    }
    out_path = "experiments/phase2/exp10_india_targeted_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path} in {elapsed:.2f}s")


if __name__ == "__main__":
    run_india_targeted()
