"""
experiments/phase2/models/lgbm_ablation.py
==========================================
EXP-08: Feature Ablation Study on EXP-03 Champion.

Systematically measures the marginal contribution of each feature group
by training LightGBM with one group removed at a time.

Feature groups:
  G1 - name_exact    : exact_clean_name, exact_stripped_name, exact_sorted_name
  G2 - name_fuzzy    : name_jaro_winkler, name_token_sort, name_token_set
  G3 - name_length   : name_len_diff, name_len_ratio
  G4 - addr_exact    : exact_clean_address
  G5 - addr_null     : addr_is_null_target
  G6 - addr_fuzzy    : addr_jaro_winkler, addr_token_jaccard, addr_token_overlap
  G7 - addr_numeric  : numeric_token_jaccard, house_number_match, contradiction_house_no
  G8 - postal        : postal_code_match, postal_both_present, contradiction_postal
  G9 - provenance    : is_source3, channel_max_priority, channel_count
  G10 - interactions  : name_x_addr, name_x_postal
"""

import sys
import os
import json
import time
import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

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

ALL_FEATURES = [
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

FEATURE_GROUPS = {
    "G1_name_exact":    ["exact_clean_name", "exact_stripped_name", "exact_sorted_name"],
    "G2_name_fuzzy":    ["name_jaro_winkler", "name_token_sort", "name_token_set"],
    "G3_name_length":   ["name_len_diff", "name_len_ratio"],
    "G4_addr_exact":    ["exact_clean_address"],
    "G5_addr_null":     ["addr_is_null_target"],
    "G6_addr_fuzzy":    ["addr_jaro_winkler", "addr_token_jaccard", "addr_token_overlap"],
    "G7_addr_numeric":  ["numeric_token_jaccard", "house_number_match", "contradiction_house_no"],
    "G8_postal":        ["postal_code_match", "postal_both_present", "contradiction_postal"],
    "G9_provenance":    ["is_source3", "channel_max_priority", "channel_count"],
    "G10_interactions": ["name_x_addr", "name_x_postal"],
}


def train_and_eval(X_train, y_train, X_test, y_test, test_df, gt_dict, holdout_queries,
                   s1_holdout, qid_col, cid_col, theta=0.55):
    """Train LightGBM and evaluate at fixed threshold theta."""
    model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        objective="binary", random_state=SEED, n_jobs=-1, verbose=-1,
    )
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    roc   = roc_auc_score(y_test, proba)

    td = test_df.copy()
    td["score"] = proba
    preds = {}
    for qid, grp in td[td["score"] >= theta].groupby(qid_col):
        preds[qid] = set(grp[cid_col].tolist())
    for qid in holdout_queries:
        if qid not in preds:
            preds[qid] = set()

    holdout_gt = {sid: gt_dict[sid] for sid in holdout_queries if sid in gt_dict}
    ev = evaluate_matching_predictions(preds, holdout_gt, s1_df=s1_holdout, verbose=False)
    return ev["macro_f05"], roc


def run_ablation(features_path="experiments/phase2/val_50k_features.parquet"):
    import pandas as pd

    print("=" * 70, flush=True)
    print("EXP-08: FEATURE GROUP ABLATION STUDY (EXP-03 BASE MODEL)", flush=True)
    print("=" * 70, flush=True)
    t_start = time.time()

    val_path = "experiments/val_sample_50k.parquet"
    s1_df, gt_dict = load_validation_dataset(val_path)

    feats_pd = pl.read_parquet(features_path).to_pandas()
    print(f"Loaded {len(feats_pd):,} candidate pairs.", flush=True)

    all_queries = s1_df["entity_id"].unique().to_list()
    np.random.seed(SEED)
    np.random.shuffle(all_queries)
    n_train_q = int(len(all_queries) * 0.70)
    train_queries   = set(all_queries[:n_train_q])
    holdout_queries = set(all_queries[n_train_q:])

    qid_col = "query_id" if "query_id" in feats_pd.columns else "entity_id"
    cid_col = "target_id" if "target_id" in feats_pd.columns else "candidate_id"
    feature_cols = [c for c in ALL_FEATURES if c in feats_pd.columns]

    train_mask = feats_pd[qid_col].isin(train_queries)
    test_mask  = feats_pd[qid_col].isin(holdout_queries)
    train_df   = feats_pd[train_mask].copy()
    test_df    = feats_pd[test_mask].copy()

    print("Labeling pairs...", flush=True)
    def label_row(row):
        return int(row[cid_col] in gt_dict.get(row[qid_col], set()))
    train_df["label"] = train_df.apply(label_row, axis=1)
    test_df["label"]  = test_df.apply(label_row, axis=1)

    holdout_s1 = s1_df.filter(pl.col("entity_id").is_in(list(holdout_queries)))
    y_train = train_df["label"].values
    y_test  = test_df["label"].values

    # ── Baseline: all features
    print(f"\n[FULL MODEL] Training with all {len(feature_cols)} features...", flush=True)
    X_train_full = train_df[feature_cols].values.astype(np.float32)
    X_test_full  = test_df[feature_cols].values.astype(np.float32)
    full_f05, full_roc = train_and_eval(
        X_train_full, y_train, X_test_full, y_test, test_df, gt_dict,
        holdout_queries, holdout_s1, qid_col, cid_col, theta=0.55
    )
    print(f"  [FULL] Macro F0.5 = {full_f05:.5f} | ROC-AUC = {full_roc:.4f}", flush=True)

    # ── Leave-one-group-out ablation
    ablation_results = {"FULL": {"macro_f05": full_f05, "roc_auc": full_roc, "features_used": len(feature_cols)}}
    print("\n--- LEAVE-ONE-GROUP-OUT ABLATION ---")
    print(f"  {'Group':<22} {'Features':<8} {'F0.5':<10} {'Delta':<10} {'ROC-AUC'}")
    print(f"  {'-'*65}")

    for group_name, group_feats in FEATURE_GROUPS.items():
        ablated_cols = [c for c in feature_cols if c not in group_feats]
        if not ablated_cols:
            continue
        X_tr = train_df[ablated_cols].values.astype(np.float32)
        X_te = test_df[ablated_cols].values.astype(np.float32)
        f05, roc_a = train_and_eval(
            X_tr, y_train, X_te, y_test, test_df, gt_dict,
            holdout_queries, holdout_s1, qid_col, cid_col, theta=0.55
        )
        delta = f05 - full_f05
        ablation_results[group_name] = {
            "macro_f05": f05, "delta": delta, "roc_auc": roc_a,
            "features_removed": group_feats, "features_used": len(ablated_cols)
        }
        marker = " <<< CRITICAL" if delta < -0.005 else (" <<< HELPFUL" if delta < -0.001 else "")
        print(f"  {group_name:<22} -{len(group_feats):<7} {f05:.5f}   {delta:+.5f}   {roc_a:.4f}{marker}")

    print(f"\n  {'FULL MODEL':<22} {len(feature_cols):<8} {full_f05:.5f}   {'baseline':>9}", flush=True)

    elapsed = time.time() - t_start
    output = {"experiment": "EXP-08", "model": "LightGBM Ablation",
               "full_model_f05": full_f05, "ablation_results": ablation_results,
               "elapsed_seconds": elapsed}
    out_path = "experiments/phase2/exp08_ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path} in {elapsed:.2f}s")


if __name__ == "__main__":
    run_ablation()
