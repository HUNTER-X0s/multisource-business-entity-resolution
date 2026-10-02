"""
experiments/phase2/models/lgbm_expanded.py
==========================================
EXP-06: LightGBM Pointwise - Expanded Feature Set (40 features).
Extends EXP-03 champion (Macro F0.5 = 0.7786) with 16 additional signals.
Target slices: India (F0.5=0.691), match_bin_1 (F0.5=0.604)
"""

import sys
import os
import json
import time
import re
from typing import Dict, Set, List, Optional
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

def _tokens(s):
    if not s or (isinstance(s, float) and np.isnan(s)):
        return []
    return re.split(r"\s+", str(s).strip().lower())

def _char_ngrams(s, n):
    s = str(s).lower() if s else ""
    return {s[i:i+n] for i in range(len(s) - n + 1)} if len(s) >= n else set()

def char_ngram_jaccard(a, b, n):
    if not a or not b:
        return 0.0
    sa, sb = _char_ngrams(a, n), _char_ngrams(b, n)
    union = len(sa | sb)
    return len(sa & sb) / union if union > 0 else 0.0

def prefix_match(a, b, k):
    if not a or not b:
        return 0
    return int(str(a).lower()[:k] == str(b).lower()[:k])

def abbrev_ratio(s):
    toks = _tokens(s)
    return sum(1 for t in toks if len(t) == 1) / len(toks) if toks else 0.0

def word_count_diff(a, b):
    return abs(len(_tokens(a)) - len(_tokens(b)))

def word_count_ratio(a, b):
    ca, cb = len(_tokens(a)), len(_tokens(b))
    if ca == 0 and cb == 0:
        return 1.0
    return min(ca, cb) / max(ca, cb) if max(ca, cb) > 0 else 0.0

def city_token_overlap(a, b):
    def city_toks(s):
        return {t for t in _tokens(s) if len(t) > 3 and not t.isdigit()}
    sa, sb = city_toks(a), city_toks(b)
    if not sa and not sb:
        return 1.0
    union = len(sa | sb)
    return len(sa & sb) / union if union > 0 else 0.0

def pincode_prefix_match(a, b, k=3):
    if not a or not b:
        return 0
    sa = re.sub(r"\s+", "", str(a))
    sb = re.sub(r"\s+", "", str(b))
    return int(sa[:k] == sb[:k]) if len(sa) >= k and len(sb) >= k else 0

NEW_FEATURES = [
    "name_char2_jaccard", "name_char3_jaccard",
    "name_prefix3_match", "name_prefix5_match",
    "name_abbrev_ratio_q", "name_abbrev_ratio_t",
    "name_word_count_diff", "name_word_count_ratio",
    "addr_char3_jaccard", "addr_word_count_diff",
    "city_token_overlap", "pincode_prefix_match",
    "candidate_rank", "n_candidates",
    "name_x_addr_x_postal", "india_flag",
]

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

def compute_new_features(df_pd):
    n = len(df_pd)
    def get_col(*names):
        for c in names:
            if c in df_pd.columns:
                return df_pd[c].tolist()
        return [None] * n

    qnames = get_col("query_name", "name_query")
    tnames = get_col("target_name", "name_target")
    qaddrs = get_col("query_address", "std_address_query", "addr_query")
    taddrs = get_col("target_address", "std_address_target", "addr_target")
    qposts = get_col("query_postal", "postal_query")
    tposts = get_col("target_postal", "postal_target")

    arrs = {k: np.zeros(n, np.float32) for k in [
        "nc2","nc3","np3","np5","nar_q","nar_t","nwcd","nwcr","ac3","awcd","cto","ppm"
    ]}
    for i in range(n):
        qn, tn = qnames[i], tnames[i]
        qa, ta = qaddrs[i], taddrs[i]
        qp, tp = qposts[i], tposts[i]
        arrs["nc2"][i]  = char_ngram_jaccard(qn, tn, 2)
        arrs["nc3"][i]  = char_ngram_jaccard(qn, tn, 3)
        arrs["np3"][i]  = prefix_match(qn, tn, 3)
        arrs["np5"][i]  = prefix_match(qn, tn, 5)
        arrs["nar_q"][i]= abbrev_ratio(qn)
        arrs["nar_t"][i]= abbrev_ratio(tn)
        arrs["nwcd"][i] = word_count_diff(qn, tn)
        arrs["nwcr"][i] = word_count_ratio(qn, tn)
        arrs["ac3"][i]  = char_ngram_jaccard(qa, ta, 3)
        arrs["awcd"][i] = word_count_diff(qa, ta)
        arrs["cto"][i]  = city_token_overlap(qa, ta)
        arrs["ppm"][i]  = pincode_prefix_match(qp, tp, 3)

    df_pd = df_pd.copy()
    df_pd["name_char2_jaccard"]    = arrs["nc2"]
    df_pd["name_char3_jaccard"]    = arrs["nc3"]
    df_pd["name_prefix3_match"]    = arrs["np3"]
    df_pd["name_prefix5_match"]    = arrs["np5"]
    df_pd["name_abbrev_ratio_q"]   = arrs["nar_q"]
    df_pd["name_abbrev_ratio_t"]   = arrs["nar_t"]
    df_pd["name_word_count_diff"]  = arrs["nwcd"]
    df_pd["name_word_count_ratio"] = arrs["nwcr"]
    df_pd["addr_char3_jaccard"]    = arrs["ac3"]
    df_pd["addr_word_count_diff"]  = arrs["awcd"]
    df_pd["city_token_overlap"]    = arrs["cto"]
    df_pd["pincode_prefix_match"]  = arrs["ppm"]

    qid_col = "query_id" if "query_id" in df_pd.columns else "entity_id"
    if "name_x_addr" in df_pd.columns:
        df_pd["candidate_rank"] = (
            df_pd.groupby(qid_col)["name_x_addr"]
            .rank(ascending=False, method="first")
            .astype(np.float32)
        )
        df_pd["n_candidates"] = df_pd.groupby(qid_col)[qid_col].transform("count").astype(np.float32)
    else:
        df_pd["candidate_rank"] = 0.0
        df_pd["n_candidates"]   = 0.0

    pm = df_pd.get("postal_code_match", 0).fillna(0) if "postal_code_match" in df_pd.columns else 0
    na = df_pd.get("name_x_addr", 0).fillna(0) if "name_x_addr" in df_pd.columns else 0
    df_pd["name_x_addr_x_postal"] = (na * pm).astype(np.float32)

    india = np.zeros(n, np.float32)
    for col in ["query_country", "country_query", "country"]:
        if col in df_pd.columns:
            india = (df_pd[col].str.upper() == "IN").astype(np.float32).values
            break
    df_pd["india_flag"] = india
    return df_pd


def run_lgbm_expanded(features_path="experiments/phase2/val_50k_features.parquet"):
    import pandas as pd

    print("=" * 70, flush=True)
    print("EXP-06: LIGHTGBM POINTWISE - EXPANDED FEATURES (40 FEATURES)", flush=True)
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
    train_queries = set(all_queries[:n_train_q])
    holdout_queries = set(all_queries[n_train_q:])
    print(f"Query Split: {len(train_queries):,} train queries, {len(holdout_queries):,} evaluation queries.", flush=True)

    print(f"\nComputing {len(NEW_FEATURES)} new features on {len(feats_pd):,} rows...", flush=True)
    t0 = time.time()
    feats_pd = compute_new_features(feats_pd)
    print(f"  Done in {time.time()-t0:.1f}s", flush=True)

    qid_col = "query_id" if "query_id" in feats_pd.columns else "entity_id"
    cid_col = "target_id" if "target_id" in feats_pd.columns else "candidate_id"
    feature_cols = [c for c in BASE_FEATURES if c in feats_pd.columns] + \
                   [c for c in NEW_FEATURES if c in feats_pd.columns]
    print(f"Using {len(feature_cols)} features: {len([c for c in BASE_FEATURES if c in feats_pd.columns])} v1 + {len([c for c in NEW_FEATURES if c in feats_pd.columns])} new", flush=True)

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
    print(f"X_train: {X_train.shape} (Positives: {y_train.sum():,})", flush=True)
    print(f"X_test:  {X_test.shape}  (Positives: {y_test.sum():,})", flush=True)

    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    print(f"\nTraining LightGBM (n_est=500, leaves=63, lr=0.04, scale_pos_weight={pos_weight:.1f})...", flush=True)
    model = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.04, num_leaves=63,
        min_child_samples=20, feature_fraction=0.8,
        bagging_fraction=0.8, bagging_freq=5,
        reg_alpha=0.1, reg_lambda=0.1,
        scale_pos_weight=pos_weight,
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
    print("\nTop Feature Importances (Split Gain):")
    for fname, gain in fi_sorted[:20]:
        print(f"  {fname:<30}: {gain:>12.1f} ({100*gain/total_gain:>5.2f}%)")

    holdout_gt  = {sid: gt_dict[sid] for sid in holdout_queries if sid in gt_dict}
    holdout_s1  = s1_df.filter(pl.col("entity_id").is_in(list(holdout_queries)))
    test_df["score"] = proba

    print("\n--- THRESHOLD GRID SEARCH FOR MACRO F0.5 ON EVALUATION QUERIES ---")
    best_theta, best_f05 = 0.5, 0.0
    thresholds = [0.10,0.20,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85]
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
        print(f"  theta = {theta:.2f} | Macro F0.5 = {f05:.4f} | Prec = {prec:.4f} | Rec = {rec:.4f} | SingAcc = {sing:.4f} | Pred Links = {npred:,}")
        grid_results.append({"theta": theta, "macro_f05": f05, "mean_precision": prec,
                              "mean_recall": rec, "singleton_accuracy": sing, "n_pred_links": npred})
        if f05 > best_f05:
            best_f05, best_theta = f05, theta

    print(f"\nOptimal Decision Threshold: theta* = {best_theta:.2f} with Macro F0.5 = {best_f05:.5f}", flush=True)

    best_preds = {}
    for qid, grp in test_df[test_df["score"] >= best_theta].groupby(qid_col):
        best_preds[qid] = set(grp[cid_col].tolist())
    for qid in holdout_queries:
        if qid not in best_preds:
            best_preds[qid] = set()

    final_eval = evaluate_matching_predictions(best_preds, holdout_gt, s1_df=holdout_s1, verbose=True)

    elapsed = time.time() - t_start
    output = {
        "experiment": "EXP-06",
        "model": "LightGBM Pointwise Expanded (40 features)",
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "new_features": NEW_FEATURES,
        "n_train_queries": len(train_queries),
        "n_eval_queries":  len(holdout_queries),
        "holdout_roc_auc": float(roc),
        "holdout_pr_auc":  float(pr),
        "optimal_threshold": best_theta,
        "threshold_grid": grid_results,
        "final_metrics": {k: float(v) if isinstance(v, (float, int, np.floating)) else str(v)
                          for k, v in final_eval.items() if not isinstance(v, dict)},
        "feature_importances": {n: float(g) for n, g in fi_sorted},
        "elapsed_seconds": elapsed,
    }
    out_path = "experiments/phase2/exp06_lgbm_expanded_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path} in {elapsed:.2f}s")


if __name__ == "__main__":
    run_lgbm_expanded()
