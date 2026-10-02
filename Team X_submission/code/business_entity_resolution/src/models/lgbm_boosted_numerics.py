"""
experiments/phase2/models/lgbm_boosted_numerics.py
====================================================
EXP-11: LightGBM with Boosted Numeric & Address Features.

Motivation from EXP-08 Ablation:
  - G7_addr_numeric is the most critical group: delta = -0.016 when removed
    (house_number_match, numeric_token_jaccard, contradiction_house_no)
  - G2_name_fuzzy is second most critical: delta = -0.012 when removed

Strategy: Augment EXP-03 24-feature set with 8 additional numeric/address
signals targeting these two critical groups without diluting core calibration:

New features (8):
  house_number_exact     : int(house_num_query == house_num_target)
  house_number_diff      : abs difference in house numbers (numeric)
  street_token_overlap   : Jaccard of tokens after removing numbers
  addr_digit_seq_match   : longest common digit sequence in addresses
  name_digit_overlap     : Jaccard of digit tokens in names
  name_initials_match    : first-char of each token matches
  name_bigram_jaccard    : character bigram Jaccard (name, n=2)
  addr_bigram_jaccard    : character bigram Jaccard (address, n=2)
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

BOOST_FEATURES = [
    "house_number_diff", "street_token_overlap",
    "name_digit_overlap", "name_initials_match",
    "name_bigram_jaccard", "addr_bigram_jaccard",
    "name_only_score", "addr_available_score",
]


def _tokens(s):
    if not s or (isinstance(s, float) and np.isnan(s)):
        return []
    return re.split(r"\s+", str(s).strip().lower())

def _digit_tokens(s):
    return [t for t in _tokens(s) if t.isdigit()]

def _nondigit_tokens(s):
    return [t for t in _tokens(s) if not t.isdigit()]

def _char_ngrams(s, n):
    s = str(s).lower() if s else ""
    return {s[i:i+n] for i in range(len(s) - n + 1)} if len(s) >= n else set()

def char_ngram_jaccard(a, b, n):
    if not a or not b:
        return 0.0
    sa, sb = _char_ngrams(a, n), _char_ngrams(b, n)
    union = len(sa | sb)
    return len(sa & sb) / union if union > 0 else 0.0

def extract_house_number(s):
    """Extract first numeric token from address string."""
    if not s or (isinstance(s, float) and np.isnan(s)):
        return None
    toks = _tokens(str(s))
    for t in toks:
        if t.isdigit():
            return int(t)
    return None

def token_jaccard(a_toks, b_toks):
    sa, sb = set(a_toks), set(b_toks)
    if not sa and not sb:
        return 1.0
    union = len(sa | sb)
    return len(sa & sb) / union if union > 0 else 0.0

def initials_match(a, b):
    """Check if first character of each token matches."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    ia = "".join(t[0] for t in ta if t)
    ib = "".join(t[0] for t in tb if t)
    if not ia or not ib:
        return 0.0
    return char_ngram_jaccard(ia, ib, 1) if len(ia) >= 1 and len(ib) >= 1 else 0.0


def add_boost_features(df_pd):
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

    h_diff = np.zeros(n, np.float32)
    st_ov  = np.zeros(n, np.float32)
    nd_ov  = np.zeros(n, np.float32)
    n_init = np.zeros(n, np.float32)
    n_bi   = np.zeros(n, np.float32)
    a_bi   = np.zeros(n, np.float32)

    for i in range(n):
        qn, tn = qnames[i], tnames[i]
        qa, ta = qaddrs[i], taddrs[i]

        # House number difference
        qh = extract_house_number(qa)
        th = extract_house_number(ta)
        if qh is not None and th is not None:
            h_diff[i] = min(abs(qh - th), 999) / 999.0  # normalized 0-1
        else:
            h_diff[i] = 0.5  # unknown

        # Street tokens (non-digit) Jaccard
        st_ov[i] = token_jaccard(_nondigit_tokens(qa), _nondigit_tokens(ta))

        # Name digit token overlap
        nd_ov[i] = token_jaccard(_digit_tokens(qn), _digit_tokens(tn))

        # Name initials match
        n_init[i] = initials_match(qn, tn)

        # Name bigram Jaccard
        n_bi[i] = char_ngram_jaccard(qn, tn, 2)

        # Address bigram Jaccard
        a_bi[i] = char_ngram_jaccard(qa, ta, 2)

    df_pd = df_pd.copy()
    df_pd["house_number_diff"]  = h_diff
    df_pd["street_token_overlap"] = st_ov
    df_pd["name_digit_overlap"] = nd_ov
    df_pd["name_initials_match"] = n_init
    df_pd["name_bigram_jaccard"] = n_bi
    df_pd["addr_bigram_jaccard"] = a_bi

    # Address-null-conditional features from EXP-10
    addr_null = df_pd.get("addr_is_null_target", 0).fillna(0) if "addr_is_null_target" in df_pd.columns else 0
    name_ts   = df_pd.get("name_token_set", 0).fillna(0) if "name_token_set" in df_pd.columns else 0
    name_xa   = df_pd.get("name_x_addr", 0).fillna(0) if "name_x_addr" in df_pd.columns else 0
    df_pd["name_only_score"]      = (name_ts * addr_null).astype(np.float32)
    df_pd["addr_available_score"] = (name_xa * (1 - addr_null)).astype(np.float32)

    return df_pd


def run_boosted_numerics(features_path="experiments/phase2/val_50k_features.parquet"):
    import pandas as pd

    print("=" * 70, flush=True)
    print("EXP-11: LIGHTGBM BOOSTED NUMERIC & ADDRESS FEATURES (32 FEATURES)", flush=True)
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

    print(f"\nComputing {len(BOOST_FEATURES)} boost features...", flush=True)
    t0 = time.time()
    feats_pd = add_boost_features(feats_pd)
    print(f"  Done in {time.time()-t0:.1f}s", flush=True)

    feature_cols = [c for c in BASE_FEATURES if c in feats_pd.columns] + \
                   [c for c in BOOST_FEATURES if c in feats_pd.columns]
    print(f"Using {len(feature_cols)} features ({len(BASE_FEATURES)} base + {len(BOOST_FEATURES)} boost)", flush=True)

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
    print("\nTop Feature Importances (Gain):")
    for fname, gain in fi_sorted[:18]:
        marker = " <<< BOOST" if fname in BOOST_FEATURES else ""
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
        "experiment": "EXP-11",
        "model": "LightGBM Boosted Numeric+Address (32 features)",
        "n_features": len(feature_cols),
        "boost_features": BOOST_FEATURES,
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
    out_path = "experiments/phase2/exp11_boosted_numerics_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path} in {elapsed:.2f}s")


if __name__ == "__main__":
    run_boosted_numerics()
