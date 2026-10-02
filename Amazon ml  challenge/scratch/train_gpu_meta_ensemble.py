"""
scratch/train_gpu_meta_ensemble.py
==================================
Builds rich features for candidate pairs from 10-channel retrieval,
then trains a 5-fold cross-validated GPU meta-ensemble:
  1. LightGBM (CPU/GPU)
  2. XGBoost on GPU (device='cuda')
  3. CatBoost on GPU (task_type='GPU')
  4. Stacking meta-learner (calibrated blend)

Optimizes threshold and decision rules directly for macro F(0.5).
"""
import os
import sys
import time
import json
import pickle
from pathlib import Path
from collections import defaultdict
import numpy as np
import polars as pl
import pandas as pd
from rapidfuzz import fuzz as rfuzz
from rapidfuzz.distance import JaroWinkler
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

ROOT = Path("z:/Amazon ML")
EXP_DIR = ROOT / "experiments" / "phase2"
EXP_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = EXP_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CANDS_FILE = EXP_DIR / "phase2c_10ch_val_candidates.parquet"
VAL_NORM_FILE = ROOT / "experiments" / "cache" / "val_norm_v2.parquet"
TGT_NORM_FILE = ROOT / "experiments" / "cache" / "targets_norm_v2.parquet"
FEAT_FILE = EXP_DIR / "phase2c_10ch_val_features.parquet"

def flush(msg, end="\n"):
    print(msg, end=end, flush=True)

def char_ngram_jac(a, b, n):
    if not a or not b or len(a) < n or len(b) < n: return 0.0
    sa = {a[i:i+n] for i in range(len(a)-n+1)}
    sb = {b[i:i+n] for i in range(len(b)-n+1)}
    u = len(sa | sb)
    return len(sa & sb) / u if u > 0 else 0.0

def tok_jac(a, b):
    sa = set((a or '').split())
    sb = set((b or '').split())
    if not sa and not sb: return 1.0
    u = len(sa | sb)
    return len(sa & sb) / u if u > 0 else 0.0

def tok_overlap(a, b):
    sa = set((a or '').split())
    sb = set((b or '').split())
    if not sa or not sb: return 0.0
    return len(sa & sb) / min(len(sa), len(sb))

def compute_macro_f05(preds, queries, targets, gt_map, query_list, theta, cap_per_source=True):
    """
    Precision-optimized F0.5 evaluation with optional per-source cardinality capping.
    """
    pred_map = defaultdict(list)
    for p, q, t in zip(preds, queries, targets):
        pred_map[q].append((p, t))
    
    f05_list = []
    for qid in query_list:
        true_s = gt_map.get(qid, set())
        preds_q = pred_map.get(qid, [])
        
        # Sort candidates descending by predicted probability
        sorted_cands = sorted(preds_q, key=lambda x: x[0], reverse=True)
        
        matched = set()
        if cap_per_source:
            # At most 1 S2 and 1 S3 match per query
            s2_found = 0
            s3_found = 0
            for p, t in sorted_cands:
                if p >= theta:
                    is_s3 = t.startswith("S3-")
                    if is_s3 and s3_found < 1:
                        matched.add(t)
                        s3_found += 1
                    elif not is_s3 and s2_found < 1:
                        matched.add(t)
                        s2_found += 1
        else:
            matched = {t for (p, t) in sorted_cands if p >= theta}
        
        if not true_s:
            f05_list.append(1.0 if not matched else 0.0)
        elif not matched:
            f05_list.append(0.0)
        else:
            prec = len(matched & true_s) / len(matched)
            rec = len(matched & true_s) / len(true_s)
            if prec + rec == 0:
                f05_list.append(0.0)
            else:
                f05_list.append((1.25 * prec * rec) / (0.25 * prec + rec))
                
    return float(np.mean(f05_list))

def main():
    flush("=" * 70)
    flush("TRAIN GPU META-ENSEMBLE (LIGHTGBM + XGBOOST-GPU + CATBOOST-GPU)")
    flush("=" * 70)

    # 1. Load Ground Truth
    val_df = pl.read_parquet(ROOT / "experiments" / "val_sample_50k.parquet")
    val_gt = {}
    for row in val_df.iter_rows(named=True):
        qid = str(row["entity_id"])
        raw = str(row.get("matched_entity_ids", "") or "").strip()
        val_gt[qid] = {t.strip() for t in raw.split(",") if t.strip() and t.strip() not in ("nan", "None")}
    val_query_list = sorted(val_gt.keys())
    flush(f"Val queries: {len(val_query_list):,} | True links: {sum(len(v) for v in val_gt.values()):,}")

    # 2. Check if features already built
    if FEAT_FILE.exists():
        flush(f"Loading pre-computed features from {FEAT_FILE}...")
        feat_df = pl.read_parquet(FEAT_FILE).to_pandas()
    else:
        flush(f"Loading candidates from {CANDS_FILE}...")
        raw_cands = pl.read_parquet(CANDS_FILE)
        flush(f"Total candidate pairs in pool: {len(raw_cands):,}")

        # Retain 100% of true positive pairs, plus top-40 hard negatives per query
        flush("Filtering to 100% true positives + top-40 hard negatives per query...")
        gt_pairs = set((q, t) for q, targets in val_gt.items() for t in targets)
        
        # Add is_positive flag
        cands_with_pos = raw_cands.with_columns(
            pl.struct(["entity_id", "target_id"]).map_elements(
                lambda s: (s["entity_id"], s["target_id"]) in gt_pairs,
                return_dtype=pl.Boolean
            ).alias("is_pos")
        )
        
        pos_df = cands_with_pos.filter(pl.col("is_pos"))
        flush(f"Retained positive pairs: {len(pos_df):,}")
        
        neg_df = (
            cands_with_pos.filter(~pl.col("is_pos"))
            .sort(["entity_id", "max_prio", "n_channels"], descending=[False, True, True])
            .group_by("entity_id", maintain_order=True)
            .head(40)
        )
        flush(f"Selected hard negatives: {len(neg_df):,}")
        
        cands_df = pl.concat([pos_df, neg_df]).drop("is_pos").unique(["entity_id", "target_id"])
        flush(f"Final training candidate pairs: {len(cands_df):,}")

        flush("Loading normalized queries and targets for feature extraction...")
        q_norm = pl.read_parquet(VAL_NORM_FILE)
        t_norm = pl.read_parquet(TGT_NORM_FILE)

        # Build lookup dicts for queries and relevant targets
        q_dict = {str(r["entity_id"]): r for r in q_norm.select([
            "entity_id", "clean_name", "stripped_name", "sorted_name", "spaceless_name",
            "std_address", "sorted_addr", "house_no", "postal_code", "country", "first_word"
        ]).iter_rows(named=True)}

        target_ids_needed = set(cands_df["target_id"].to_list())
        flush(f"Filtering {len(target_ids_needed):,} target entities from target corpus...")
        t_sub = t_norm.filter(pl.col("entity_id").is_in(list(target_ids_needed)))
        t_dict = {str(r["entity_id"]): r for r in t_sub.select([
            "entity_id", "clean_name", "stripped_name", "sorted_name", "spaceless_name",
            "std_address", "sorted_addr", "house_no", "postal_code", "country", "first_word"
        ]).iter_rows(named=True)}

        flush(f"Extracting features for {len(cands_df):,} candidate pairs...")
        cand_rows = cands_df.select(["entity_id", "target_id", "max_prio", "n_channels"]).iter_rows(named=True)
        rows_list = list(cand_rows)

        feat_records = []
        t_feat = time.time()
        for i, row in enumerate(rows_list):
            if i % 250000 == 0 and i > 0:
                flush(f"    [{i:,}/{len(rows_list):,}] elapsed: {time.time()-t_feat:.1f}s")
            
            qid = str(row["entity_id"])
            tid = str(row["target_id"])
            q = q_dict.get(qid, {})
            t = t_dict.get(tid, {})
            
            qn = q.get("clean_name", "") or ""
            tn = t.get("clean_name", "") or ""
            qns = q.get("sorted_name", "") or ""
            tns = t.get("sorted_name", "") or ""
            qns2 = q.get("stripped_name", "") or ""
            tns2 = t.get("stripped_name", "") or ""
            qsp = q.get("spaceless_name", "") or ""
            tsp = t.get("spaceless_name", "") or ""
            qa = q.get("std_address", "") or ""
            ta = t.get("std_address", "") or ""
            qas = q.get("sorted_addr", "") or ""
            tas = t.get("sorted_addr", "") or ""
            qhno = q.get("house_no", "") or ""
            thno = t.get("house_no", "") or ""
            qpin = q.get("postal_code", "") or ""
            tpin = t.get("postal_code", "") or ""
            qfw = q.get("first_word", "") or ""
            tfw = t.get("first_word", "") or ""
            cty = (q.get("country", "") or "").lower()
            
            is_s3 = 1 if tid.startswith("S3-") else 0
            
            rec = {
                "entity_id": qid,
                "target_id": tid,
                "max_prio": float(row["max_prio"]),
                "n_channels": int(row["n_channels"]),
                
                # Exact match features
                "exact_clean_name": int(qn == tn and qn != ""),
                "exact_sorted_name": int(qns == tns and qns != ""),
                "exact_stripped_name": int(qns2 == tns2 and qns2 != ""),
                "exact_spaceless": int(len(qsp) >= 5 and qsp == tsp),
                "spaceless_subset": int(len(qsp) >= 5 and len(tsp) >= 5 and (qsp in tsp or tsp in qsp) and qsp != tsp),
                "exact_clean_addr": int(qa == ta and len(qa) >= 10),
                "exact_sorted_addr": int(qas == tas and len(qas) >= 12),
                "hno_match": int(qhno == thno and len(qhno) >= 2),
                "pin_match": int(qpin == tpin and len(qpin) >= 5),
                "fw_match": int(qfw == tfw and len(qfw) >= 4),
                
                # Fuzzy name similarities
                "name_jw": JaroWinkler.similarity(qn, tn),
                "name_tok_sort": rfuzz.token_sort_ratio(qn, tn) / 100.0,
                "name_tok_set": rfuzz.token_set_ratio(qn, tn) / 100.0,
                "name_partial": rfuzz.partial_ratio(qn, tn) / 100.0,
                "name_char2_jac": char_ngram_jac(qn, tn, 2),
                "name_char3_jac": char_ngram_jac(qn, tn, 3),
                "name_char4_jac": char_ngram_jac(qn, tn, 4),
                "name_tok_jac": tok_jac(qn, tn),
                "name_tok_ovlp": tok_overlap(qn, tn),
                
                # Address similarities
                "addr_jw": JaroWinkler.similarity(qa, ta) if qa and ta else 0.0,
                "addr_tok_sort": rfuzz.token_sort_ratio(qa, ta) / 100.0 if qa and ta else 0.0,
                "addr_tok_set": rfuzz.token_set_ratio(qa, ta) / 100.0 if qa and ta else 0.0,
                "addr_char3_jac": char_ngram_jac(qa, ta, 3),
                "addr_tok_jac": tok_jac(qas, tas),
                "addr_null_tgt": int(not ta),
                
                # Structural features
                "name_len_diff": abs(len(qn) - len(tn)),
                "name_len_ratio": min(len(qn), len(tn)) / max(len(qn), len(tn), 1),
                "addr_len_diff": abs(len(qa) - len(ta)),
                "name_wc_diff": abs(len(qn.split()) - len(tn.split())),
                "name_wc_ratio": min(len(qn.split()), len(tn.split())) / max(len(qn.split()), len(tn.split()), 1),
                "name_prefix3": int(qn[:3] == tn[:3] if len(qn) >= 3 and len(tn) >= 3 else 0),
                "name_prefix5": int(qn[:5] == tn[:5] if len(qn) >= 5 and len(tn) >= 5 else 0),
                
                # Numeric contradiction features
                "hno_contradiction": int(qhno != thno and bool(qhno) and bool(thno)),
                "pin_contradiction": int(qpin != tpin and bool(qpin) and bool(tpin)),
                "pin_prefix3": int(qpin[:3] == tpin[:3] if len(qpin) >= 3 and len(tpin) >= 3 else 0),
                "pin_both_present": int(bool(qpin) and bool(tpin)),
                
                # Cross-interactions
                "name_x_addr": JaroWinkler.similarity(qn, tn) * (rfuzz.token_sort_ratio(qa, ta) / 100.0 if qa and ta else 0.0),
                "name_x_pin": JaroWinkler.similarity(qn, tn) * float(qpin == tpin and len(qpin) >= 5),
                "sorted_x_addr": (rfuzz.token_sort_ratio(qns, tns) / 100.0) * (rfuzz.token_set_ratio(qa, ta) / 100.0 if qa and ta else 0.0),
                
                # Metadata
                "is_source3": is_s3,
                "india_flag": int("india" in cty),
                "france_flag": int("france" in cty),
            }
            feat_records.append(rec)
            
        flush(f"Feature building completed in {time.time()-t_feat:.1f}s")
        feat_df = pd.DataFrame(feat_records)
        
        # Build labels
        labels = [1 if row.target_id in val_gt.get(row.entity_id, set()) else 0
                  for row in feat_df[["entity_id", "target_id"]].itertuples(index=False)]
        feat_df["label"] = labels
        flush(f"Positive pairs: {sum(labels):,} ({np.mean(labels)*100:.3f}%)")
        
        # Save features
        pl.from_pandas(feat_df).write_parquet(FEAT_FILE)
        flush(f"Saved feature matrix to {FEAT_FILE}")

    # 3. Model Training
    FEAT_COLS = [c for c in feat_df.columns if c not in ("entity_id", "target_id", "label")]
    flush(f"\nFeature set ({len(FEAT_COLS)} features): {FEAT_COLS}")

    X = feat_df[FEAT_COLS].values.astype(np.float32)
    y = feat_df["label"].values.astype(np.int32)
    queries_arr = feat_df["entity_id"].values
    targets_arr = feat_df["target_id"].values
    _, groups = np.unique(queries_arr, return_inverse=True)

    oof_lgb = np.zeros(len(X), dtype=np.float32)
    oof_xgb = np.zeros(len(X), dtype=np.float32)
    oof_cb  = np.zeros(len(X), dtype=np.float32)

    lgb_params = dict(
        objective="binary", metric="auc", learning_rate=0.03, num_leaves=127,
        min_child_samples=20, n_estimators=1000, n_jobs=-1, random_state=42, verbose=-1,
        colsample_bytree=0.8, subsample=0.8, subsample_freq=1
    )
    xgb_params = dict(
        objective="binary:logistic", eval_metric="auc", learning_rate=0.03,
        max_depth=7, n_estimators=1000, tree_method="hist", device="cuda",
        random_state=42, colsample_bytree=0.8, subsample=0.8
    )
    cb_params = dict(
        iterations=1000, learning_rate=0.03, depth=7, task_type="GPU",
        eval_metric="AUC", random_seed=42, verbose=0
    )

    gkf = GroupKFold(n_splits=5)
    fold_aucs = {"lgb": [], "xgb": [], "cb": []}

    flush("\nStarting 5-Fold GroupKFold Cross-Validation...")
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
        flush(f"\n--- Fold {fold+1}/5 --- Train: {len(tr_idx):,} | Val: {len(va_idx):,}")
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        
        # 1. LightGBM
        t0 = time.time()
        model_lgb = lgb.LGBMClassifier(**lgb_params)
        model_lgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        oof_lgb[va_idx] = model_lgb.predict_proba(X_va)[:, 1]
        auc_lgb = roc_auc_score(y_va, oof_lgb[va_idx])
        fold_aucs["lgb"].append(auc_lgb)
        flush(f"  LightGBM: AUC={auc_lgb:.6f} ({time.time()-t0:.1f}s)")
        
        # 2. XGBoost GPU
        t0 = time.time()
        model_xgb = xgb.XGBClassifier(**xgb_params)
        model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        oof_xgb[va_idx] = model_xgb.predict_proba(X_va)[:, 1]
        auc_xgb = roc_auc_score(y_va, oof_xgb[va_idx])
        fold_aucs["xgb"].append(auc_xgb)
        flush(f"  XGBoost-GPU: AUC={auc_xgb:.6f} ({time.time()-t0:.1f}s)")
        
        # 3. CatBoost GPU
        t0 = time.time()
        model_cb = CatBoostClassifier(**cb_params)
        model_cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50)
        oof_cb[va_idx] = model_cb.predict_proba(X_va)[:, 1]
        auc_cb = roc_auc_score(y_va, oof_cb[va_idx])
        fold_aucs["cb"].append(auc_cb)
        flush(f"  CatBoost-GPU: AUC={auc_cb:.6f} ({time.time()-t0:.1f}s)")

    flush("\n" + "=" * 50)
    flush(f"5-Fold CV Mean AUC LightGBM:   {np.mean(fold_aucs['lgb']):.6f}")
    flush(f"5-Fold CV Mean AUC XGBoost:    {np.mean(fold_aucs['xgb']):.6f}")
    flush(f"5-Fold CV Mean AUC CatBoost:   {np.mean(fold_aucs['cb']):.6f}")
    flush("=" * 50)

    # 4. Meta-Ensemble Stacking
    flush("\nTraining Meta-Stacker...")
    meta_X = np.column_stack([oof_lgb, oof_xgb, oof_cb])
    meta_model = LogisticRegression(C=10.0, max_iter=500)
    meta_model.fit(meta_X, y)
    oof_meta = meta_model.predict_proba(meta_X)[:, 1]
    meta_auc = roc_auc_score(y, oof_meta)
    flush(f"Meta-Stacker Out-of-Fold AUC: {meta_auc:.6f}")
    flush(f"Meta-Learner Weights: LGB={meta_model.coef_[0][0]:.4f}, XGB={meta_model.coef_[0][1]:.4f}, CB={meta_model.coef_[0][2]:.4f}")

    # 5. Threshold & Cardinality Decision Search
    flush("\nOptimizing Decision Rule for Macro F(0.5)...")
    best_f05 = 0.0
    best_theta = 0.5
    best_cap = True

    for cap_choice in [True, False]:
        cap_name = "Per-source Cap (1 S2 + 1 S3)" if cap_choice else "Uncapped"
        flush(f"  Evaluating {cap_name}...")
        for theta in np.arange(0.30, 0.85, 0.02):
            score = compute_macro_f05(oof_meta, queries_arr, targets_arr, val_gt, val_query_list, theta, cap_per_source=cap_choice)
            if score > best_f05:
                best_f05 = score
                best_theta = round(float(theta), 3)
                best_cap = cap_choice

    flush("\n" + "=" * 70)
    flush(f"CHAMPION OUT-OF-FOLD RESULT:")
    flush(f"  Best Macro F(0.5): {best_f05:.5f}")
    flush(f"  Optimal Threshold: {best_theta}")
    flush(f"  Cardinality Rule:  {'Per-source cap (max 1 S2, 1 S3)' if best_cap else 'Flat threshold'}")
    flush("=" * 70)

    # Save models & metadata
    pickle.dump(model_lgb, open(MODEL_DIR / "meta_lgb.pkl", "wb"))
    pickle.dump(model_xgb, open(MODEL_DIR / "meta_xgb.pkl", "wb"))
    pickle.dump(model_cb, open(MODEL_DIR / "meta_cb.pkl", "wb"))
    pickle.dump(meta_model, open(MODEL_DIR / "meta_stacker.pkl", "wb"))

    summary = {
        "best_macro_f05": float(best_f05),
        "best_theta": float(best_theta),
        "best_cap": best_cap,
        "mean_auc_lgb": float(np.mean(fold_aucs["lgb"])),
        "mean_auc_xgb": float(np.mean(fold_aucs["xgb"])),
        "mean_auc_cb": float(np.mean(fold_aucs["cb"])),
        "meta_auc": float(meta_auc),
        "features": FEAT_COLS,
    }
    with open(EXP_DIR / "meta_ensemble_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    flush("Artifacts and models saved successfully.")

if __name__ == "__main__":
    main()
