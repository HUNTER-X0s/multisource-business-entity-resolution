"""
experiments/phase2/oof_stacking_exp16.py
==========================================
EXP-16: 5-Fold Leakage-Safe Out-Of-Fold (OOF) Stacking

Uses query-grouped 5-fold cross-validation to produce OOF predictions
from LightGBM (CPU), XGBoost-GPU, and CatBoost-GPU.
Then trains a logistic regression meta-stacker on OOF probabilities.

Split guarantee: no query appears in both train and val within any fold.

Outputs:
  - experiments/phase2/exp16_oof_stacking_results.json
  - experiments/phase2/models/meta_stacker.pkl
  - experiments/phase2/oof_preds.npz
"""
import os, sys, json, time, pickle, re
from pathlib import Path
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.evaluate import macro_f05_from_dicts

FEAT_FILE = ROOT / "experiments" / "phase2" / "val_50k_features.parquet"
VAL_SAMP  = ROOT / "experiments" / "val_sample_50k.parquet"
OUT_JSON  = ROOT / "experiments" / "phase2" / "exp16_oof_stacking_results.json"
OUT_MODEL = ROOT / "experiments" / "phase2" / "models" / "meta_stacker.pkl"
OOF_NPZ   = ROOT / "experiments" / "phase2" / "oof_preds.npz"
N_FOLDS   = 5

def flush(*args, **kwargs):
    print(*args, **kwargs, flush=True)

flush("=" * 70)
flush("EXP-16: 5-FOLD OOF STACKING META-LEARNER")
flush("=" * 70)
flush(f"PyTorch CUDA Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# [0] Load data
flush("\n[0] Loading features and GT...")
t0 = time.time()
feat_df = pl.read_parquet(FEAT_FILE)
val_df  = pl.read_parquet(VAL_SAMP)

# Build GT map from val_sample (matched_entity_ids column)
gt_map = {}
for row in val_df.iter_rows(named=True):
    qid = str(row["entity_id"])
    raw = str(row.get("matched_entity_ids", "") or "").strip()
    targets = set()
    if raw and raw not in ("nan", "None", ""):
        for t in raw.split(","):
            t = t.strip()
            if t:
                targets.add(t)
    gt_map[qid] = targets
val_query_list = sorted(gt_map.keys())

# Label candidate pairs
labels = []
for row in feat_df.select(["entity_id", "target_id"]).iter_rows(named=True):
    qid = str(row["entity_id"])
    tid = str(row["target_id"])
    labels.append(1 if tid in gt_map.get(qid, set()) else 0)

feat_df = feat_df.with_columns(pl.Series("label", labels))

FEATURE_COLS = [c for c in feat_df.columns
                if c not in ("entity_id", "target_id", "label", "business_name",
                             "business_address", "country")]
flush(f"  Pairs: {len(feat_df):,}  |  Features: {len(FEATURE_COLS)}  |  Pos rate: {np.mean(labels)*100:.2f}%")

# Arrays
X = feat_df.select(FEATURE_COLS).to_numpy().astype(np.float32)
y = np.array(labels, dtype=np.int32)
queries = feat_df["entity_id"].to_numpy().astype(str)
target_ids = feat_df["target_id"].to_numpy().astype(str)
_, groups = np.unique(queries, return_inverse=True)
flush(f"  X shape: {X.shape}  |  Unique queries: {len(set(queries)):,}")
flush(f"  Time: {time.time()-t0:.1f}s")

# OOF containers
oof_lgb = np.zeros(len(X), dtype=np.float32)
oof_xgb = np.zeros(len(X), dtype=np.float32)
oof_cb  = np.zeros(len(X), dtype=np.float32)
oof_y   = y.copy()

# LGB params
lgb_params = dict(
    objective="binary", metric="auc", learning_rate=0.05,
    num_leaves=63, min_child_samples=20, n_estimators=600,
    n_jobs=-1, random_state=42, verbose=-1,
)
# XGBoost GPU params
xgb_params = dict(
    objective="binary:logistic", eval_metric="auc",
    learning_rate=0.05, max_depth=6, n_estimators=600,
    device="cuda", random_state=42,
)
# CatBoost GPU params
cb_params = dict(
    iterations=600, learning_rate=0.05, depth=6,
    task_type="GPU", eval_metric="AUC",
    random_seed=42, verbose=0,
)

gkf = GroupKFold(n_splits=N_FOLDS)
fold_aucs = {"lgb": [], "xgb": [], "cb": []}
vram_readings = []

flush(f"\n[1] Running {N_FOLDS}-fold OOF...")
for fold_idx, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
    t_fold = time.time()
    flush(f"\n  Fold {fold_idx+1}/{N_FOLDS}:")
    flush(f"    Train: {len(tr_idx):,} pairs  |  Val: {len(val_idx):,} pairs")

    X_tr, X_va = X[tr_idx], X[val_idx]
    y_tr, y_va = y[tr_idx], y[val_idx]

    # LightGBM
    t_lgb = time.time()
    lgb_model = lgb.LGBMClassifier(**lgb_params)
    lgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)])
    oof_lgb[val_idx] = lgb_model.predict_proba(X_va)[:, 1]
    auc_lgb = roc_auc_score(y_va, oof_lgb[val_idx])
    fold_aucs["lgb"].append(auc_lgb)
    flush(f"    LGB  CPU : AUC={auc_lgb:.5f} ({time.time()-t_lgb:.1f}s)")

    # XGBoost GPU
    t_xgb = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_xgb[val_idx] = xgb_model.predict_proba(X_va)[:, 1]
    auc_xgb = roc_auc_score(y_va, oof_xgb[val_idx])
    fold_aucs["xgb"].append(auc_xgb)
    xgb_vram = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
    vram_readings.append(xgb_vram)
    flush(f"    XGB  GPU : AUC={auc_xgb:.5f} ({time.time()-t_xgb:.1f}s) | Peak VRAM: {xgb_vram:.1f}MB")

    # CatBoost GPU
    t_cb = time.time()
    cb_model = CatBoostClassifier(**cb_params)
    cb_model.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=False)
    oof_cb[val_idx] = cb_model.predict_proba(X_va)[:, 1]
    auc_cb = roc_auc_score(y_va, oof_cb[val_idx])
    fold_aucs["cb"].append(auc_cb)
    cb_vram = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
    vram_readings.append(cb_vram)
    flush(f"    CatBoost GPU: AUC={auc_cb:.5f} ({time.time()-t_cb:.1f}s) | Peak VRAM: {cb_vram:.1f}MB")
    flush(f"    Fold time: {time.time()-t_fold:.1f}s")

flush(f"\n  OOF AUCs:")
flush(f"    LGB  : mean={np.mean(fold_aucs['lgb']):.5f} std={np.std(fold_aucs['lgb']):.5f}")
flush(f"    XGB  : mean={np.mean(fold_aucs['xgb']):.5f} std={np.std(fold_aucs['xgb']):.5f}")
flush(f"    CB   : mean={np.mean(fold_aucs['cb']):.5f}  std={np.std(fold_aucs['cb']):.5f}")

# Save OOF predictions
np.savez_compressed(OOF_NPZ, oof_lgb=oof_lgb, oof_xgb=oof_xgb, oof_cb=oof_cb, oof_y=oof_y)
flush(f"  OOF predictions saved: {OOF_NPZ}")

# [2] Train meta-stacker
flush("\n[2] Training meta-stacker (Logistic Regression)...")
meta_X = np.column_stack([oof_lgb, oof_xgb, oof_cb])
meta_lr = LogisticRegression(C=1.0, max_iter=500, random_state=42)
meta_lr.fit(meta_X, oof_y)
meta_oof_pred = meta_lr.predict_proba(meta_X)[:, 1]
meta_auc = roc_auc_score(oof_y, meta_oof_pred)
flush(f"  Meta-stacker OOF AUC: {meta_auc:.5f}")
flush(f"  Meta coefficients: LGB={meta_lr.coef_[0][0]:.4f}  XGB={meta_lr.coef_[0][1]:.4f}  CB={meta_lr.coef_[0][2]:.4f}")

# [3] Evaluate individual and meta-stacker F0.5 on OOF predictions
flush("\n[3] Evaluating individual models and meta-stacker on OOF predictions...")

def eval_f05_thresholds(prob_array, model_name):
    q_map = {}
    for qid, tid, p in zip(queries, target_ids, prob_array):
        if qid not in q_map:
            q_map[qid] = []
        q_map[qid].append((tid, p))
    best_f, best_th = 0.0, 0.0
    for th in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        pred_dict = {q: {t for t, p in q_map.get(q, []) if p >= th} for q in val_query_list}
        res = macro_f05_from_dicts(gt_map, pred_dict, val_query_list)
        f_val = res["macro_f05"]
        if f_val > best_f:
            best_f, best_th = f_val, th
    flush(f"  {model_name:18s}: Best theta*={best_th:.2f} -> Macro F0.5 = {best_f:.5f}")
    return best_f, best_th

lgb_best_f, lgb_th = eval_f05_thresholds(oof_lgb, "LGB OOF")
xgb_best_f, xgb_th = eval_f05_thresholds(oof_xgb, "XGB-GPU OOF")
cb_best_f,  cb_th  = eval_f05_thresholds(oof_cb,  "CatBoost-GPU OOF")
meta_best_f, meta_th = eval_f05_thresholds(meta_oof_pred, "Meta-Stacker OOF")

# Save model
OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_MODEL, "wb") as f:
    pickle.dump({
        "meta_lr": meta_lr,
        "lgb_coeff": meta_lr.coef_[0][0],
        "xgb_coeff": meta_lr.coef_[0][1],
        "cb_coeff":  meta_lr.coef_[0][2],
        "oof_auc":   meta_auc,
        "best_theta": meta_th,
        "best_f05":   meta_best_f,
    }, f)
flush(f"\n  Model saved: {OUT_MODEL}")

output = {
    "experiment": "EXP-16",
    "n_folds": N_FOLDS,
    "fold_aucs_lgb": fold_aucs["lgb"],
    "fold_aucs_xgb": fold_aucs["xgb"],
    "fold_aucs_cb":  fold_aucs["cb"],
    "mean_auc_lgb": float(np.mean(fold_aucs["lgb"])),
    "mean_auc_xgb": float(np.mean(fold_aucs["xgb"])),
    "mean_auc_cb":  float(np.mean(fold_aucs["cb"])),
    "meta_stacker_oof_auc": meta_auc,
    "meta_coeff_lgb": float(meta_lr.coef_[0][0]),
    "meta_coeff_xgb": float(meta_lr.coef_[0][1]),
    "meta_coeff_cb":  float(meta_lr.coef_[0][2]),
    "lgb_oof_best_f05": lgb_best_f,
    "xgb_oof_best_f05": xgb_best_f,
    "cb_oof_best_f05":  cb_best_f,
    "meta_best_f05":    meta_best_f,
    "meta_best_theta":  meta_th,
    "gain_vs_lgb":      meta_best_f - lgb_best_f,
    "peak_vram_mb":     max(vram_readings) if vram_readings else 0.0
}
with open(OUT_JSON, "w") as f:
    json.dump(output, f, indent=2)
flush(f"Saved: {OUT_JSON}")
