"""
scratch/test_margin_plus_cap.py
Test combining Strategy 2 (Margin Gating) with Strategy 3 (Cardinality Capping)
"""
import sys, pickle, json
from pathlib import Path
from collections import defaultdict
import numpy as np
import polars as pl

ROOT = Path("z:/Amazon ML")
sys.path.insert(0, str(ROOT))
from src.evaluate import macro_f05_from_dicts

FEAT_FILE   = ROOT / "experiments" / "phase2" / "val_50k_features.parquet"
VAL_SAMP    = ROOT / "experiments" / "val_sample_50k.parquet"
OOF_NPZ     = ROOT / "experiments" / "phase2" / "oof_preds.npz"
STACKER_PKL = ROOT / "experiments" / "phase2" / "models" / "meta_stacker.pkl"

val_df = pl.read_parquet(VAL_SAMP)
gt_map = {}
query_country = {}
for row in val_df.iter_rows(named=True):
    qid = str(row["entity_id"])
    query_country[qid] = str(row.get("country", "") or "").upper()
    raw = str(row.get("matched_entity_ids", "") or "").strip()
    targets = set()
    if raw and raw not in ("nan", "None", ""):
        for t in raw.split(","):
            t = t.strip()
            if t:
                targets.add(t)
    gt_map[qid] = targets
val_query_list = sorted(gt_map.keys())

feat_df = pl.read_parquet(FEAT_FILE)
queries = feat_df["entity_id"].to_numpy().astype(str)
target_ids = feat_df["target_id"].to_numpy().astype(str)

npz = np.load(OOF_NPZ)
oof_lgb = npz["oof_lgb"]
oof_xgb = npz["oof_xgb"]
oof_cb  = npz["oof_cb"]

with open(STACKER_PKL, "rb") as f:
    meta_data = pickle.load(f)
meta_lr = meta_data["meta_lr"]

meta_X = np.column_stack([oof_lgb, oof_xgb, oof_cb])
meta_probs = meta_lr.predict_proba(meta_X)[:, 1]

query_cands = defaultdict(list)
for qid, tid, p in zip(queries, target_ids, meta_probs):
    is_s3 = tid.startswith("S3-")
    query_cands[qid].append((tid, float(p), is_s3))

for qid in query_cands:
    query_cands[qid].sort(key=lambda x: x[1], reverse=True)

# Grid search combining margin gating and source cap
best_f05 = 0.0
best_cfg = None

for th_high in [0.70, 0.72, 0.75, 0.78]:
    for th_low in [0.48, 0.50, 0.52, 0.55]:
        for min_gap in [0.03, 0.05, 0.07, 0.10]:
            for cap in [4, 5, 6, 8]:
                pred_dict = {}
                for q in val_query_list:
                    cands = query_cands.get(q, [])
                    if not cands:
                        pred_dict[q] = set()
                        continue
                    preds = set()
                    top_p = cands[0][1]
                    s2_c = s3_c = 0
                    for idx, (t, p, is_s3) in enumerate(cands):
                        accept = False
                        if p >= th_high:
                            accept = True
                        elif p >= th_low:
                            if idx == 0 and len(cands) > 1 and (top_p - cands[1][1] >= min_gap):
                                accept = True
                            elif idx == 0 and len(cands) == 1:
                                accept = True
                        if accept:
                            if is_s3 and s3_c < cap:
                                preds.add(t); s3_c += 1
                            elif not is_s3 and s2_c < cap:
                                preds.add(t); s2_c += 1
                    pred_dict[q] = preds
                res = macro_f05_from_dicts(gt_map, pred_dict, val_query_list)
                f05 = res["macro_f05"]
                if f05 > best_f05:
                    best_f05 = f05
                    best_cfg = (th_high, th_low, min_gap, cap, res)
                    print(f"New Best: th_high={th_high}, th_low={th_low}, gap={min_gap}, cap={cap} -> F0.5={f05:.5f}", flush=True)

print("=" * 60)
print(f"ULTIMATE BEST SET-DECISION: F0.5 = {best_f05:.5f}")
print("Config:", best_cfg[:4])
print("Metrics:", best_cfg[4])
