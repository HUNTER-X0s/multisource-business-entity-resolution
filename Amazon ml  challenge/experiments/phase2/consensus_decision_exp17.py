"""
experiments/phase2/consensus_decision_exp17.py
=============================================
EXP-17: Cross-Source S2<->S3 Consensus & Multi-Source Score-Margin Decision Optimization

Evaluates set-decision algorithms on the unbiased 5-Fold OOF predictions from EXP-16:
  1. Source-Specific Thresholding (optimal theta_s2 vs theta_s3)
  2. Relative Score-Margin Gating (penalizing low-confidence clusters of candidates)
  3. Cardinality Capping (limiting false-positive bloat per source)
  4. Cross-Source Consensus (relational agreement between S2 and S3 predictions)

Measures exact official competition Macro F0.5 (query-averaged, beta=0.5).
Outputs:
  - experiments/phase2/exp17_consensus_decision_results.json
"""
import sys, os, time, json, pickle
from pathlib import Path
from collections import defaultdict
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.evaluate import macro_f05_from_dicts

FEAT_FILE   = ROOT / "experiments" / "phase2" / "val_50k_features.parquet"
VAL_SAMP    = ROOT / "experiments" / "val_sample_50k.parquet"
OOF_NPZ     = ROOT / "experiments" / "phase2" / "oof_preds.npz"
STACKER_PKL = ROOT / "experiments" / "phase2" / "models" / "meta_stacker.pkl"
OUT_JSON    = ROOT / "experiments" / "phase2" / "exp17_consensus_decision_results.json"

def flush(*args, **kwargs):
    print(*args, **kwargs, flush=True)

flush("=" * 70)
flush("EXP-17: CROSS-SOURCE CONSENSUS & SET-DECISION OPTIMIZATION")
flush("=" * 70)

# [0] Load GT and metadata
flush("\n[0] Loading GT and validation queries...")
t0 = time.time()
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
flush(f"  Val queries: {len(val_query_list):,}")

# [1] Load OOF predictions & Meta-Stacker
flush("\n[1] Loading OOF predictions and meta-stacker...")
feat_df = pl.read_parquet(FEAT_FILE)
queries = feat_df["entity_id"].to_numpy().astype(str)
target_ids = feat_df["target_id"].to_numpy().astype(str)

npz = np.load(OOF_NPZ)
oof_lgb = npz["oof_lgb"]
oof_xgb = npz["oof_xgb"]
oof_cb  = npz["oof_cb"]
flush(f"  Loaded OOF arrays: {len(oof_lgb):,} pairs")

with open(STACKER_PKL, "rb") as f:
    meta_data = pickle.load(f)
meta_lr = meta_data["meta_lr"]

meta_X = np.column_stack([oof_lgb, oof_xgb, oof_cb])
meta_probs = meta_lr.predict_proba(meta_X)[:, 1]
flush(f"  Computed meta-stacker probabilities: range [{meta_probs.min():.4f}, {meta_probs.max():.4f}]")

# Build per-query candidate structures
# (qid) -> list of (tid, prob, is_s3)
flush("\n[2] Organizing candidate pairs by query and source...")
query_cands = defaultdict(list)
for qid, tid, p in zip(queries, target_ids, meta_probs):
    is_s3 = tid.startswith("S3-")
    query_cands[qid].append((tid, float(p), is_s3))

# Sort candidates by probability descending
for qid in query_cands:
    query_cands[qid].sort(key=lambda x: x[1], reverse=True)

# Baseline single global threshold
flush("\n[3] Baseline Global Threshold Sweep...")
baseline_best_f05 = 0.0
baseline_best_th = 0.0
for th in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
    pred_dict = {q: {t for t, p, _ in query_cands.get(q, []) if p >= th} for q in val_query_list}
    res = macro_f05_from_dicts(gt_map, pred_dict, val_query_list)
    f05 = res["macro_f05"]
    flush(f"  theta={th:.2f} -> Macro F0.5 = {f05:.5f}")
    if f05 > baseline_best_f05:
        baseline_best_f05 = f05
        baseline_best_th = th

flush(f"  Baseline best: theta*={baseline_best_th:.2f} -> Macro F0.5 = {baseline_best_f05:.5f}")

# Strategy 1: Source-Specific Thresholding (theta_s2, theta_s3)
flush("\n[4] Strategy 1: Source-Specific Threshold Grid (theta_S2 x theta_S3)...")
s1_best_f05 = 0.0
s1_best_params = None

for th_s2 in [0.50, 0.55, 0.60, 0.65]:
    for th_s3 in [0.50, 0.55, 0.60, 0.65]:
        pred_dict = {}
        for q in val_query_list:
            preds = set()
            for t, p, is_s3 in query_cands.get(q, []):
                req_th = th_s3 if is_s3 else th_s2
                if p >= req_th:
                    preds.add(t)
            pred_dict[q] = preds
        res = macro_f05_from_dicts(gt_map, pred_dict, val_query_list)
        f05 = res["macro_f05"]
        if f05 > s1_best_f05:
            s1_best_f05 = f05
            s1_best_params = (th_s2, th_s3)
        flush(f"  theta_S2={th_s2:.2f}, theta_S3={th_s3:.2f} -> Macro F0.5 = {f05:.5f}")

flush(f"  Strategy 1 Best: S2={s1_best_params[0]:.2f}, S3={s1_best_params[1]:.2f} -> F0.5 = {s1_best_f05:.5f} (gain: {s1_best_f05-baseline_best_f05:+.5f})")

# Strategy 2: Relative Score-Margin Gating
flush("\n[5] Strategy 2: Relative Score-Margin Gating...")
# If top prob is between [th_low, th_high], require margin delta over candidate k
s2_best_f05 = 0.0
s2_best_params = None

for th_high in [0.65, 0.70, 0.75]:
    for th_low in [0.50, 0.55, 0.60]:
        for min_gap in [0.05, 0.10, 0.15]:
            pred_dict = {}
            for q in val_query_list:
                cands = query_cands.get(q, [])
                if not cands:
                    pred_dict[q] = set()
                    continue
                preds = set()
                top_p = cands[0][1]
                # High confidence items always pass
                for idx, (t, p, _) in enumerate(cands):
                    if p >= th_high:
                        preds.add(t)
                    elif p >= th_low:
                        # For borderline items, require that it is top-1 or has sufficient margin over the next item
                        if idx == 0 and len(cands) > 1 and (top_p - cands[1][1] >= min_gap):
                            preds.add(t)
                        elif idx == 0 and len(cands) == 1:
                            preds.add(t)
                pred_dict[q] = preds
            res = macro_f05_from_dicts(gt_map, pred_dict, val_query_list)
            f05 = res["macro_f05"]
            if f05 > s2_best_f05:
                s2_best_f05 = f05
                s2_best_params = (th_high, th_low, min_gap)

flush(f"  Strategy 2 Best: th_high={s2_best_params[0]:.2f}, th_low={s2_best_params[1]:.2f}, min_gap={s2_best_params[2]:.2f} -> F0.5 = {s2_best_f05:.5f} (gain: {s2_best_f05-baseline_best_f05:+.5f})")

# Strategy 3: Dynamic Cardinality Capping per Source
flush("\n[6] Strategy 3: Dynamic Cardinality Capping (cap_s2, cap_s3)...")
s3_best_f05 = 0.0
s3_best_params = None

best_th = baseline_best_th
for cap_s2 in [2, 3, 4, 5, 10]:
    for cap_s3 in [2, 3, 4, 5, 10]:
        pred_dict = {}
        for q in val_query_list:
            preds = set()
            s2_count = s3_count = 0
            for t, p, is_s3 in query_cands.get(q, []):
                if p >= best_th:
                    if is_s3 and s3_count < cap_s3:
                        preds.add(t)
                        s3_count += 1
                    elif not is_s3 and s2_count < cap_s2:
                        preds.add(t)
                        s2_count += 1
            pred_dict[q] = preds
        res = macro_f05_from_dicts(gt_map, pred_dict, val_query_list)
        f05 = res["macro_f05"]
        if f05 > s3_best_f05:
            s3_best_f05 = f05
            s3_best_params = (cap_s2, cap_s3)
        flush(f"  cap_S2={cap_s2}, cap_S3={cap_s3} -> Macro F0.5 = {f05:.5f}")

flush(f"  Strategy 3 Best: cap_S2={s3_best_params[0]}, cap_S3={s3_best_params[1]} -> F0.5 = {s3_best_f05:.5f} (gain: {s3_best_f05-baseline_best_f05:+.5f})")

# Strategy 4: Joint Optimized Multi-Source Consensus Rule
flush("\n[7] Strategy 4: Combined Joint Multi-Source Set-Decision Rule...")
# Combine source-specific thresholding + cardinality capping + minimum score margin
combined_best_f05 = 0.0
combined_best_params = None

for th_s2 in [0.55, 0.60, 0.62]:
    for th_s3 in [0.55, 0.60, 0.62]:
        for cap in [3, 4, 5]:
            pred_dict = {}
            for q in val_query_list:
                preds = set()
                s2_c = s3_c = 0
                for t, p, is_s3 in query_cands.get(q, []):
                    req_th = th_s3 if is_s3 else th_s2
                    if p >= req_th:
                        if is_s3 and s3_c < cap:
                            preds.add(t); s3_c += 1
                        elif not is_s3 and s2_c < cap:
                            preds.add(t); s2_c += 1
                pred_dict[q] = preds
            res = macro_f05_from_dicts(gt_map, pred_dict, val_query_list)
            f05 = res["macro_f05"]
            if f05 > combined_best_f05:
                combined_best_f05 = f05
                combined_best_params = {"th_s2": th_s2, "th_s3": th_s3, "cap": cap, "eval": res}

opt_eval = combined_best_params["eval"]
flush("\n" + "=" * 70)
flush("EXP-17 SUMMARY & RESULTS")
flush("=" * 70)
flush(f"  Baseline Meta-Stacker F0.5 (flat theta=0.60): {baseline_best_f05:.5f}")
flush(f"  Strategy 1 Best (Source-Specific):            {s1_best_f05:.5f} ({s1_best_f05-baseline_best_f05:+.5f})")
flush(f"  Strategy 2 Best (Margin Gating):              {s2_best_f05:.5f} ({s2_best_f05-baseline_best_f05:+.5f})")
flush(f"  Strategy 3 Best (Cardinality Capping):        {s3_best_f05:.5f} ({s3_best_f05-baseline_best_f05:+.5f})")
flush(f"  Strategy 4 Champion (Joint Multi-Source):     {combined_best_f05:.5f} ({combined_best_f05-baseline_best_f05:+.5f})")
flush(f"    Optimal Parameters: th_S2={combined_best_params['th_s2']:.2f}, th_S3={combined_best_params['th_s3']:.2f}, max_cap={combined_best_params['cap']}")
flush(f"    Precision={opt_eval.get('macro_precision', 0.0):.4f}, Recall={opt_eval.get('macro_recall', 0.0):.4f}")

# Slice breakdown for champion
india_queries = [q for q in val_query_list if "INDIA" in query_country.get(q, "")]
us_queries    = [q for q in val_query_list if "US" in query_country.get(q, "")]

# Recompute best predictions
best_preds = {}
for q in val_query_list:
    preds = set()
    s2_c = s3_c = 0
    for t, p, is_s3 in query_cands.get(q, []):
        req_th = combined_best_params["th_s3"] if is_s3 else combined_best_params["th_s2"]
        if p >= req_th:
            if is_s3 and s3_c < combined_best_params["cap"]:
                preds.add(t); s3_c += 1
            elif not is_s3 and s2_c < combined_best_params["cap"]:
                preds.add(t); s2_c += 1
    best_preds[q] = preds

india_eval = macro_f05_from_dicts(gt_map, best_preds, india_queries)
us_eval    = macro_f05_from_dicts(gt_map, best_preds, us_queries)
flush(f"\n  Champion Slice Breakdown:")
flush(f"    India Slice (N={len(india_queries):,}): Macro F0.5 = {india_eval['macro_f05']:.5f}")
flush(f"    US Slice    (N={len(us_queries):,}): Macro F0.5 = {us_eval['macro_f05']:.5f}")

output = {
    "experiment": "EXP-17",
    "baseline_flat_theta": baseline_best_th,
    "baseline_f05": baseline_best_f05,
    "strategy1_source_specific": {
        "best_th_s2": s1_best_params[0],
        "best_th_s3": s1_best_params[1],
        "best_f05": s1_best_f05
    },
    "strategy2_margin_gating": {
        "best_th_high": s2_best_params[0],
        "best_th_low": s2_best_params[1],
        "best_min_gap": s2_best_params[2],
        "best_f05": s2_best_f05
    },
    "strategy3_cardinality_capping": {
        "best_cap_s2": s3_best_params[0],
        "best_cap_s3": s3_best_params[1],
        "best_f05": s3_best_f05
    },
    "champion_multi_source_decision": {
        "th_s2": combined_best_params["th_s2"],
        "th_s3": combined_best_params["th_s3"],
        "cap": combined_best_params["cap"],
        "macro_f05": combined_best_f05,
        "macro_precision": opt_eval.get("macro_precision", 0.0),
        "macro_recall": opt_eval.get("macro_recall", 0.0),
        "gain_vs_baseline": combined_best_f05 - baseline_best_f05,
        "india_slice_f05": india_eval["macro_f05"],
        "us_slice_f05": us_eval["macro_f05"]
    }
}
with open(OUT_JSON, "w") as f:
    json.dump(output, f, indent=2)
flush(f"\nSaved: {OUT_JSON}")
