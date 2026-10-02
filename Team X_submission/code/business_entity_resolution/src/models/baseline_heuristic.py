"""
experiments/phase2/models/baseline_heuristic.py
================================================
EXP-01: Baseline Heuristic Scorer with Calibrated Threshold Search.
Computes composite lexical + provenance matching scores:
  score = channel_prio + 0.15*(n_channels - 1) + 0.5*name_jw + 0.3*addr_jaccard + 0.2*postal_match - 0.4*contradiction_postal
Performs exact macro F0.5 optimization over decision threshold theta.
"""

import sys
import os
import json
import time
from typing import Dict, Set
import polars as pl
import numpy as np

sys.path.insert(0, os.path.abspath("."))
try:
    from src.validation_framework import load_validation_dataset, evaluate_matching_predictions
except ImportError:
    try:
        from validation_framework import load_validation_dataset, evaluate_matching_predictions
    except ImportError:
        load_validation_dataset, evaluate_matching_predictions = None, None


def run_heuristic_baseline(features_path: str = "experiments/phase2/val_50k_features.parquet"):
    print("=" * 70, flush=True)
    print("EXP-01: EVALUATING BASELINE HEURISTIC SCORER (CALIBRATED THRESHOLDS)", flush=True)
    print("=" * 70, flush=True)

    t0 = time.time()
    # 1. Load validation queries and ground truth
    val_path = "experiments/val_sample_50k.parquet"
    print(f"Loading ground truth from {val_path}...", flush=True)
    val_df, gt_dict = load_validation_dataset(val_path)

    # 2. Load precomputed candidate features
    print(f"Loading candidate features from {features_path}...", flush=True)
    feats_df = pl.read_parquet(features_path)
    print(f"Loaded {len(feats_df):,} candidate pairs.", flush=True)

    # 3. Compute heuristic composite score
    print("Computing heuristic composite score...", flush=True)
    scored_df = feats_df.with_columns([
        (
            pl.col("channel_max_priority")
            + (pl.col("channel_count") - 1.0) * 0.15
            + pl.col("name_jaro_winkler") * 0.50
            + pl.col("addr_token_jaccard") * 0.30
            + pl.col("postal_code_match") * 0.20
            - pl.col("contradiction_postal") * 0.40
            - pl.col("contradiction_house_no") * 0.25
        ).alias("heuristic_score")
    ])

    print("Score distribution summary:")
    stats = scored_df["heuristic_score"].describe()
    for row in stats.iter_rows(named=True):
        print(f"  {row['statistic']:<10}: {row['value']:.4f}", flush=True)

    # 4. Grid search over decision threshold theta
    all_s1_ids = list(gt_dict.keys())
    thresholds = [0.8, 1.0, 1.1, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.6]
    
    best_f05 = -1.0
    best_theta = None
    best_pred_dict = None
    best_eval = None
    grid_records = []

    print("\n--- THRESHOLD GRID SEARCH FOR MACRO F0.5 ---", flush=True)
    # Pre-extract entity_id, target_id, and scores as numpy arrays for fast filtering
    q_arr = scored_df["entity_id"].to_numpy()
    t_arr = scored_df["target_id"].to_numpy()
    s_arr = scored_df["heuristic_score"].to_numpy()

    for theta in thresholds:
        mask = s_arr >= theta
        filt_q = q_arr[mask]
        filt_t = t_arr[mask]

        # Build prediction dict
        pred_dict: Dict[str, Set[str]] = {sid: set() for sid in all_s1_ids}
        for q, t in zip(filt_q, filt_t):
            pred_dict[q].add(t)

        # Quick evaluation
        ev = evaluate_matching_predictions(pred_dict, gt_dict, s1_df=None, verbose=False)
        f05 = ev["macro_f05"]
        p = ev["mean_precision"]
        r = ev["mean_recall"]
        sing_acc = ev["singleton_accuracy"]
        tot_preds = ev["link_metrics"]["total_pred_links"]

        grid_records.append({
            "theta": theta,
            "macro_f05": f05,
            "precision": p,
            "recall": r,
            "singleton_acc": sing_acc,
            "pred_links": tot_preds
        })
        print(f"  theta = {theta:.2f} | Macro F0.5 = {f05:.4f} | Prec = {p:.4f} | Rec = {r:.4f} | SingAcc = {sing_acc:.4f} | Pred Links = {tot_preds:,}", flush=True)

        if f05 > best_f05:
            best_f05 = f05
            best_theta = theta
            best_pred_dict = pred_dict
            best_eval = ev

    print(f"\nOptimal Decision Threshold: theta* = {best_theta:.2f} with Macro F0.5 = {best_f05:.5f}")

    # 5. Full Slice Evaluation with optimal threshold
    print("\nExecuting comprehensive slice evaluation at optimal threshold...")
    final_eval = evaluate_matching_predictions(best_pred_dict, gt_dict, s1_df=val_df, verbose=True)

    # 6. Save results artifact
    result_artifact = {
        "experiment_id": "EXP-01-BASE-HEURISTIC",
        "optimal_theta": best_theta,
        "grid_search": grid_records,
        "final_evaluation": final_eval,
        "runtime_seconds": round(time.time() - t0, 2)
    }

    out_file = "experiments/phase2/exp01_heuristic_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result_artifact, f, indent=2)

    print(f"Results saved to {out_file} in {result_artifact['runtime_seconds']}s")
    return result_artifact

if __name__ == "__main__":
    run_heuristic_baseline()
