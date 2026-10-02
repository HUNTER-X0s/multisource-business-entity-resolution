"""
experiments/phase2/validation_framework.py
===========================================
Formal Evaluation & Validation Framework for Phase 2.
Implements the exact competition Macro F0.5 evaluation metric,
slice-level diagnostic audits, and ground-truth pairing logic.
"""

import sys
import os
import json
import time
from typing import Dict, Set, List, Optional, Any
import polars as pl
import numpy as np

# Ensure root workspace is in sys.path
sys.path.insert(0, os.path.abspath("."))
from src.evaluate import f05_single, macro_f05_from_dicts


def load_validation_dataset(val_parquet_path: str = "experiments/val_sample_50k.parquet"):
    """
    Loads the validation sample and parses true match sets.
    Returns:
        s1_df: Polars DataFrame of S1 queries and metadata
        gt_dict: Dict[str, Set[str]] mapping entity_id to set of true target_ids
    """
    if not os.path.exists(val_parquet_path):
        raise FileNotFoundError(f"Validation dataset not found at {val_parquet_path}")

    s1_df = pl.read_parquet(val_parquet_path)
    
    gt_dict: Dict[str, Set[str]] = {}
    for row in s1_df.iter_rows(named=True):
        sid = row["entity_id"]
        m_str = row.get("matched_entity_ids") or ""
        gt_dict[sid] = set(m.strip() for m in m_str.split(",") if m.strip())
        
    return s1_df, gt_dict


def evaluate_matching_predictions(
    pred_dict: Dict[str, Set[str]],
    gt_dict: Dict[str, Set[str]],
    s1_df: Optional[pl.DataFrame] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Evaluates a candidate prediction dictionary {entity_id -> set of predicted target_ids}
    against ground truth using the official competition Macro F0.5 formula and
    comprehensive slice-level diagnostic breakdowns.
    """
    t0 = time.time()
    all_s1_ids = list(gt_dict.keys())
    
    # 1. Official Global Macro F0.5 Metric
    base_metrics = macro_f05_from_dicts(gt_dict=gt_dict, pred_dict=pred_dict, s1_ids=all_s1_ids)
    
    # 2. Link-Level Aggregate Metrics
    total_true_links = sum(len(s) for s in gt_dict.values())
    total_pred_links = sum(len(pred_dict.get(sid, set())) for sid in all_s1_ids)
    
    tp_links = 0
    s2_true_total = 0
    s2_true_recovered = 0
    s3_true_total = 0
    s3_true_recovered = 0
    
    for sid, true_set in gt_dict.items():
        preds = pred_dict.get(sid, set())
        matched = true_set & preds
        tp_links += len(matched)
        
        for tid in true_set:
            if tid.startswith("S2-"):
                s2_true_total += 1
                if tid in preds:
                    s2_true_recovered += 1
            elif tid.startswith("S3-"):
                s3_true_total += 1
                if tid in preds:
                    s3_true_recovered += 1
                    
    link_precision = tp_links / total_pred_links if total_pred_links > 0 else 0.0
    link_recall = tp_links / total_true_links if total_true_links > 0 else 0.0
    link_f05_denom = 0.25 * link_precision + link_recall
    link_f05 = (1.25 * link_precision * link_recall / link_f05_denom) if link_f05_denom > 0 else 0.0

    s2_recall = (s2_true_recovered / s2_true_total) if s2_true_total > 0 else 0.0
    s3_recall = (s3_true_recovered / s3_true_total) if s3_true_total > 0 else 0.0

    # 3. Slice Breakdown (if s1_df is provided)
    slice_metrics = {}
    if s1_df is not None:
        # A. Country Slices
        for country in ["US", "India"]:
            c_ids = s1_df.filter(pl.col("country") == country)["entity_id"].to_list()
            if c_ids:
                c_gt = {k: gt_dict[k] for k in c_ids if k in gt_dict}
                c_pred = {k: pred_dict.get(k, set()) for k in c_ids}
                c_res = macro_f05_from_dicts(gt_dict=c_gt, pred_dict=c_pred, s1_ids=c_ids)
                slice_metrics[f"country_{country}"] = {
                    "macro_f05": c_res["macro_f05"],
                    "mean_precision": c_res["mean_precision"],
                    "mean_recall": c_res["mean_recall"],
                    "n_entities": len(c_ids),
                }

        # B. Match Bin Slices
        # 0: Singletons, 1: Exactly 1 match, 2: 2-3 matches, 3: 4-6 matches, 4: 7+ matches
        for m_bin in [0, 1, 2, 3, 4]:
            b_ids = s1_df.filter(pl.col("match_bin") == m_bin)["entity_id"].to_list()
            if b_ids:
                b_gt = {k: gt_dict[k] for k in b_ids if k in gt_dict}
                b_pred = {k: pred_dict.get(k, set()) for k in b_ids}
                b_res = macro_f05_from_dicts(gt_dict=b_gt, pred_dict=b_pred, s1_ids=b_ids)
                slice_metrics[f"match_bin_{m_bin}"] = {
                    "macro_f05": b_res["macro_f05"],
                    "mean_precision": b_res["mean_precision"],
                    "mean_recall": b_res["mean_recall"],
                    "n_entities": len(b_ids),
                }

    output = {
        "macro_f05": round(base_metrics["macro_f05"], 5),
        "mean_precision": round(base_metrics["mean_precision"], 5),
        "mean_recall": round(base_metrics["mean_recall"], 5),
        "singleton_accuracy": round(base_metrics["singleton_accuracy"], 5),
        "n_entities": base_metrics["n_entities"],
        "n_singletons": base_metrics["n_singletons_gt"],
        "link_metrics": {
            "total_true_links": total_true_links,
            "total_pred_links": total_pred_links,
            "true_positive_links": tp_links,
            "link_precision": round(link_precision, 5),
            "link_recall": round(link_recall, 5),
            "link_f05": round(link_f05, 5),
            "s2_recall": round(s2_recall, 5),
            "s3_recall": round(s3_recall, 5),
        },
        "slice_metrics": slice_metrics,
        "eval_time_sec": round(time.time() - t0, 3)
    }

    if verbose:
        print("\n" + "=" * 65)
        print(f"OFFICIAL VALIDATION EVALUATION (N = {output['n_entities']:,})")
        print("=" * 65)
        print(f"  Macro F0.5:          {output['macro_f05']:.5f}  (PRIMARY COMPETITION METRIC)")
        print(f"  Macro Precision:     {output['mean_precision']:.5f}")
        print(f"  Macro Recall:        {output['mean_recall']:.5f}")
        print(f"  Singleton Accuracy:  {output['singleton_accuracy']:.5f} ({output['n_singletons']:,} true singletons)")
        print("-" * 65)
        print(f"  Link Precision:      {output['link_metrics']['link_precision']:.5f} ({tp_links:,} / {total_pred_links:,} predicted)")
        print(f"  Link Recall:         {output['link_metrics']['link_recall']:.5f} ({tp_links:,} / {total_true_links:,} true)")
        print(f"  S2 Link Recall:      {output['link_metrics']['s2_recall']:.5f}")
        print(f"  S3 Link Recall:      {output['link_metrics']['s3_recall']:.5f}")
        if slice_metrics:
            print("-" * 65)
            print("  Slice Breakdown (Macro F0.5):")
            for k, v in slice_metrics.items():
                print(f"    {k:<16}: F0.5 = {v['macro_f05']:.4f} | Prec = {v['mean_precision']:.4f} | Rec = {v['mean_recall']:.4f} (N={v['n_entities']:,})")
        print("=" * 65 + "\n")

    return output


def label_candidate_pairs(candidates_df: pl.DataFrame, gt_dict: Dict[str, Set[str]]) -> pl.DataFrame:
    """
    Annotates candidate pairs with binary ground-truth match labels (1 = match, 0 = non-match).
    Input DataFrame must have columns: ['entity_id', 'target_id'].
    """
    # Create mapping of (entity_id, target_id) pairs to label
    cand_s1 = candidates_df["entity_id"].to_list()
    cand_tgt = candidates_df["target_id"].to_list()
    
    labels = [
        1 if t in gt_dict.get(s, set()) else 0
        for s, t in zip(cand_s1, cand_tgt)
    ]
    
    return candidates_df.with_columns(pl.Series("label", labels, dtype=pl.Int8))
