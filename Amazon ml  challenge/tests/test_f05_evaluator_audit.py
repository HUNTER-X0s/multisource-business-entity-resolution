"""
tests/test_f05_evaluator_audit.py
=================================
Independent Verification & Forensic Audit of the Competition Macro F0.5 Evaluator.

Checks:
  1. Mathematical correctness of F0.5 formula (beta=0.5).
  2. Singleton boundary semantics (empty GT vs empty pred = 1.0, empty GT vs non-empty pred = 0.0).
  3. No-match recall boundary (non-empty GT vs empty pred = 0.0).
  4. Zero-denominator safety.
  5. Exact agreement between independent reference evaluator and src/evaluate.py.
  6. Randomized fuzz-testing across 5,000 synthetic queries with diverse cardinality & noise.
"""

import sys
import os
import random
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from src.evaluate import f05_single, macro_f05_from_dicts


def independent_f05(true_set: set, pred_set: set) -> float:
    """Independent from-scratch implementation of per-query F0.5."""
    t_len = len(true_set)
    p_len = len(pred_set)
    
    # Singleton case
    if t_len == 0:
        return 1.0 if p_len == 0 else 0.0
    if p_len == 0:
        return 0.0
        
    tp = len(set(true_set) & set(pred_set))
    if tp == 0:
        return 0.0
        
    # Official beta=0.5 formula: 1.25 * tp / (0.25 * |true| + |pred|)
    return (1.25 * tp) / (0.25 * t_len + p_len)


def independent_macro_f05(gt_dict: dict, pred_dict: dict, query_ids: list) -> float:
    """Independent from-scratch macro average."""
    scores = [independent_f05(gt_dict.get(q, set()), pred_dict.get(q, set())) for q in query_ids]
    return float(np.mean(scores)) if scores else 0.0


def test_singleton_boundary_cases():
    # Empty GT, Empty Pred -> 1.0
    assert f05_single(set(), set()) == 1.0
    assert independent_f05(set(), set()) == 1.0
    
    # Empty GT, Non-empty Pred -> 0.0 (FP penalty on singleton)
    assert f05_single(set(), {"S2-123"}) == 0.0
    assert independent_f05(set(), {"S2-123"}) == 0.0
    
    # Non-empty GT, Empty Pred -> 0.0 (FN penalty on true match)
    assert f05_single({"S2-123"}, set()) == 0.0
    assert independent_f05({"S2-123"}, set()) == 0.0


def test_standard_precision_recall_cases():
    # Perfect match 1-to-1
    assert f05_single({"A"}, {"A"}) == 1.0
    assert independent_f05({"A"}, {"A"}) == 1.0
    
    # 1 true, 2 pred (1 TP, 1 FP) -> P=0.5, R=1.0 -> 5/9
    score = f05_single({"A"}, {"A", "B"})
    assert abs(score - (5.0 / 9.0)) < 1e-9
    assert abs(independent_f05({"A"}, {"A", "B"}) - (5.0 / 9.0)) < 1e-9
    
    # 2 true, 1 pred (1 TP, 1 FN) -> P=1.0, R=0.5 -> 1.25 / 1.5 = 5/6
    score2 = f05_single({"A", "B"}, {"A"})
    assert abs(score2 - (5.0 / 6.0)) < 1e-9
    assert abs(independent_f05({"A", "B"}, {"A"}) - (5.0 / 6.0)) < 1e-9


def test_macro_averaging_fuzz():
    """Fuzz test across 5,000 randomized queries."""
    rng = random.Random(2026)
    pool = [f"TGT-{i:05d}" for i in range(500)]
    
    gt_dict = {}
    pred_dict = {}
    query_ids = [f"QRY-{i:05d}" for i in range(5000)]
    
    for qid in query_ids:
        # 10% chance of singleton
        if rng.random() < 0.10:
            gt_dict[qid] = set()
        else:
            k_gt = rng.randint(1, 8)
            gt_dict[qid] = set(rng.sample(pool, k_gt))
            
        # Prediction generation with diverse overlap
        r = rng.random()
        if r < 0.15:
            pred_dict[qid] = set()  # Predict no match
        elif r < 0.40:
            pred_dict[qid] = set(gt_dict[qid])  # Perfect match
        elif r < 0.70:
            # Partial overlap + noise
            sample_gt = set(rng.sample(list(gt_dict[qid]), max(1, len(gt_dict[qid]) // 2))) if gt_dict[qid] else set()
            noise = set(rng.sample(pool, rng.randint(1, 3)))
            pred_dict[qid] = sample_gt | noise
        else:
            # Pure noise
            pred_dict[qid] = set(rng.sample(pool, rng.randint(1, 5)))
            
    # Compute using existing framework
    eval_res = macro_f05_from_dicts(gt_dict=gt_dict, pred_dict=pred_dict, s1_ids=query_ids)
    existing_macro = eval_res["macro_f05"]
    
    # Compute using independent implementation
    independent_macro = independent_macro_f05(gt_dict, pred_dict, query_ids)
    
    assert abs(existing_macro - independent_macro) < 1e-12, (
        f"Evaluator mismatch! Existing: {existing_macro}, Independent: {independent_macro}"
    )


if __name__ == "__main__":
    test_singleton_boundary_cases()
    test_standard_precision_recall_cases()
    test_macro_averaging_fuzz()
    print("ALL EVALUATOR VERIFICATION AUDIT TESTS PASSED!")
