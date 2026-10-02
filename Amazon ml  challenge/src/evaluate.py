"""
evaluate.py
===========
Local Macro F0.5 evaluator matching the competition's exact scoring formula.

Usage (standalone):
    python src/evaluate.py \\
        --pred  output/matching_results.tsv \\
        --gt    dataset/raw/train/train_ground_truth.tsv \\
        --s1ids dataset/raw/train/train_source1.tsv

Usage (from code):
    from src.evaluate import macro_f05_from_dicts, load_gt_dict, load_pred_dict
"""

import argparse
import sys
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Core F0.5 computation
# ---------------------------------------------------------------------------

def f05_single(true_set: set, pred_set: set) -> float:
    """
    Compute F0.5 for a single Source-1 entity.

    F0.5 = (1 + 0.25) * P * R / (0.25 * P + R)
         = 1.25 * P * R / (0.25 * P + R)

    Returns 1.0 for the singleton case (true_set empty, pred_set empty).
    Returns 0.0 if pred_set is non-empty but true_set is empty (false positive on singleton).
    """
    if len(true_set) == 0 and len(pred_set) == 0:
        return 1.0
    if len(true_set) == 0 and len(pred_set) > 0:
        return 0.0  # FP on a singleton -> cliff
    if len(pred_set) == 0:
        # All true matches missed, recall = 0 -> F0.5 = 0
        return 0.0

    tp = len(true_set & pred_set)
    precision = tp / len(pred_set)
    recall    = tp / len(true_set)

    denom = 0.25 * precision + recall
    if denom == 0.0:
        return 0.0
    return 1.25 * precision * recall / denom


def macro_f05_from_dicts(
    gt_dict:   dict[str, set[str]],
    pred_dict: dict[str, set[str]],
    s1_ids:    Optional[list[str]] = None,
) -> dict:
    """
    Compute macro F0.5 over a set of Source-1 entity IDs.

    Parameters
    ----------
    gt_dict : dict  {s1_id -> set of matched ids}  (empty set for singletons)
    pred_dict : dict {s1_id -> set of predicted matched ids}
    s1_ids : list   If provided, evaluate only over these IDs.
                    If None, evaluate over union of gt_dict and pred_dict keys.

    Returns
    -------
    dict with keys:
        macro_f05          : float  (the primary metric)
        n_entities         : int
        n_singletons_gt    : int
        singleton_accuracy : float  (fraction of singletons correctly predicted)
        mean_precision     : float
        mean_recall        : float
        per_entity_f05     : dict {s1_id -> float}  (full breakdown)
    """
    if s1_ids is None:
        s1_ids = sorted(set(gt_dict.keys()) | set(pred_dict.keys()))

    per_entity: dict[str, float] = {}
    precisions: list[float] = []
    recalls:    list[float] = []
    singleton_gt_count    = 0
    singleton_correct     = 0

    for sid in s1_ids:
        true_set = gt_dict.get(sid, set())
        pred_set = pred_dict.get(sid, set())

        score = f05_single(true_set, pred_set)
        per_entity[sid] = score

        # Precision / recall breakdown (skip singletons for these averages)
        if len(true_set) == 0:
            singleton_gt_count += 1
            if len(pred_set) == 0:
                singleton_correct += 1
        else:
            if len(pred_set) > 0:
                tp = len(true_set & pred_set)
                precisions.append(tp / len(pred_set))
                recalls.append(tp / len(true_set))
            else:
                precisions.append(0.0)
                recalls.append(0.0)

    macro_f05 = sum(per_entity.values()) / len(per_entity) if per_entity else 0.0
    mean_p    = sum(precisions) / len(precisions) if precisions else 0.0
    mean_r    = sum(recalls)    / len(recalls)    if recalls    else 0.0
    sing_acc  = singleton_correct / singleton_gt_count if singleton_gt_count > 0 else 1.0

    return {
        "macro_f05":          macro_f05,
        "n_entities":         len(s1_ids),
        "n_singletons_gt":    singleton_gt_count,
        "singleton_accuracy": sing_acc,
        "mean_precision":     mean_p,
        "mean_recall":        mean_r,
        "per_entity_f05":     per_entity,
    }


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------

def load_gt_dict(gt_path: str | Path) -> dict[str, set[str]]:
    """
    Load train_ground_truth.tsv into a dict {s1_id -> set of matched ids}.
    Rows with empty matched_entity_ids are treated as singletons (empty set).
    """
    gt: dict[str, set[str]] = {}
    with open(gt_path, encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            s1_id = parts[0].strip()
            if len(parts) < 2 or not parts[1].strip():
                gt[s1_id] = set()
            else:
                gt[s1_id] = set(x.strip() for x in parts[1].split(",") if x.strip())
    return gt


def load_pred_dict(pred_path: str | Path) -> dict[str, set[str]]:
    """
    Load a predictions TSV (matching_results.tsv format) into a dict.
    {s1_id -> set of predicted matched ids}
    """
    pred: dict[str, set[str]] = {}
    with open(pred_path, encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            s1_id = parts[0].strip()
            if len(parts) < 2 or not parts[1].strip():
                pred[s1_id] = set()
            else:
                pred[s1_id] = set(x.strip() for x in parts[1].split(",") if x.strip())
    return pred


def load_s1_ids(s1_path: str | Path) -> list[str]:
    """Load entity_id list from a source1 TSV file."""
    ids = []
    with open(s1_path, encoding="utf-8") as f:
        f.readline()  # skip header
        for line in f:
            ids.append(line.split("\t")[0].strip())
    return ids


def save_predictions(
    pred_dict: dict[str, set[str]],
    s1_ids:    list[str],
    out_path:  str | Path,
) -> None:
    """
    Write predictions to a TSV file in competition format.

    Parameters
    ----------
    pred_dict : {s1_id -> set of matched ids}
    s1_ids    : all source1 entity IDs (in submission order)
    out_path  : path to write matching_results.tsv
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in s1_ids:
            matches = pred_dict.get(sid, set())
            if matches:
                f.write(f"{sid}\t{','.join(sorted(matches))}\n")
            else:
                f.write(f"{sid}\t\n")


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------

def print_evaluation_report(results: dict, title: str = "Evaluation Results") -> None:
    """Print a nicely formatted evaluation report."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  Macro F0.5          : {results['macro_f05']:.6f}")
    print(f"  Entities evaluated  : {results['n_entities']:,}")
    print(f"  Singletons (GT)     : {results['n_singletons_gt']:,}  "
          f"({100*results['n_singletons_gt']/max(1,results['n_entities']):.2f}%)")
    print(f"  Singleton accuracy  : {results['singleton_accuracy']:.4f}")
    print(f"  Mean precision      : {results['mean_precision']:.4f}")
    print(f"  Mean recall         : {results['mean_recall']:.4f}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute macro F0.5 for a prediction file against ground truth."
    )
    parser.add_argument("--pred",  required=True, help="Path to matching_results.tsv (predictions)")
    parser.add_argument("--gt",    required=True, help="Path to train_ground_truth.tsv")
    parser.add_argument("--s1ids", required=False,
                        help="Optional: path to source1 TSV to fix the set of evaluated entities.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-entity scores for lowest-scoring entities.")
    args = parser.parse_args()

    print(f"Loading ground truth from: {args.gt}")
    gt_dict = load_gt_dict(args.gt)

    print(f"Loading predictions from:  {args.pred}")
    pred_dict = load_pred_dict(args.pred)

    s1_ids = None
    if args.s1ids:
        print(f"Restricting to S1 IDs from: {args.s1ids}")
        s1_ids = load_s1_ids(args.s1ids)

    results = macro_f05_from_dicts(gt_dict, pred_dict, s1_ids)
    print_evaluation_report(results)

    if args.verbose:
        # Show 20 worst-scoring entities
        worst = sorted(results["per_entity_f05"].items(), key=lambda x: x[1])[:20]
        print("  20 worst-scoring entities:")
        for sid, score in worst:
            true_cnt = len(gt_dict.get(sid, set()))
            pred_cnt = len(pred_dict.get(sid, set()))
            print(f"    {sid}  F0.5={score:.3f}  GT={true_cnt}  Pred={pred_cnt}")

    return results["macro_f05"]


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
