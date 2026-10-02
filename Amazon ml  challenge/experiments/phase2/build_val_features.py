"""
experiments/phase2/build_val_features.py
========================================
Generates candidate pairs and extracts full 24-feature pairwise matrix
for the 50,000 query validation set (val_sample_50k.parquet).
Saves labeled features to experiments/phase2/val_50k_features.parquet.
"""

import sys
import os
import time
import polars as pl

sys.path.insert(0, os.path.abspath("."))
from src.candidate_generation import generate_candidates, normalize_entities
from experiments.phase2.validation_framework import load_validation_dataset, label_candidate_pairs
from experiments.phase2.feature_engineering import extract_pairwise_features

def main():
    t_start = time.time()
    print("=" * 70, flush=True)
    print("PHASE 2: BUILDING VALIDATION CANDIDATE FEATURE MATRIX (50K QUERIES)", flush=True)
    print("=" * 70, flush=True)

    # 1. Load validation sample and ground truth
    val_path = "experiments/val_sample_50k.parquet"
    print(f"[1/5] Loading validation dataset from {val_path}...", flush=True)
    val_df, gt_dict = load_validation_dataset(val_path)
    total_true_links = sum(len(v) for v in gt_dict.values())
    print(f"      Loaded {len(val_df):,} queries with {total_true_links:,} true links.", flush=True)

    # 2. Load cached normalized targets
    tgt_cache = "experiments/cache/targets_train_normalized.parquet"
    print(f"[2/5] Loading cached normalized targets from {tgt_cache}...", flush=True)
    targets_df = pl.read_parquet(tgt_cache)
    print(f"      Loaded {len(targets_df):,} normalized targets.", flush=True)

    # 3. Generate candidate pairs with channel provenance
    print("[3/5] Running 7-channel candidate generation (top_k=40, return_provenance=True)...", flush=True)
    t_cand = time.time()
    val_norm = normalize_entities(val_df)
    cands_df = generate_candidates(
        queries_df=val_df,
        targets_df=targets_df,
        top_k=40,
        bigram_max_freq=500,
        brand_max_freq=300,
        verbose=True,
        return_provenance=True,
    )
    print(f"      Candidate generation completed in {time.time()-t_cand:.2f}s.", flush=True)
    print(f"      Total candidate pairs: {len(cands_df):,}", flush=True)

    # 4. Label candidate pairs with ground truth
    print("[4/5] Labeling candidate pairs with ground-truth matches...", flush=True)
    cands_labeled = label_candidate_pairs(cands_df, gt_dict)
    n_pos = (cands_labeled["label"] == 1).sum()
    n_neg = (cands_labeled["label"] == 0).sum()
    link_recall = (n_pos / total_true_links) * 100.0
    print(f"      Positive pairs (True Matches captured): {n_pos:,} / {total_true_links:,} ({link_recall:.2f}% Link Recall@40)")
    print(f"      Negative pairs (Hard & Lexical Negatives): {n_neg:,} ({n_neg/n_pos:.1f}:1 class imbalance)")

    # 5. Extract 24 pairwise features
    print("[5/5] Extracting 24 pairwise features...", flush=True)
    t_feat = time.time()
    unique_tgt_ids = cands_labeled["target_id"].unique().to_list()
    print(f"      Filtering {len(targets_df):,} targets to {len(unique_tgt_ids):,} candidate targets...", flush=True)
    targets_subset = targets_df.filter(pl.col("entity_id").is_in(unique_tgt_ids))
    
    features_df, feature_names = extract_pairwise_features(
        cand_df=cands_labeled,
        queries_df=val_norm,
        targets_df=targets_subset,
        verbose=True
    )
    print(f"      Feature extraction finished in {time.time()-t_feat:.2f}s.", flush=True)

    # 6. Save feature matrix to parquet
    out_path = "experiments/phase2/val_50k_features.parquet"
    print(f"Saving feature matrix to {out_path}...", flush=True)
    features_df.write_parquet(out_path)
    file_size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"Saved {len(features_df):,} rows x {len(features_df.columns)} columns ({file_size_mb:.2f} MB).", flush=True)

    print("=" * 70, flush=True)
    print(f"BUILD COMPLETED IN {time.time()-t_start:.1f}s. Validation dataset ready for model training!", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    main()
