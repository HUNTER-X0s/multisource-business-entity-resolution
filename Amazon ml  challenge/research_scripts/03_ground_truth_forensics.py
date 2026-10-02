#!/usr/bin/env python3
"""
Step 3: Ground Truth Forensics & Linkage Cardinality Analysis
Analyzes train_ground_truth.tsv in relation to train_source1, train_source2, train_source3.
Measures singleton rate, multi-match rate, match distributions, source-breakdown,
and integrity of all entity references.
"""

import os
import json
from collections import Counter

def load_ids_from_tsv(path):
    print(f"Loading IDs from {path}...")
    ids = set()
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        next(f, None)
        for line in f:
            if not line.strip():
                continue
            parts = line.split("\t", 1)
            ids.add(parts[0].strip())
    return ids

def main():
    gt_path = "dataset/raw/train/train_ground_truth.tsv"
    s1_path = "dataset/raw/train/train_source1.tsv"
    s2_path = "dataset/raw/train/train_source2.tsv"
    s3_path = "dataset/raw/train/train_source3.tsv"

    print("Step 3: Ground Truth Forensics Analysis...")

    s1_ids = load_ids_from_tsv(s1_path)
    s2_ids = load_ids_from_tsv(s2_path)
    s3_ids = load_ids_from_tsv(s3_path)
    print(f"Loaded: S1={len(s1_ids):,} IDs, S2={len(s2_ids):,} IDs, S3={len(s3_ids):,} IDs")

    total_gt_rows = 0
    gt_s1_ids = set()
    singletons = 0
    multi_matches = 0
    exact_one_match = 0
    
    match_counts = Counter()
    source_breakdown = Counter() # 'singleton', 's2_only', 's3_only', 'both_s2_s3'
    
    s2_match_counts = Counter()
    s3_match_counts = Counter()
    
    missing_s1_in_source1 = set()
    invalid_s2_ids = set()
    invalid_s3_ids = set()
    self_matches = set()
    unknown_prefix_ids = set()
    intra_list_dupes = 0
    
    total_matched_links = 0
    total_s2_links = 0
    total_s3_links = 0

    all_match_lengths = []

    print(f"Parsing ground truth {gt_path}...")
    with open(gt_path, "r", encoding="utf-8", errors="replace") as f:
        header = f.readline().rstrip("\r\n").split("\t")
        for line_num, line in enumerate(f, start=2):
            total_gt_rows += 1
            parts = line.rstrip("\r\n").split("\t")
            s1_id = parts[0].strip()
            gt_s1_ids.add(s1_id)
            
            if s1_id not in s1_ids:
                missing_s1_in_source1.add(s1_id)
                
            raw_matches = parts[1].strip() if len(parts) > 1 else ""
            if not raw_matches:
                match_list = []
            else:
                match_list = [m.strip() for m in raw_matches.split(",") if m.strip()]
                
            n_matches = len(match_list)
            all_match_lengths.append(n_matches)
            match_counts[n_matches] += 1
            
            if len(match_list) != len(set(match_list)):
                intra_list_dupes += 1
                
            s2_in_list = 0
            s3_in_list = 0
            
            for mid in match_list:
                total_matched_links += 1
                if mid.startswith("S2-"):
                    s2_in_list += 1
                    total_s2_links += 1
                    if mid not in s2_ids:
                        invalid_s2_ids.add(mid)
                elif mid.startswith("S3-"):
                    s3_in_list += 1
                    total_s3_links += 1
                    if mid not in s3_ids:
                        invalid_s3_ids.add(mid)
                elif mid.startswith("S1-"):
                    self_matches.add(mid)
                else:
                    unknown_prefix_ids.add(mid)
                    
            s2_match_counts[s2_in_list] += 1
            s3_match_counts[s3_in_list] += 1
            
            if n_matches == 0:
                singletons += 1
                source_breakdown["singleton"] += 1
            elif n_matches == 1:
                exact_one_match += 1
                if s2_in_list == 1:
                    source_breakdown["s2_only"] += 1
                elif s3_in_list == 1:
                    source_breakdown["s3_only"] += 1
            else:
                multi_matches += 1
                if s2_in_list > 0 and s3_in_list > 0:
                    source_breakdown["both_s2_s3"] += 1
                elif s2_in_list > 0:
                    source_breakdown["s2_only"] += 1
                elif s3_in_list > 0:
                    source_breakdown["s3_only"] += 1

    all_match_lengths.sort()
    n_total = len(all_match_lengths)
    
    # Check if any S1 IDs from train_source1 are missing from ground truth
    s1_missing_from_gt = s1_ids - gt_s1_ids

    summary = {
        "ground_truth_file": gt_path,
        "total_gt_rows": total_gt_rows,
        "total_s1_records_in_source1": len(s1_ids),
        "s1_coverage_complete": (len(s1_missing_from_gt) == 0 and total_gt_rows == len(s1_ids)),
        "s1_missing_from_gt_count": len(s1_missing_from_gt),
        "gt_s1_missing_from_source1_count": len(missing_s1_in_source1),
        
        "singletons_count": singletons,
        "singleton_rate": round(singletons / total_gt_rows, 5) if total_gt_rows else 0,
        
        "exact_one_match_count": exact_one_match,
        "exact_one_match_rate": round(exact_one_match / total_gt_rows, 5) if total_gt_rows else 0,
        
        "multi_matches_count": multi_matches,
        "multi_match_rate": round(multi_matches / total_gt_rows, 5) if total_gt_rows else 0,
        
        "total_matched_links": total_matched_links,
        "total_s2_links": total_s2_links,
        "total_s3_links": total_s3_links,
        
        "match_distribution": {
            "min": all_match_lengths[0],
            "max": all_match_lengths[-1],
            "mean": round(sum(all_match_lengths) / n_total, 4),
            "median": all_match_lengths[int(n_total * 0.5)],
            "p75": all_match_lengths[int(n_total * 0.75)],
            "p95": all_match_lengths[int(n_total * 0.95)],
            "p99": all_match_lengths[min(int(n_total * 0.99), n_total - 1)],
        },
        
        "match_count_histogram": {str(k): match_counts[k] for k in sorted(match_counts.keys())},
        "source_breakdown": dict(source_breakdown),
        
        "integrity_checks": {
            "intra_list_dupes": intra_list_dupes,
            "self_matches_count": len(self_matches),
            "unknown_prefix_count": len(unknown_prefix_ids),
            "invalid_s2_ids_count": len(invalid_s2_ids),
            "invalid_s3_ids_count": len(invalid_s3_ids),
        }
    }

    out_path = "research_scripts/03_ground_truth_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nGround Truth analysis complete. Results saved to {out_path}")
    print(f"Total S1 entities: {total_gt_rows:,}")
    print(f"Singletons: {singletons:,} ({summary['singleton_rate']*100:.2f}%)")
    print(f"1-Match: {exact_one_match:,} ({summary['exact_one_match_rate']*100:.2f}%)")
    print(f"Multi-Match: {multi_matches:,} ({summary['multi_match_rate']*100:.2f}%)")
    print(f"Total Links: S2={total_s2_links:,}, S3={total_s3_links:,}, Total={total_matched_links:,}")
    print(f"Source breakdown: {dict(source_breakdown)}")

if __name__ == "__main__":
    main()
