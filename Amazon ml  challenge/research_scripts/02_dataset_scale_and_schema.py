#!/usr/bin/env python3
"""
Step 2: Dataset Scale, Schema, Field Distributions, and Cartesian Space Analysis
Computes full dataset statistics across all source and test files without downsampling.
Streams lines to preserve memory.
"""

import os
import json
import re
from collections import Counter
import math

TSV_FILES = [
    ("dataset/raw/train/train_source1.tsv", "train_source1", "train"),
    ("dataset/raw/train/train_source2.tsv", "train_source2", "train"),
    ("dataset/raw/train/train_source3.tsv", "train_source3", "train"),
    ("dataset/raw/test/test_source1.tsv", "test_source1", "test"),
    ("dataset/raw/test/test_source2.tsv", "test_source2", "test"),
    ("dataset/raw/test/test_source3.tsv", "test_source3", "test"),
]

def analyze_tsv(file_path, file_id, split):
    print(f"Analyzing {file_path}...")
    abs_path = os.path.abspath(file_path)
    
    total_rows = 0
    null_counts = {"entity_id": 0, "business_name": 0, "business_address": 0, "country": 0}
    empty_counts = {"entity_id": 0, "business_name": 0, "business_address": 0, "country": 0}
    whitespace_counts = {"entity_id": 0, "business_name": 0, "business_address": 0, "country": 0}
    
    country_counts = Counter()
    
    name_lengths = []
    addr_lengths = []
    name_token_counts = []
    addr_token_counts = []
    
    id_prefixes = Counter()
    id_numbers = []
    duplicate_ids = 0
    seen_ids = set()
    
    # Track character set anomalies (e.g. non-ascii)
    non_ascii_names = 0
    non_ascii_addrs = 0
    
    with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
        header_line = f.readline().rstrip("\r\n")
        header = header_line.split("\t")
        col_to_idx = {col: idx for idx, col in enumerate(header)}
        
        for line_num, line in enumerate(f, start=2):
            total_rows += 1
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 4:
                parts += [""] * (4 - len(parts))
                
            eid = parts[col_to_idx.get("entity_id", 0)].strip()
            name = parts[col_to_idx.get("business_name", 1)].strip()
            addr = parts[col_to_idx.get("business_address", 2)].strip()
            country = parts[col_to_idx.get("country", 3)].strip()
            
            # ID checks
            if eid in seen_ids:
                duplicate_ids += 1
            seen_ids.add(eid)
            
            if "-" in eid:
                pfx, num_str = eid.split("-", 1)
                id_prefixes[pfx] += 1
                if num_str.isdigit():
                    id_numbers.append(int(num_str))
            else:
                id_prefixes["NO_HYPHEN"] += 1
                
            # Missing checks
            for field_name, val in [("entity_id", eid), ("business_name", name), ("business_address", addr), ("country", country)]:
                if not val:
                    empty_counts[field_name] += 1
                elif val.isspace():
                    whitespace_counts[field_name] += 1
                    
            country_counts[country if country else "EMPTY"] += 1
            
            # Length and token stats
            name_len = len(name)
            addr_len = len(addr)
            name_tokens = len(name.split()) if name else 0
            addr_tokens = len(addr.split()) if addr else 0
            
            name_lengths.append(name_len)
            addr_lengths.append(addr_len)
            name_token_counts.append(name_tokens)
            addr_token_counts.append(addr_tokens)
            
            if any(ord(c) > 127 for c in name):
                non_ascii_names += 1
            if any(ord(c) > 127 for c in addr):
                non_ascii_addrs += 1

    def compute_distribution(vals):
        if not vals:
            return {}
        s_vals = sorted(vals)
        n = len(s_vals)
        return {
            "min": s_vals[0],
            "max": s_vals[-1],
            "mean": round(sum(s_vals) / n, 2),
            "p25": s_vals[int(n * 0.25)],
            "median": s_vals[int(n * 0.50)],
            "p75": s_vals[int(n * 0.75)],
            "p95": s_vals[int(n * 0.95)],
            "p99": s_vals[min(int(n * 0.99), n - 1)],
        }

    id_num_stats = compute_distribution(id_numbers) if id_numbers else {}
    
    return {
        "file": file_path,
        "file_id": file_id,
        "split": split,
        "total_records": total_rows,
        "unique_entity_ids": len(seen_ids),
        "duplicate_ids": duplicate_ids,
        "id_prefixes": dict(id_prefixes),
        "id_num_min": id_num_stats.get("min"),
        "id_num_max": id_num_stats.get("max"),
        "id_num_count": len(id_numbers),
        "is_strictly_sequential": (id_num_stats.get("max") == len(id_numbers) and id_num_stats.get("min") == 1 and duplicate_ids == 0) if id_numbers else False,
        "empty_counts": empty_counts,
        "whitespace_counts": whitespace_counts,
        "country_counts": dict(country_counts),
        "non_ascii_name_pct": round(100 * non_ascii_names / total_rows, 2) if total_rows else 0,
        "non_ascii_addr_pct": round(100 * non_ascii_addrs / total_rows, 2) if total_rows else 0,
        "name_char_len": compute_distribution(name_lengths),
        "addr_char_len": compute_distribution(addr_lengths),
        "name_token_count": compute_distribution(name_token_counts),
        "addr_token_count": compute_distribution(addr_token_counts),
    }

def main():
    print("Starting Deep Dataset Scale & Schema Forensics...")
    results = {}
    for path, fid, split in TSV_FILES:
        res = analyze_tsv(path, fid, split)
        results[fid] = res
        print(f"  {fid}: {res['total_records']} rows, Countries: {res['country_counts']}")

    # Cartesian product calculations
    s1_train = results["train_source1"]["total_records"]
    s2_train = results["train_source2"]["total_records"]
    s3_train = results["train_source3"]["total_records"]
    
    s1_test = results["test_source1"]["total_records"]
    s2_test = results["test_source2"]["total_records"]
    s3_test = results["test_source3"]["total_records"]

    cartesian = {
        "train": {
            "S1_count": s1_train,
            "S2_count": s2_train,
            "S3_count": s3_train,
            "S1_x_S2_pairs": s1_train * s2_train,
            "S1_x_S3_pairs": s1_train * s3_train,
            "total_train_pairs": (s1_train * s2_train) + (s1_train * s3_train),
        },
        "test": {
            "S1_count": s1_test,
            "S2_count": s2_test,
            "S3_count": s3_test,
            "S1_x_S2_pairs": s1_test * s2_test,
            "S1_x_S3_pairs": s1_test * s3_test,
            "total_test_pairs": (s1_test * s2_test) + (s1_test * s3_test),
        },
        "theoretical_ram_gigabytes": {
            "train_at_64_bytes_per_pair": round(((s1_train * s2_train) + (s1_train * s3_train)) * 64 / (1024**3), 2),
            "test_at_64_bytes_per_pair": round(((s1_test * s2_test) + (s1_test * s3_test)) * 64 / (1024**3), 2),
        }
    }
    
    output = {
        "file_statistics": results,
        "cartesian_analysis": cartesian
    }
    
    out_path = "research_scripts/02_scale_and_schema_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved scale and schema results to {out_path}")
    print("Cartesian search space:")
    print(f"  Train: {cartesian['train']['total_train_pairs']:,} pairs ({cartesian['theoretical_ram_gigabytes']['train_at_64_bytes_per_pair']:,} GB)")
    print(f"  Test:  {cartesian['test']['total_test_pairs']:,} pairs ({cartesian['theoretical_ram_gigabytes']['test_at_64_bytes_per_pair']:,} GB)")

if __name__ == "__main__":
    main()
