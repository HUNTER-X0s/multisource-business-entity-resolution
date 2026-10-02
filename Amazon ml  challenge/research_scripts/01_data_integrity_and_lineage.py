#!/usr/bin/env python3
"""
Step 1: Data Integrity and Lineage Analysis
Computes SHA-256 hashes, exact file sizes in bytes, row counts, columns,
delimiters, encoding, and line-level integrity for all datasets and files.
"""

import os
import hashlib
import json
import time

FILES_TO_CHECK = [
    ("dataset/raw/train/train_source1.tsv", "Train Source 1"),
    ("dataset/raw/train/train_source2.tsv", "Train Source 2"),
    ("dataset/raw/train/train_source3.tsv", "Train Source 3"),
    ("dataset/raw/train/train_ground_truth.tsv", "Train Ground Truth"),
    ("dataset/raw/test/test_source1.tsv", "Test Source 1"),
    ("dataset/raw/test/test_source2.tsv", "Test Source 2"),
    ("dataset/raw/test/test_source3.tsv", "Test Source 3"),
    ("docs/README.md", "README"),
    ("docs/Documentation_template.md", "Doc Template"),
    ("src/validate_submission.py", "Validator Utility"),
]

def compute_file_stats(rel_path, label):
    abs_path = os.path.abspath(rel_path)
    if not os.path.exists(abs_path):
        return {"file": rel_path, "status": "NOT FOUND"}

    file_size = os.path.getsize(abs_path)
    sha256 = hashlib.sha256()

    total_lines = 0
    header = None
    delimiter = None
    malformed_rows = 0
    encoding = "utf-8"

    t0 = time.time()
    try:
        with open(abs_path, "rb") as f:
            while chunk := f.read(1024 * 1024 * 8):
                sha256.update(chunk)
        digest = sha256.hexdigest()
    except Exception as e:
        digest = f"ERROR: {e}"

    # Read line-by-line in text mode to count lines and inspect structure
    if rel_path.endswith(".tsv"):
        expected_cols = 4 if "ground_truth" not in rel_path else 2
        with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
            first_line = f.readline()
            if first_line:
                header = first_line.rstrip("\r\n").split("\t")
                delimiter = "\t" if "\t" in first_line else ("comma" if "," in first_line else "unknown")
                total_lines = 1
                for line_idx, line in enumerate(f, start=2):
                    total_lines += 1
                    parts = line.rstrip("\r\n").split("\t")
                    if len(parts) != expected_cols:
                        malformed_rows += 1
    else:
        with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
            total_lines = sum(1 for _ in f)

    t_elapsed = round(time.time() - t0, 2)
    return {
        "file": rel_path,
        "label": label,
        "size_bytes": file_size,
        "size_mb": round(file_size / (1024 * 1024), 2),
        "total_lines": total_lines,
        "data_rows": total_lines - 1 if header else total_lines,
        "header": header,
        "delimiter": delimiter,
        "malformed_rows": malformed_rows,
        "sha256": digest,
        "time_seconds": t_elapsed
    }

def main():
    print("Starting Data Integrity & Lineage Forensic Analysis...")
    results = []
    for rel_path, label in FILES_TO_CHECK:
        print(f"Profiling {rel_path} ({label})...")
        stat = compute_file_stats(rel_path, label)
        results.append(stat)
        print(f"  Done in {stat.get('time_seconds', 0)}s: {stat.get('size_mb')} MB, {stat.get('data_rows')} rows, SHA-256: {stat.get('sha256')[:12]}...")

    # Write output to JSON
    out_path = "research_scripts/01_data_integrity_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll results saved to {out_path}")

if __name__ == "__main__":
    main()
