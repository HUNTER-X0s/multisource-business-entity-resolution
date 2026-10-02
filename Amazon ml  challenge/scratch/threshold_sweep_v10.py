"""
THRESHOLD SWEEP v10 - Find optimal threshold from v8 shard data (NO bypass noise)
"""
import os, sys, time, subprocess

SHARDS_DIR = "Z:/Amazon ML/output/shards_v8"
OUTPUT_DIR = "Z:/Amazon ML/output"
TEMPLATE_FILE = "Z:/Amazon ML/dataset/test/test_source1.tsv"

print("Loading v8 shard match data with confidence scores...")
all_matches = {}

shard_files = sorted([f for f in os.listdir(SHARDS_DIR) if f.endswith("_matches.tsv")])
print(f"Found {len(shard_files)} shard files")

for sf in shard_files:
    path = os.path.join(SHARDS_DIR, sf)
    with open(path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            src_id = parts[0]
            matched_ids = parts[1].split(",") if parts[1] else []
            probs = parts[2].split("|") if len(parts) > 2 and parts[2] else []
            pairs = []
            for i, mid in enumerate(matched_ids):
                if not mid:
                    continue
                try:
                    prob = float(probs[i]) if i < len(probs) else 0.0
                except:
                    prob = 0.0
                pairs.append((mid, prob))
            if src_id not in all_matches:
                all_matches[src_id] = []
            all_matches[src_id].extend(pairs)

print(f"Loaded {len(all_matches)} source1 entities")

# Distribution check
all_confs = []
for src_id, pairs in all_matches.items():
    for mid, prob in pairs:
        all_confs.append(prob)
all_confs.sort(reverse=True)
print(f"Total candidate pairs: {len(all_confs):,}")
print(f"Bypass (==1.0) entries: {sum(1 for c in all_confs if c >= 1.0):,}")
print("")
for t in [0.50, 0.55, 0.60, 0.62, 0.63, 0.64, 0.65, 0.66]:
    kept = sum(1 for c in all_confs if t <= c < 1.0)
    print(f"  threshold={t:.2f}: kept={kept:>8,} clean pairs")

# Load source1 IDs
print("\nLoading test source1 IDs...")
source1_ids = []
with open(TEMPLATE_FILE, "r", encoding="utf-8") as f:
    f.readline()
    for line in f:
        parts = line.strip().split("\t")
        if parts:
            source1_ids.append(parts[0])
print(f"Total source1: {len(source1_ids):,}")

# Generate at threshold 0.65 (v7-equivalent but clean - no bypass)
TARGET_THRESHOLD = 0.65
print(f"\n=== Generating submission at threshold {TARGET_THRESHOLD} (bypass excluded) ===")

out_path = os.path.join(OUTPUT_DIR, "matching_results_v10_clean65.tsv")
total_matched = 0
total_edges = 0

with open(out_path, "w", encoding="utf-8") as f:
    for src_id in source1_ids:
        pairs = all_matches.get(src_id, [])
        # Exclude bypass 1.0 and apply threshold
        filtered = [(mid, prob) for mid, prob in pairs if TARGET_THRESHOLD <= prob < 1.0]
        if not filtered:
            f.write(f"{src_id}\t\t\n")
            continue
        best = {}
        for mid, prob in filtered:
            if mid not in best or prob > best[mid]:
                best[mid] = prob
        sorted_matches = sorted(best.items(), key=lambda x: -x[1])[:6]
        s2_ids = [mid for mid, _ in sorted_matches if mid.startswith("S2")][:3]
        s3_ids = [mid for mid, _ in sorted_matches if mid.startswith("S3")][:3]
        total_matched += 1
        total_edges += len(s2_ids) + len(s3_ids)
        f.write(f"{src_id}\t{','.join(s2_ids)}\t{','.join(s3_ids)}\n")

print(f"Matched: {total_matched:,} / {len(source1_ids):,} ({100*total_matched/len(source1_ids):.1f}%)")
print(f"Total edges: {total_edges:,}")
print(f"File: {out_path}")
print(f"Size: {os.path.getsize(out_path):,} bytes")

print("\nValidating...")
result = subprocess.run(
    ["python", "Z:/Amazon ML/src/validate_submission.py",
     "--submission", out_path,
     "--candidates", "Z:/Amazon ML/output/candidate_pairs.tsv"],
    capture_output=True, text=True, timeout=120
)
print("STDOUT:", result.stdout[-2000:])
print("STDERR:", result.stderr[-500:])
print("Return code:", result.returncode)
