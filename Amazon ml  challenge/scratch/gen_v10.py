import os, subprocess

SHARDS_DIR = 'Z:/Amazon ML/output/shards_v8'
OUTPUT_DIR = 'Z:/Amazon ML/output'
TEMPLATE_FILE = 'Z:/Amazon ML/dataset/raw/test/test_source1.tsv'
TARGET_THRESHOLD = 0.65

print('Loading v8 shard data (bypass-excluded)...')
all_matches = {}

shard_files = sorted([f for f in os.listdir(SHARDS_DIR) if f.endswith('_matches.tsv')])
for sf in shard_files:
    path = os.path.join(SHARDS_DIR, sf)
    with open(path, 'r', encoding='utf-8') as f:
        f.readline()
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split('\t')
            if len(parts) < 2: continue
            src_id = parts[0]
            matched_ids = parts[1].split(',') if parts[1] else []
            probs = parts[2].split('|') if len(parts) > 2 and parts[2] else []
            pairs = []
            for i, mid in enumerate(matched_ids):
                if not mid: continue
                try:
                    prob = float(probs[i]) if i < len(probs) else 0.0
                except:
                    prob = 0.0
                pairs.append((mid, prob))
            if src_id not in all_matches:
                all_matches[src_id] = []
            all_matches[src_id].extend(pairs)

print('Loaded', len(all_matches), 'entities')

source1_ids = []
with open(TEMPLATE_FILE, 'r', encoding='utf-8') as f:
    f.readline()
    for line in f:
        parts = line.strip().split('\t')
        if parts and parts[0]:
            source1_ids.append(parts[0])
print('Test source1 IDs:', len(source1_ids))

out_path = os.path.join(OUTPUT_DIR, 'matching_results_v10_clean65.tsv')
total_matched = 0
total_edges = 0

with open(out_path, 'w', encoding='utf-8') as f:
    for src_id in source1_ids:
        pairs = all_matches.get(src_id, [])
        filtered = [(mid, prob) for mid, prob in pairs if TARGET_THRESHOLD <= prob < 1.0]
        if not filtered:
            f.write(src_id + '\t\t\n')
            continue
        best = {}
        for mid, prob in filtered:
            if mid not in best or prob > best[mid]:
                best[mid] = prob
        sorted_matches = sorted(best.items(), key=lambda x: -x[1])[:6]
        s2_ids = [mid for mid, _ in sorted_matches if mid.startswith('S2')][:3]
        s3_ids = [mid for mid, _ in sorted_matches if mid.startswith('S3')][:3]
        total_matched += 1
        total_edges += len(s2_ids) + len(s3_ids)
        f.write(src_id + '\t' + ','.join(s2_ids) + '\t' + ','.join(s3_ids) + '\n')

print('Matched:', total_matched, '/', len(source1_ids))
print('Total edges:', total_edges)
print('File size:', os.path.getsize(out_path))

print('Validating...')
result = subprocess.run(
    ['python', 'Z:/Amazon ML/src/validate_submission.py',
     '--submission', out_path,
     '--candidates', 'Z:/Amazon ML/output/candidate_pairs.tsv'],
    capture_output=True, text=True, timeout=120
)
print('STDOUT:', result.stdout[-2000:])
print('STDERR:', result.stderr[-500:])
print('Return code:', result.returncode)
