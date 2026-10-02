import os, subprocess

# STRATEGY: v7_restored + add bypass matches ONLY for v7-singletons
# v7 had 1.48M matched, ~250K singletons
# For those singletons: try adding high-evidence bypass pairs from v8
# Risk is low: singletons have nothing to lose, bypass might find real matches

SHARDS_V7_DIR = 'Z:/Amazon ML/output/shards'
SHARDS_V8_DIR = 'Z:/Amazon ML/output/shards_v8'
TEMPLATE_FILE  = 'Z:/Amazon ML/dataset/raw/test/test_source1.tsv'
OUT_FILE = 'Z:/Amazon ML/output/matching_results_v11_singleton_boost.tsv'

print('Step 1: Loading v7 shard matches...')
v7_matches = {}
shard_files = sorted([f for f in os.listdir(SHARDS_V7_DIR) if f.endswith('_matches.tsv')])
for sf in shard_files:
    path = os.path.join(SHARDS_V7_DIR, sf)
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split('\t')
            src_id = parts[0]
            all_ids = parts[1].split(',') if len(parts) > 1 and parts[1].strip() else []
            v7_matches[src_id] = all_ids

v7_matched = sum(1 for v in v7_matches.values() if v)
v7_singletons = sum(1 for v in v7_matches.values() if not v)
print(f'v7: {v7_matched:,} matched, {v7_singletons:,} singletons')

print('Step 2: Loading v8 BYPASS-ONLY pairs for singletons...')
# For each v7-singleton, look for bypass (prob=1.0) matches in v8
bypass_for_singletons = {}

for sf in shard_files:
    sf8 = sf  # same naming
    path8 = os.path.join(SHARDS_V8_DIR, sf8)
    if not os.path.exists(path8):
        print(f'  Missing: {path8}')
        continue
    with open(path8, 'r', encoding='utf-8') as f:
        hdr = f.readline()
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split('\t')
            if len(parts) < 2: continue
            src_id = parts[0]
            # ONLY process v7 singletons
            if src_id not in v7_matches or v7_matches[src_id]:
                continue
            matched_ids = parts[1].split(',') if parts[1] else []
            probs = parts[2].split('|') if len(parts) > 2 and parts[2] else []
            bypass_pairs = []
            for i, mid in enumerate(matched_ids):
                if not mid: continue
                try:
                    prob = float(probs[i]) if i < len(probs) else 0.0
                except:
                    prob = 0.0
                # Only take BYPASS entries (1.0) AND clean ML entries >= 0.65
                # For singletons we accept bypass since ML already said "no match"
                if prob >= 1.0 or prob >= 0.65:
                    bypass_pairs.append(mid)
            if bypass_pairs:
                bypass_for_singletons[src_id] = bypass_pairs

print(f'Bypass/ML matches found for {len(bypass_for_singletons):,} v7-singletons')

print('Step 3: Loading source1 IDs and writing output...')
source1_ids = []
with open(TEMPLATE_FILE, 'r', encoding='utf-8') as f:
    f.readline()
    for line in f:
        parts = line.strip().split('\t')
        if parts and parts[0]:
            source1_ids.append(parts[0])

total_matched = 0
total_edges = 0
singleton_boosted = 0

with open(OUT_FILE, 'w', encoding='utf-8') as f:
    for src_id in source1_ids:
        v7_ids = v7_matches.get(src_id, [])
        
        if v7_ids:
            # Has v7 matches - use v7 predictions (trusted)
            s2 = [x for x in v7_ids if x.startswith('S2')][:3]
            s3 = [x for x in v7_ids if x.startswith('S3')][:3]
            total_matched += 1
            total_edges += len(s2) + len(s3)
            f.write(src_id + '\t' + ','.join(s2) + '\t' + ','.join(s3) + '\n')
        elif src_id in bypass_for_singletons:
            # v7 singleton that has bypass/ML matches in v8 - ADD them
            extra_ids = bypass_for_singletons[src_id]
            s2 = [x for x in extra_ids if x.startswith('S2')][:3]
            s3 = [x for x in extra_ids if x.startswith('S3')][:3]
            if s2 or s3:
                total_matched += 1
                total_edges += len(s2) + len(s3)
                singleton_boosted += 1
                f.write(src_id + '\t' + ','.join(s2) + '\t' + ','.join(s3) + '\n')
            else:
                f.write(src_id + '\t\t\n')
        else:
            f.write(src_id + '\t\t\n')

print(f'Total matched: {total_matched:,} / {len(source1_ids):,}')
print(f'  - From v7 ML: {v7_matched:,}')
print(f'  - Singleton boosts: {singleton_boosted:,}')
print(f'Total edges: {total_edges:,}')
print(f'File size: {os.path.getsize(OUT_FILE):,} bytes')

print('\nValidating...')
result = subprocess.run(
    ['python', 'Z:/Amazon ML/src/validate_submission.py',
     '-m', OUT_FILE,
     '-t', 'Z:/Amazon ML/dataset/raw/test'],
    capture_output=True, text=True, timeout=120
)
print('STDOUT:', result.stdout[-3000:])
print('STDERR:', result.stderr[-500:])
print('Return code:', result.returncode)
if result.returncode == 0:
    print('VALIDATION PASSED')
