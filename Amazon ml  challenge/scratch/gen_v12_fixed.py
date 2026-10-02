import os

SHARDS_V7_DIR = 'Z:/Amazon ML/output/shards'
SHARDS_V8_DIR = 'Z:/Amazon ML/output/shards_v8'
TEMPLATE_FILE  = 'Z:/Amazon ML/dataset/raw/test/test_source1.tsv'
OUT_FILE = 'Z:/Amazon ML/output/matching_results_v12_correct.tsv'
ML_THRESHOLD = 0.60

print('Step 1: Loading v7 shard matches (2-col format)...')
v7_matches = {}
shard_files = sorted([f for f in os.listdir(SHARDS_V7_DIR) if f.endswith('_matches.tsv')])
for sf in shard_files:
    with open(os.path.join(SHARDS_V7_DIR, sf), 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split('\t')
            src_id = parts[0]
            all_ids = parts[1] if len(parts) > 1 else ''
            v7_matches[src_id] = all_ids  # keep raw comma-sep string

v7_matched = sum(1 for v in v7_matches.values() if v)
v7_singletons = sum(1 for v in v7_matches.values() if not v)
print(f'v7: {v7_matched:,} matched, {v7_singletons:,} singletons')

print(f'Step 2: Loading v8 ML-only (>={ML_THRESHOLD}, no bypass) for singletons...')
ml_for_singletons = {}
for sf in shard_files:
    path8 = os.path.join(SHARDS_V8_DIR, sf)
    if not os.path.exists(path8): continue
    with open(path8, 'r', encoding='utf-8') as f:
        f.readline()
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split('\t')
            if len(parts) < 2: continue
            src_id = parts[0]
            if src_id not in v7_matches or v7_matches[src_id]:
                continue  # only singletons
            matched_ids = parts[1].split(',') if parts[1] else []
            probs_str = parts[2].split('|') if len(parts) > 2 and parts[2] else []
            good_pairs = []
            for i, mid in enumerate(matched_ids):
                if not mid: continue
                try: prob = float(probs_str[i]) if i < len(probs_str) else 0.0
                except: prob = 0.0
                if ML_THRESHOLD <= prob < 1.0:
                    good_pairs.append((mid, prob))
            if good_pairs:
                good_pairs.sort(key=lambda x: -x[1])
                ml_for_singletons[src_id] = [m for m,_ in good_pairs[:6]]

print(f'ML matches found for {len(ml_for_singletons):,} singletons')

print('Step 3: Writing CORRECT 2-column format with header...')
source1_ids = []
with open(TEMPLATE_FILE, 'r', encoding='utf-8') as f:
    f.readline()
    for line in f:
        p = line.strip().split('\t')
        if p and p[0]: source1_ids.append(p[0])

total_matched = 0
singleton_boosted = 0

with open(OUT_FILE, 'w', encoding='utf-8') as f:
    f.write('source1_entity_id\tmatched_entity_ids\n')  # REQUIRED HEADER
    for src_id in source1_ids:
        v7_ids_str = v7_matches.get(src_id, '')
        if v7_ids_str:
            f.write(src_id + '\t' + v7_ids_str + '\n')
            total_matched += 1
        elif src_id in ml_for_singletons:
            ids_str = ','.join(ml_for_singletons[src_id])
            f.write(src_id + '\t' + ids_str + '\n')
            total_matched += 1
            singleton_boosted += 1
        else:
            f.write(src_id + '\t\n')

print(f'Total matched: {total_matched:,} / {len(source1_ids):,}')
print(f'  v7 base:          {v7_matched:,}')
print(f'  Singleton boosts: {singleton_boosted:,}')
print(f'File: {OUT_FILE}')
print(f'Size: {os.path.getsize(OUT_FILE):,} bytes')
print()
print('=== First 4 lines ===')
with open(OUT_FILE,'r',encoding='utf-8') as f:
    for i,line in enumerate(f):
        print(repr(line[:120]))
        if i>=3: break
print('DONE')
