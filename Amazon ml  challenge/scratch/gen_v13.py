import os

SHARDS_V7_DIR = 'Z:/Amazon ML/output/shards'
SHARDS_V8_DIR = 'Z:/Amazon ML/output/shards_v8'
TEMPLATE_FILE  = 'Z:/Amazon ML/dataset/raw/test/test_source1.tsv'
OUT_FILE = 'Z:/Amazon ML/output/matching_results_v13.tsv'

# Strategy: v12 base (v7 + singleton boosts at 0.60)
# PLUS: for entities with only 1 match in v7, add more from v8 ML >= 0.62
SINGLETON_ML_THRESH = 0.60    # same as v12
UNDER_MATCHED_THRESH = 0.62   # for 1-match entities: add more if ML >= 0.62

print('Step 1: Loading v7 shard matches...')
v7_matches = {}
shard_files = sorted([f for f in os.listdir(SHARDS_V7_DIR) if f.endswith('_matches.tsv')])
for sf in shard_files:
    with open(os.path.join(SHARDS_V7_DIR, sf), 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split('\t')
            src_id = parts[0]
            ids_str = parts[1] if len(parts) > 1 else ''
            v7_matches[src_id] = ids_str

v7_matched = sum(1 for v in v7_matches.values() if v)
v7_singletons = sum(1 for v in v7_matches.values() if not v)
one_match = sum(1 for v in v7_matches.values() if v and len(v.split(',')) == 1)
print(f'v7: {v7_matched:,} matched, {v7_singletons:,} singletons, {one_match:,} with exactly 1 match')

print('Step 2: Loading v8 ML extras for singletons AND 1-match entities...')
ml_extras = {}  # src_id -> list of (mid, prob)

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
            v7_ids_str = v7_matches.get(src_id, '')
            v7_count = len(v7_ids_str.split(',')) if v7_ids_str else 0

            # Process: singletons (v7_count==0) or under-matched (v7_count==1)
            if v7_count == 0:
                thresh = SINGLETON_ML_THRESH
            elif v7_count == 1:
                thresh = UNDER_MATCHED_THRESH
            else:
                continue  # skip well-matched entities

            matched_ids = parts[1].split(',') if parts[1] else []
            probs_str = parts[2].split('|') if len(parts) > 2 and parts[2] else []
            good_pairs = []
            for i, mid in enumerate(matched_ids):
                if not mid: continue
                # skip if already in v7
                if v7_ids_str and mid in v7_ids_str:
                    continue
                try: prob = float(probs_str[i]) if i < len(probs_str) else 0.0
                except: prob = 0.0
                if thresh <= prob < 1.0:
                    good_pairs.append((mid, prob))
            if good_pairs:
                good_pairs.sort(key=lambda x: -x[1])
                ml_extras[src_id] = [m for m,_ in good_pairs[:4]]

singletons_boosted = sum(1 for sid,_ in ml_extras.items() if not v7_matches.get(sid,''))
under_boosted = sum(1 for sid,_ in ml_extras.items() if v7_matches.get(sid,'') and len(v7_matches[sid].split(','))==1)
print(f'Extras found: {len(ml_extras):,} ({singletons_boosted:,} singletons + {under_boosted:,} under-matched)')

print('Step 3: Writing v13...')
source1_ids = []
with open(TEMPLATE_FILE, 'r', encoding='utf-8') as f:
    f.readline()
    for line in f:
        p = line.strip().split('\t')
        if p and p[0]: source1_ids.append(p[0])

total_matched = 0
with open(OUT_FILE, 'w', encoding='utf-8') as f:
    f.write('source1_entity_id\tmatched_entity_ids\n')
    for src_id in source1_ids:
        v7_ids_str = v7_matches.get(src_id, '')
        extras = ml_extras.get(src_id, [])
        
        if v7_ids_str:
            if extras:
                # Combine: v7 ids + new extras (dedup)
                existing = set(v7_ids_str.split(','))
                new_ids = [m for m in extras if m not in existing][:3]
                combined = v7_ids_str + (',' + ','.join(new_ids) if new_ids else '')
                f.write(src_id + '\t' + combined + '\n')
            else:
                f.write(src_id + '\t' + v7_ids_str + '\n')
            total_matched += 1
        elif extras:
            f.write(src_id + '\t' + ','.join(extras) + '\n')
            total_matched += 1
        else:
            f.write(src_id + '\t\n')

print(f'Total matched: {total_matched:,} / {len(source1_ids):,}')
print(f'File: {OUT_FILE}')
print(f'Size: {os.path.getsize(OUT_FILE):,} bytes ({os.path.getsize(OUT_FILE)/1024/1024:.1f} MB)')

print('=== First 4 lines ===')
with open(OUT_FILE,'r',encoding='utf-8') as f:
    for i,line in enumerate(f):
        print(repr(line[:120]))
        if i>=3: break
print('DONE - v13 ready')
