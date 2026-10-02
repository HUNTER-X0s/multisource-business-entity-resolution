import os

SHARDS_V7_DIR = 'Z:/Team X_submission/output/shards'
SHARDS_V8_DIR = 'Z:/Team X_submission/output/shards_v8'
TEMPLATE_FILE = 'Z:/Team X_submission/dataset/raw/test/test_source1.tsv'
OUT_FILE      = 'Z:/Team X_submission/output/matching_results_v18.tsv'
T1_THRESH = 0.55
T2_THRESH = 0.65

print('Step 1: Loading v7 shard matches (exact unchanged baseline)...')
v7_matches = {}
shard_files = sorted([f for f in os.listdir(SHARDS_V7_DIR) if f.endswith('_matches.tsv')])
for sf in shard_files:
    with open(os.path.join(SHARDS_V7_DIR, sf), 'r', encoding='utf-8') as sf_in:
        for line in sf_in:
            line = line.strip()
            if not line: continue
            parts = line.split('\t')
            src_id = parts[0]
            ids_str = parts[1] if len(parts) > 1 else ''
            v7_matches[src_id] = ids_str

v7_matched = sum(1 for v in v7_matches.values() if v)
v7_singletons = sum(1 for v in v7_matches.values() if not v)
print(f'v7 base: {v7_matched:,} matched, {v7_singletons:,} singletons')

print('Step 2: Processing singletons with Tiered Precision Filtering...')
ml_for_singletons = {}

for sf in shard_files:
    path8 = os.path.join(SHARDS_V8_DIR, sf)
    if not os.path.exists(path8): continue
    with open(path8, 'r', encoding='utf-8') as sf_in:
        sf_in.readline()
        for line in sf_in:
            line = line.strip()
            if not line: continue
            parts = line.split('\t')
            if len(parts) < 2: continue
            src_id = parts[0]
            if src_id not in v7_matches or v7_matches[src_id]:
                continue  # ONLY boost singletons! Never touch v7 matches.
            
            mids = parts[1].split(',') if parts[1] else []
            pbs = parts[2].split('|') if len(parts) > 2 and parts[2] else []
            
            cands = []
            for i, mid in enumerate(mids):
                if not mid: continue
                try: p = float(pbs[i]) if i < len(pbs) else 0.0
                except: p = 0.0
                if p < 1.0:
                    cands.append((mid, p))
            
            if cands:
                cands.sort(key=lambda x: -x[1])
                kept = []
                for i, (m, p) in enumerate(cands):
                    if i == 0 and p >= T1_THRESH:
                        kept.append(m)
                    elif i > 0 and p >= T2_THRESH:
                        kept.append(m)
                if kept:
                    ml_for_singletons[src_id] = kept[:6]

print(f'Tiered ML matches added for {len(ml_for_singletons):,} singletons')

print('Step 3: Loading template source1 IDs and writing output...')
source1_ids = []
with open(TEMPLATE_FILE, 'r', encoding='utf-8') as f_tpl:
    f_tpl.readline()
    for line in f_tpl:
        p = line.strip().split('\t')
        if p and p[0]: source1_ids.append(p[0])

total_matched = 0
singleton_boosted = 0

with open(OUT_FILE, 'w', encoding='utf-8') as f_out:
    f_out.write('source1_entity_id\tmatched_entity_ids\n')
    for src_id in source1_ids:
        v7_ids_str = v7_matches.get(src_id, '')
        if v7_ids_str:
            f_out.write(src_id + '\t' + v7_ids_str + '\n')
            total_matched += 1
        elif src_id in ml_for_singletons:
            ids_str = ','.join(ml_for_singletons[src_id])
            f_out.write(src_id + '\t' + ids_str + '\n')
            total_matched += 1
            singleton_boosted += 1
        else:
            f_out.write(src_id + '\t\n')

print(f'Total matched: {total_matched:,} / {len(source1_ids):,}')
print(f'  v7 base:          {v7_matched:,}')
print(f'  Singleton boosts: {singleton_boosted:,}')
print(f'Output: {OUT_FILE}')
print(f'File size: {os.path.getsize(OUT_FILE):,} bytes')
print('=== First 4 lines ===')
with open(OUT_FILE, 'r', encoding='utf-8') as f_chk:
    for i, line in enumerate(f_chk):
        print(repr(line[:100]))
        if i >= 3: break
print('DONE')
