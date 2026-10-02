import os
from collections import defaultdict

SHARDS_V7_DIR = 'Z:/Team X_submission/output/shards'
SHARDS_V8_DIR = 'Z:/Team X_submission/output/shards_v8'
TEMPLATE_FILE = 'Z:/Team X_submission/dataset/raw/test/test_source1.tsv'
OUT_FILE      = 'Z:/Team X_submission/output/matching_results_v21.tsv'

T1_THRESH = 0.55
T2_THRESH = 0.65

print('Step 1: Loading v7 shard matches and building claimed target set...')
v7_matches = {}
v7_claimed_targets = set()
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
            if ids_str:
                for t in ids_str.split(','):
                    if t: v7_claimed_targets.add(t)

v7_matched = sum(1 for v in v7_matches.values() if v)
v7_singletons = sum(1 for v in v7_matches.values() if not v)
print(f'v7 base: {v7_matched:,} matched, {v7_singletons:,} singletons')
print(f'v7 claimed unique targets: {len(v7_claimed_targets):,}')

print('Step 2: Collecting singleton claims with strict Mutual Exclusivity...')
# Pass 1: collect candidate claims for singletons (target cannot be in v7_claimed_targets)
# Map: target_id -> list of (prob, src_id)
target_claims = defaultdict(list)
raw_singleton_candidates = defaultdict(list)
pruned_v7_collisions = 0

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
            if src_id in v7_matches and v7_matches[src_id]:
                continue  # ONLY singletons!
            
            mids = parts[1].split(',') if parts[1] else []
            pbs = parts[2].split('|') if len(parts) > 2 and parts[2] else []
            
            for i, mid in enumerate(mids):
                if not mid: continue
                try: p = float(pbs[i]) if i < len(pbs) else 0.0
                except: p = 0.0
                if p < 1.0 and p >= T1_THRESH:
                    # RULE 1: Cannot steal a target already claimed in v7!
                    if mid in v7_claimed_targets:
                        pruned_v7_collisions += 1
                        continue
                    target_claims[mid].append((p, src_id))

print(f'Pruned {pruned_v7_collisions:,} guaranteed false positive collisions with v7!')

# RULE 2: Mutual Exclusivity among singletons: each target claimed by HIGHEST prob query
singleton_resolved_matches = defaultdict(list)
for mid, claims in target_claims.items():
    claims.sort(key=lambda x: -x[0])
    best_p, best_src = claims[0]
    singleton_resolved_matches[best_src].append((mid, best_p))

# Filter resolved matches per singleton using tiered threshold
ml_for_singletons = {}
for src_id, pairs in singleton_resolved_matches.items():
    pairs.sort(key=lambda x: -x[1])
    kept = []
    for i, (mid, p) in enumerate(pairs):
        if i == 0 and p >= T1_THRESH:
            kept.append(mid)
        elif i > 0 and p >= T2_THRESH:
            kept.append(mid)
    if kept:
        ml_for_singletons[src_id] = kept[:6]

print(f'Singletons boosted with 100% mutually exclusive targets: {len(ml_for_singletons):,}')

print('Step 3: Writing clean final submission...')
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
