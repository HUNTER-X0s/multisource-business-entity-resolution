import os
from collections import Counter

SHARDS_V7_DIR = 'Z:/Amazon ML/output/shards'
SHARDS_V8_DIR = 'Z:/Amazon ML/output/shards_v8'
TEMPLATE_FILE  = 'Z:/Amazon ML/dataset/raw/test/test_source1.tsv'
OUT_FILE = 'Z:/Amazon ML/output/matching_results_v16.tsv'

SINGLETON_ML_THRESH = 0.60      # keep v12's ML singleton boosts
BYPASS_FREQ_THRESHOLD = 5       # bypass target must appear for <=5 S1 entities to be "specific"

print('Step 1: Count bypass target frequencies (all shards)...')
bypass_target_freq = Counter()  # target_id -> how many S1 entities use it as bypass
shard_files = sorted([f for f in os.listdir(SHARDS_V8_DIR) if f.endswith('_matches.tsv')])
for sf in shard_files:
    path8 = os.path.join(SHARDS_V8_DIR, sf)
    with open(path8,'r',encoding='utf-8') as f:
        f.readline()
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3: continue
            mids = parts[1].split(',') if parts[1] else []
            pbs = parts[2].split('|') if parts[2] else []
            for i,mid in enumerate(mids):
                if not mid: continue
                try: prob = float(pbs[i]) if i < len(pbs) else 0.0
                except: prob = 0.0
                if prob == 1.0:  # bypass
                    bypass_target_freq[mid] += 1

print(f'Unique bypass targets: {len(bypass_target_freq):,}')
specific = sum(1 for c in bypass_target_freq.values() if c <= BYPASS_FREQ_THRESHOLD)
generic = sum(1 for c in bypass_target_freq.values() if c > BYPASS_FREQ_THRESHOLD)
print(f'  Specific (<=5 S1): {specific:,}')
print(f'  Generic (>5 S1):   {generic:,}')

print()
print('Step 2: Load v7 matches...')
v7_matches = {}
shard_files_v7 = sorted([f for f in os.listdir(SHARDS_V7_DIR) if f.endswith('_matches.tsv')])
for sf in shard_files_v7:
    with open(os.path.join(SHARDS_V7_DIR, sf),'r',encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if parts[0]:
                v7_matches[parts[0]] = parts[1] if len(parts)>1 else ''

print(f'v7 matches loaded: {sum(1 for v in v7_matches.values() if v):,} matched')

print()
print('Step 3: For SINGLETONS - add specific bypass OR ML>=0.60...')
extra_from_specific_bypass = {}
extra_from_ml = {}

for sf in shard_files:
    path8 = os.path.join(SHARDS_V8_DIR, sf)
    with open(path8,'r',encoding='utf-8') as f:
        f.readline()
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2: continue
            src_id = parts[0]
            if v7_matches.get(src_id,''):
                continue  # already matched in v7 - skip

            mids = parts[1].split(',') if parts[1] else []
            pbs = parts[2].split('|') if len(parts)>2 and parts[2] else []
            
            ml_pairs = []
            specific_bypass_pairs = []
            
            for i,mid in enumerate(mids):
                if not mid: continue
                try: prob = float(pbs[i]) if i < len(pbs) else 0.0
                except: prob = 0.0
                
                if SINGLETON_ML_THRESH <= prob < 1.0:
                    ml_pairs.append((mid, prob))
                elif prob == 1.0:
                    # Only accept specific bypass
                    if bypass_target_freq.get(mid, 999) <= BYPASS_FREQ_THRESHOLD:
                        specific_bypass_pairs.append(mid)
            
            if ml_pairs:
                ml_pairs.sort(key=lambda x:-x[1])
                extra_from_ml[src_id] = [m for m,_ in ml_pairs[:6]]
            elif specific_bypass_pairs:
                extra_from_specific_bypass[src_id] = specific_bypass_pairs[:6]

print(f'Singletons recovered via ML>=0.60:        {len(extra_from_ml):,}')
print(f'Singletons recovered via specific bypass:  {len(extra_from_specific_bypass):,}')
print(f'Total new singleton matches:               {len(extra_from_ml)+len(extra_from_specific_bypass):,}')

print()
print('Step 4: Writing v16...')
source1_ids = []
with open(TEMPLATE_FILE,'r',encoding='utf-8') as f:
    f.readline()
    for line in f:
        p = line.strip().split('\t')
        if p and p[0]: source1_ids.append(p[0])

total_matched = 0
with open(OUT_FILE,'w',encoding='utf-8') as f:
    f.write('source1_entity_id\tmatched_entity_ids\n')
    for src_id in source1_ids:
        v7_str = v7_matches.get(src_id,'')
        if v7_str:
            f.write(src_id + '\t' + v7_str + '\n')
            total_matched += 1
        elif src_id in extra_from_ml:
            f.write(src_id + '\t' + ','.join(extra_from_ml[src_id]) + '\n')
            total_matched += 1
        elif src_id in extra_from_specific_bypass:
            f.write(src_id + '\t' + ','.join(extra_from_specific_bypass[src_id]) + '\n')
            total_matched += 1
        else:
            f.write(src_id + '\t\n')

print(f'Total matched: {total_matched:,} / {len(source1_ids):,}')
print(f'  v7 base: {sum(1 for v in v7_matches.values() if v):,}')
print(f'  ML singletons: {len(extra_from_ml):,}')
print(f'  Specific bypass singletons: {len(extra_from_specific_bypass):,}')
print(f'File size: {os.path.getsize(OUT_FILE)/1024/1024:.1f} MB')
print('v16 DONE')
