import os
from collections import Counter

SHARDS_V7_DIR = 'Z:/Amazon ML/output/shards'
SHARDS_V8_DIR = 'Z:/Amazon ML/output/shards_v8'
TEMPLATE_FILE  = 'Z:/Amazon ML/dataset/raw/test/test_source1.tsv'
OUT_FILE = 'Z:/Amazon ML/output/matching_results_v17.tsv'

SINGLETON_ML_THRESH = 0.60
BYPASS_FREQ_THRESHOLD = 1  # STRICT: target must appear for EXACTLY 1 S1 entity

print('Step 1: Count bypass target frequencies...')
bypass_target_freq = Counter()
shard_files = sorted([f for f in os.listdir(SHARDS_V8_DIR) if f.endswith('_matches.tsv')])
for sf in shard_files:
    with open(os.path.join(SHARDS_V8_DIR, sf),'r',encoding='utf-8') as f:
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
                if prob == 1.0:
                    bypass_target_freq[mid] += 1

singleton_targets = sum(1 for c in bypass_target_freq.values() if c == 1)
print(f'Bypass targets appearing for EXACTLY 1 S1: {singleton_targets:,}')
print(f'Bypass targets appearing for >1 S1: {len(bypass_target_freq)-singleton_targets:,}')

print('Step 2: Load v7 matches...')
v7_matches = {}
for sf in sorted([f for f in os.listdir(SHARDS_V7_DIR) if f.endswith('_matches.tsv')]):
    with open(os.path.join(SHARDS_V7_DIR, sf),'r',encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if parts[0]: v7_matches[parts[0]] = parts[1] if len(parts)>1 else ''

print('Step 3: Build singleton extras...')
extra_from_ml = {}
extra_from_singleton_bypass = {}

for sf in shard_files:
    with open(os.path.join(SHARDS_V8_DIR, sf),'r',encoding='utf-8') as f:
        f.readline()
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2: continue
            src_id = parts[0]
            if v7_matches.get(src_id,''): continue  # already matched
            mids = parts[1].split(',') if parts[1] else []
            pbs = parts[2].split('|') if len(parts)>2 and parts[2] else []
            ml_pairs = []
            sbp = []
            for i,mid in enumerate(mids):
                if not mid: continue
                try: p = float(pbs[i]) if i<len(pbs) else 0.0
                except: p = 0.0
                if SINGLETON_ML_THRESH <= p < 1.0:
                    ml_pairs.append((mid,p))
                elif p == 1.0 and bypass_target_freq.get(mid,999) <= BYPASS_FREQ_THRESHOLD:
                    sbp.append(mid)
            if ml_pairs:
                ml_pairs.sort(key=lambda x:-x[1])
                extra_from_ml[src_id] = [m for m,_ in ml_pairs[:6]]
            elif sbp:
                extra_from_singleton_bypass[src_id] = sbp[:4]

print(f'ML>=0.60 singletons: {len(extra_from_ml):,}')
print(f'Unique-target bypass singletons: {len(extra_from_singleton_bypass):,}')
total_added = len(extra_from_ml) + len(extra_from_singleton_bypass)
expected_total = 1481980 + total_added
print(f'Expected total matched: {expected_total:,} ({100*expected_total/1732544:.1f}%)')
print(f'Expected singleton rate: {100*(1732544-expected_total)/1732544:.1f}% (training GT: 5.6%)')

print('Step 4: Writing v17...')
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
        v7 = v7_matches.get(src_id,'')
        if v7:
            f.write(src_id+'\t'+v7+'\n'); total_matched+=1
        elif src_id in extra_from_ml:
            f.write(src_id+'\t'+','.join(extra_from_ml[src_id])+'\n'); total_matched+=1
        elif src_id in extra_from_singleton_bypass:
            f.write(src_id+'\t'+','.join(extra_from_singleton_bypass[src_id])+'\n'); total_matched+=1
        else:
            f.write(src_id+'\t\n')

print(f'Total matched: {total_matched:,} / {len(source1_ids):,} ({100*total_matched/len(source1_ids):.1f}%)')
print(f'File size: {os.path.getsize(OUT_FILE)/1024/1024:.1f} MB')
print('v17 DONE')
