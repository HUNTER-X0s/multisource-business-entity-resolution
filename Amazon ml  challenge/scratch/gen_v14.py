import os

SHARDS_V7_DIR = 'Z:/Amazon ML/output/shards'
SHARDS_V8_DIR = 'Z:/Amazon ML/output/shards_v8'
TEMPLATE_FILE  = 'Z:/Amazon ML/dataset/raw/test/test_source1.tsv'
OUT_FILE = 'Z:/Amazon ML/output/matching_results_v14.tsv'

# STRATEGY: v12 (best=0.799) but ALSO remove pairs from v7 where
# v8 ML gave < 0.50 probability (bypass-only, no ML confirmation)
# These are the WEAKEST v7 pairs and likely the biggest source of FP

SINGLETON_ML_THRESH = 0.60    # same as v12
BYPASS_KEEP_THRESH = 0.50     # for existing v7 pairs: keep if ML >= 0.50

print('Loading source1 IDs...')
source1_ids = []
with open(TEMPLATE_FILE, 'r', encoding='utf-8') as f:
    f.readline()
    for line in f:
        p = line.strip().split('\t')
        if p and p[0]: source1_ids.append(p[0])
print(f'Total: {len(source1_ids):,}')

print('Processing shard by shard (memory efficient)...')
shard_files = sorted([f for f in os.listdir(SHARDS_V7_DIR) if f.endswith('_matches.tsv')])

result = {}  # src_id -> final ids string

for sf in shard_files:
    print(f'  Processing {sf}...')
    # Load v7 shard
    v7_shard = {}
    with open(os.path.join(SHARDS_V7_DIR, sf), 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split('\t')
            src_id = parts[0]
            ids_str = parts[1] if len(parts) > 1 else ''
            v7_shard[src_id] = ids_str.split(',') if ids_str else []

    # Load v8 shard: build {src_id: {target_id: max_ml_prob}}
    v8_ml = {}  # ML probs ONLY (no bypass)
    v8_singleton_ml = {}  # ML probs for singletons
    path8 = os.path.join(SHARDS_V8_DIR, sf)
    if os.path.exists(path8):
        with open(path8, 'r', encoding='utf-8') as f:
            f.readline()
            for line in f:
                line = line.strip()
                if not line: continue
                parts = line.split('\t')
                if len(parts) < 2: continue
                src_id = parts[0]
                matched_ids = parts[1].split(',') if parts[1] else []
                probs_str = parts[2].split('|') if len(parts) > 2 and parts[2] else []
                
                ml_pairs = {}
                singleton_ml_pairs = []
                for i, mid in enumerate(matched_ids):
                    if not mid: continue
                    try: prob = float(probs_str[i]) if i < len(probs_str) else 0.0
                    except: prob = 0.0
                    # ML-only (not bypass)
                    if prob < 1.0:
                        ml_pairs[mid] = max(ml_pairs.get(mid, 0.0), prob)
                        # For singletons: collect >=0.60
                        if prob >= SINGLETON_ML_THRESH and not v7_shard.get(src_id):
                            singleton_ml_pairs.append((mid, prob))
                
                if ml_pairs:
                    v8_ml[src_id] = ml_pairs
                if singleton_ml_pairs:
                    singleton_ml_pairs.sort(key=lambda x: -x[1])
                    v8_singleton_ml[src_id] = [m for m,_ in singleton_ml_pairs[:6]]

    # Build results for this shard
    removed_pairs = 0
    kept_bypass = 0
    singleton_added = 0
    
    for src_id, v7_ids in v7_shard.items():
        ml_preds = v8_ml.get(src_id, {})
        
        if v7_ids:
            # Filter: keep v7 pairs that have ML >= 0.50 OR are ML-confirmed
            filtered = []
            for mid in v7_ids:
                if not mid: continue
                ml_prob = ml_preds.get(mid, -1)
                if ml_prob >= BYPASS_KEEP_THRESH:
                    filtered.append(mid)  # ML confirmed
                    kept_bypass += 1
                elif ml_prob == -1:
                    # Not in v8 ML at all - might be from v7's own ML or bypass
                    # KEEP it (conservative - don't remove unknown pairs)
                    filtered.append(mid)
                    kept_bypass += 1
                else:
                    # ML prob < 0.50: likely FP bypass pair, remove
                    removed_pairs += 1
            
            result[src_id] = ','.join(filtered) if filtered else ''
        else:
            # Singleton: add ML >= 0.60 from v8
            extras = v8_singleton_ml.get(src_id, [])
            if extras:
                result[src_id] = ','.join(extras)
                singleton_added += 1
            else:
                result[src_id] = ''
    
    print(f'    Removed {removed_pairs:,} bypass-FP pairs, kept {kept_bypass:,}, added {singleton_added:,} singletons')

print('Writing v14...')
total_matched = 0
with open(OUT_FILE, 'w', encoding='utf-8') as f:
    f.write('source1_entity_id\tmatched_entity_ids\n')
    for src_id in source1_ids:
        ids_str = result.get(src_id, '')
        f.write(src_id + '\t' + ids_str + '\n')
        if ids_str: total_matched += 1

print(f'Total matched: {total_matched:,} / {len(source1_ids):,}')
print(f'File size: {os.path.getsize(OUT_FILE)/1024/1024:.1f} MB')
print('v14 DONE')
