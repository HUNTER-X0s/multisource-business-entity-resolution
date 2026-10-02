import os

SHARDS_V7_DIR = 'Z:/Amazon ML/output/shards'
SHARDS_V8_DIR = 'Z:/Amazon ML/output/shards_v8'
TEMPLATE_FILE  = 'Z:/Amazon ML/dataset/raw/test/test_source1.tsv'
OUT_FILE = 'Z:/Amazon ML/output/matching_results_v15.tsv'

SINGLETON_ML_THRESH = 0.60
BYPASS_REMOVE_IF_ALL_ML_BELOW = 0.65  # remove entity if ALL its v8 ML probs < 0.65

print('Loading source1 IDs...')
source1_ids = []
with open(TEMPLATE_FILE,'r',encoding='utf-8') as f:
    f.readline()
    for line in f:
        p = line.strip().split('\t')
        if p and p[0]: source1_ids.append(p[0])
print(f'Total: {len(source1_ids):,}')

shard_files = sorted([f for f in os.listdir(SHARDS_V7_DIR) if f.endswith('_matches.tsv')])
result = {}

bypass_removed_entities = 0
ml_confirmed_kept = 0
not_in_v8_kept = 0
singletons_added = 0

for sf in shard_files:
    print(f'Processing {sf}...')
    v7_shard = {}
    with open(os.path.join(SHARDS_V7_DIR, sf),'r',encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if not parts[0]: continue
            ids = parts[1].split(',') if len(parts)>1 and parts[1] else []
            v7_shard[parts[0]] = ids

    # Build v8 ML lookup per entity: {src_id: {target_id: max_ml_prob}}
    v8_ml = {}
    v8_singleton_ml = {}
    path8 = os.path.join(SHARDS_V8_DIR, sf)
    if os.path.exists(path8):
        with open(path8,'r',encoding='utf-8') as f:
            f.readline()
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 2: continue
                src = parts[0]
                mids = parts[1].split(',') if parts[1] else []
                pbs = parts[2].split('|') if len(parts)>2 and parts[2] else []
                ml = {}
                sl = []
                for i,mid in enumerate(mids):
                    if not mid: continue
                    try: p = float(pbs[i]) if i<len(pbs) else 0.0
                    except: p = 0.0
                    if p < 1.0:
                        ml[mid] = max(ml.get(mid,0.0), p)
                        if p >= SINGLETON_ML_THRESH and not v7_shard.get(src):
                            sl.append((mid,p))
                if ml: v8_ml[src] = ml
                if sl:
                    sl.sort(key=lambda x:-x[1])
                    v8_singleton_ml[src] = [m for m,_ in sl[:6]]

    for src_id, v7_ids in v7_shard.items():
        if not v7_ids:
            # Singleton: add ML boosts
            extras = v8_singleton_ml.get(src_id, [])
            if extras:
                result[src_id] = ','.join(extras)
                singletons_added += 1
            else:
                result[src_id] = ''
            continue

        # Matched entity: check if bypass-only
        ml_preds = v8_ml.get(src_id, None)

        if ml_preds is None:
            # Entity not in v8 at all → keep conservatively (likely from v7's own ML)
            result[src_id] = ','.join(v7_ids)
            not_in_v8_kept += 1
        else:
            # Check: does this entity have ANY v7 pair confirmed by ML >= 0.65?
            has_ml_confirmed = any(ml_preds.get(mid, 0.0) >= BYPASS_REMOVE_IF_ALL_ML_BELOW
                                   for mid in v7_ids)
            if has_ml_confirmed:
                # Has ML confirmation → keep all v7 pairs
                result[src_id] = ','.join(v7_ids)
                ml_confirmed_kept += 1
            else:
                # BYPASS-ONLY: all v7 pairs have ML < 0.65 → REMOVE
                result[src_id] = ''
                bypass_removed_entities += 1

print(f'Stats:')
print(f'  ML-confirmed kept:        {ml_confirmed_kept:,}')
print(f'  Not-in-v8 kept:           {not_in_v8_kept:,}')
print(f'  Bypass-only REMOVED:      {bypass_removed_entities:,}')
print(f'  Singletons added (v12):   {singletons_added:,}')

print('Writing v15...')
total_matched = 0
with open(OUT_FILE,'w',encoding='utf-8') as f:
    f.write('source1_entity_id\tmatched_entity_ids\n')
    for src_id in source1_ids:
        ids_str = result.get(src_id,'')
        f.write(src_id + '\t' + ids_str + '\n')
        if ids_str: total_matched += 1

print(f'Total matched: {total_matched:,} / {len(source1_ids):,}')
print(f'File size: {os.path.getsize(OUT_FILE)/1024/1024:.1f} MB')
print('v15 DONE')
