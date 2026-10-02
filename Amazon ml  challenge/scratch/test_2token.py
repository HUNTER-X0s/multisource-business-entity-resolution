import polars as pl
from collections import defaultdict
import re

print("Testing Distinctive 2-Token Address Blocking...")

# Stop words for address
ADDR_STOPS = {
    'street', 'avenue', 'road', 'drive', 'lane', 'boulevard', 'circle', 'court', 'place', 'way',
    'highway', 'suite', 'floor', 'unit', 'building', 'tower', 'block', 'phase', 'sector', 'plot',
    'near', 'behind', 'opposite', 'cross', 'main', 'first', 'second', 'third', 'ground', 'upper',
    'delhi', 'mumbai', 'bangalore', 'kolkata', 'chennai', 'hyderabad', 'pune', 'ahmedabad', 'jaipur',
    'india', 'texas', 'california', 'florida', 'york', 'washington', 'illinois', 'carolina', 'georgia',
    'pradesh', 'maharashtra', 'karnataka', 'tamil', 'nadu', 'gujarat', 'bengal', 'rajasthan', 'kerala',
}

def get_distinctive_addr_pair(addr):
    toks = [w for w in re.sub(r'[^\w\s]', ' ', str(addr or '').lower()).split() 
            if len(w) >= 5 and not w.isdigit() and w not in ADDR_STOPS]
    if len(toks) >= 2:
        # Sort by length descending, pick top 2 longest words, then alphabetically sort
        top2 = sorted(sorted(toks, key=len, reverse=True)[:2])
        return f"{top2[0]}_{top2[1]}"
    return ""

def get_distinctive_name_pair(nm):
    toks = [w for w in re.sub(r'[^\w\s]', ' ', str(nm or '').lower()).split() 
            if len(w) >= 4 and not w.isdigit()]
    if len(toks) >= 2:
        top2 = sorted(sorted(toks, key=len, reverse=True)[:2])
        return f"{top2[0]}_{top2[1]}"
    return ""

# Test on 5,000 ground truth queries
gt = pl.read_csv('dataset/raw/train/train_ground_truth.tsv', separator='\t', quote_char=None)
gt_pos = gt.filter(pl.col('matched_entity_ids').is_not_null() & (pl.col('matched_entity_ids') != '')).slice(0, 5000)

s1_ids = set(gt_pos['source1_entity_id'])
all_targets = set()
gt_map = {}
for row in gt_pos.iter_rows(named=True):
    qid = row['source1_entity_id']
    tids = [t.strip() for t in row['matched_entity_ids'].split(',') if t.strip()]
    gt_map[qid] = set(tids)
    all_targets.update(tids)

s1_df = pl.read_csv('dataset/raw/train/train_source1.tsv', separator='\t', quote_char=None).filter(pl.col('entity_id').is_in(list(s1_ids)))
s2_df = pl.read_csv('dataset/raw/train/train_source2.tsv', separator='\t', quote_char=None).filter(pl.col('entity_id').is_in(list(all_targets)))
s3_df = pl.read_csv('dataset/raw/train/train_source3.tsv', separator='\t', quote_char=None).filter(pl.col('entity_id').is_in(list(all_targets)))
targets_df = pl.concat([s2_df, s3_df])

s1_dict = {r['entity_id']: r for r in s1_df.to_dicts()}
tgt_dict = {r['entity_id']: r for r in targets_df.to_dicts()}

total_pairs = sum(len(v) for v in gt_map.values())
addr_match = 0
name_match = 0
either_match = 0

for qid, tids in gt_map.items():
    q = s1_dict.get(qid)
    if not q: continue
    q_ap = get_distinctive_addr_pair(q['business_address'])
    q_np = get_distinctive_name_pair(q['business_name'])
    for tid in tids:
        t = tgt_dict.get(tid)
        if not t: continue
        t_ap = get_distinctive_addr_pair(t['business_address'])
        t_np = get_distinctive_name_pair(t['business_name'])
        
        a_m = bool(q_ap and t_ap and q_ap == t_ap)
        n_m = bool(q_np and t_np and q_np == t_np)
        if a_m: addr_match += 1
        if n_m: name_match += 1
        if a_m or n_m: either_match += 1

print(f"Total True Pairs:              {total_pairs:,}")
print(f"Distinctive Name 2-Token:      {name_match:,} ({name_match/total_pairs*100:.2f}%)")
print(f"Distinctive Address 2-Token:   {addr_match:,} ({addr_match/total_pairs*100:.2f}%)")
print(f"Either Name or Address 2-Tok:  {either_match:,} ({either_match/total_pairs*100:.2f}%)")
