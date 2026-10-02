import polars as pl

val_df = pl.read_parquet('experiments/val_sample_50k.parquet')
val_gt = {}
for row in val_df.iter_rows(named=True):
    qid = str(row['entity_id'])
    raw = str(row.get('matched_entity_ids', '') or '').strip()
    val_gt[qid] = {t.strip() for t in raw.split(',') if t.strip() and t.strip() not in ('nan', 'None')}

all_gt_pairs = set((q, t) for q, targets in val_gt.items() for t in targets)
cands = pl.read_parquet('experiments/phase2/phase2c_10ch_val_candidates.parquet')
cand_pairs = set((row['entity_id'], row['target_id']) for row in cands.iter_rows(named=True))
missed_pairs = list(all_gt_pairs - cand_pairs)

q_df = pl.read_parquet('experiments/cache/val_norm_v2.parquet').to_pandas().set_index('entity_id')
t_df = pl.read_parquet('experiments/cache/targets_norm_v2.parquet').to_pandas().set_index('entity_id')

stopwords = {'village', 'h', 'no', 'road', 'rd', 'street', 'st', 'ave', 'avenue', 'dr', 'drive', 'up', 'uttar', 'pradesh', 'india', 'us', 'near', 'floor', 'opp'}
legal_stops = {'private', 'limited', 'ltd', 'pvt', 'inc', 'llc', 'corp', 'co', 'services', 'solutions'}

addr_match = 0
name_match = 0
either_match = 0

for qid, tid in missed_pairs:
    if qid in q_df.index and tid in t_df.index:
        qa = set((q_df.loc[qid, 'std_address'] or '').split()) - stopwords
        ta = set((t_df.loc[tid, 'std_address'] or '').split()) - stopwords
        qn = set((q_df.loc[qid, 'clean_name'] or '').split()) - legal_stops
        tn = set((t_df.loc[tid, 'clean_name'] or '').split()) - legal_stops
        
        has_addr = len(qa & ta) >= 2
        has_name = len(qn & tn) >= 1
        
        if has_addr: addr_match += 1
        if has_name: name_match += 1
        if has_addr or has_name: either_match += 1

print(f"Total missed pairs: {len(missed_pairs):,}")
print(f"Missed pairs with >= 2 shared non-stop address tokens: {addr_match:,} ({addr_match/len(missed_pairs)*100:.2f}%)")
print(f"Missed pairs with >= 1 shared non-stop name token: {name_match:,} ({name_match/len(missed_pairs)*100:.2f}%)")
print(f"Missed pairs recoverable by either address or name tokens: {either_match:,} ({either_match/len(missed_pairs)*100:.2f}%)")
