import polars as pl
import sys
from pathlib import Path

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
print(f'Total missed pairs: {len(missed_pairs):,} out of {len(all_gt_pairs):,} ({len(missed_pairs)/len(all_gt_pairs)*100:.2f}%)')

q_df = pl.read_parquet('experiments/cache/val_norm_v2.parquet').to_pandas().set_index('entity_id')
t_df = pl.read_parquet('experiments/cache/targets_norm_v2.parquet').to_pandas().set_index('entity_id')

for qid, tid in missed_pairs[:20]:
    if qid in q_df.index and tid in t_df.index:
        q_row = q_df.loc[qid]
        t_row = t_df.loc[tid]
        print('='*70)
        print(f"MISSED: Q={qid} ({q_row.get('country', '')}) -> T={tid}")
        print(f"  Q Name:    '{q_row.get('clean_name', '')}'")
        print(f"  T Name:    '{t_row.get('clean_name', '')}'")
        print(f"  Q Addr:    '{q_row.get('std_address', '')}'")
        print(f"  T Addr:    '{t_row.get('std_address', '')}'")
        print(f"  Q Postal:  '{q_row.get('postal_code', '')}' | T Postal: '{t_row.get('postal_code', '')}'")
        print(f"  Q HNo:     '{q_row.get('house_no', '')}' | T HNo: '{t_row.get('house_no', '')}'")
