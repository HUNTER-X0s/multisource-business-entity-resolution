import polars as pl
import sys
sys.stdout.reconfigure(encoding='utf-8')

gt = pl.read_csv('dataset/raw/train/train_ground_truth.tsv', separator='\t', quote_char=None)
gt_pos = gt.filter(pl.col('matched_entity_ids').is_not_null() & (pl.col('matched_entity_ids') != '')).slice(10, 10)

s1_df = pl.read_csv('dataset/raw/train/train_source1.tsv', separator='\t', quote_char=None)
s2_df = pl.read_csv('dataset/raw/train/train_source2.tsv', separator='\t', quote_char=None)
s3_df = pl.read_csv('dataset/raw/train/train_source3.tsv', separator='\t', quote_char=None)

for row in gt_pos.iter_rows(named=True):
    qid = row['source1_entity_id']
    m = [x.strip() for x in row['matched_entity_ids'].split(',') if x.strip()]
    s1_row = s1_df.filter(pl.col('entity_id') == qid).to_dicts()[0]
    print(f"\n=== QID: {qid} | Name: '{s1_row['business_name']}' | Addr: '{s1_row['business_address']}' | Country: {s1_row['country']} ===")
    for tid in m:
        df = s2_df if tid.startswith('S2-') else s3_df
        t_row = df.filter(pl.col('entity_id') == tid).to_dicts()[0]
        print(f"   -> {tid}: Name='{t_row['business_name']}' | Addr='{t_row['business_address']}'")
