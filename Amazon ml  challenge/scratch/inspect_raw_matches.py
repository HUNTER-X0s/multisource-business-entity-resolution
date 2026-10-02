import pandas as pd

gt = pd.read_csv('dataset/raw/train/train_ground_truth.tsv', sep='\t', nrows=25)
gt['matched_entity_ids'] = gt['matched_entity_ids'].fillna('')

s1 = pd.read_csv('dataset/raw/train/train_source1.tsv', sep='\t')
s2 = pd.read_csv('dataset/raw/train/train_source2.tsv', sep='\t')
s3 = pd.read_csv('dataset/raw/train/train_source3.tsv', sep='\t')

s1_map = s1.set_index('entity_id').to_dict('index')
s2_map = s2.set_index('entity_id').to_dict('index')
s3_map = s3.set_index('entity_id').to_dict('index')

for i, row in gt.head(10).iterrows():
    qid = row['source1_entity_id']
    mids = [x.strip() for x in str(row['matched_entity_ids']).split(',') if x.strip()]
    q_data = s1_map.get(qid, {})
    print(f"\n==================== QUERY {qid} ====================")
    print(f"S1: {q_data.get('business_name')} || {q_data.get('business_address')} || {q_data.get('country')}")
    for m in mids:
        if m.startswith('S2-'):
            t_data = s2_map.get(m, {})
            print(f"  {m}: {t_data.get('business_name')} || {t_data.get('business_address')} || {t_data.get('country')}")
        elif m.startswith('S3-'):
            t_data = s3_map.get(m, {})
            print(f"  {m}: {t_data.get('business_name')} || {t_data.get('business_address')} || {t_data.get('country')}")
