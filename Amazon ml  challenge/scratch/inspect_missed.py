import polars as pl
import re, unicodedata
from collections import defaultdict
import sys
sys.stdout.reconfigure(encoding='utf-8')

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

def strip_accents(s):
    if not s: return ""
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def deleet(s):
    if not s: return ""
    s = re.sub(r'(?<=[a-zA-Z])0(?=[a-zA-Z])|(?<=[a-zA-Z])0|0(?=[a-zA-Z])', 'o', s)
    s = re.sub(r'(?<=[a-zA-Z])1(?=[a-zA-Z])|(?<=[a-zA-Z])1|1(?=[a-zA-Z])', 'l', s)
    s = re.sub(r'(?<=[a-zA-Z])3(?=[a-zA-Z])|(?<=[a-zA-Z])3|3(?=[a-zA-Z])', 'e', s)
    s = re.sub(r'(?<=[a-zA-Z])4(?=[a-zA-Z])|(?<=[a-zA-Z])4|4(?=[a-zA-Z])', 'a', s)
    s = re.sub(r'(?<=[a-zA-Z])5(?=[a-zA-Z])|(?<=[a-zA-Z])5|5(?=[a-zA-Z])', 's', s)
    return s

def clean_name(s):
    s = strip_accents(str(s or '')).lower()
    s = re.sub(r'\.(com|org|net|in|co|us|gov|io)\b', '', s)
    s = re.sub(r'\b(www|http|https)\b', '', s)
    s = re.sub(r'&', ' and ', s)
    s = re.sub(r'[^\w\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def strip_suffixes(s):
    LEGAL = r'\b(private limited|pvt ltd|pvt limited|private ltd|limited|ltd|pvt|llc|inc|incorporated|corporation|corp|llp|co|company|center|services|service|partners|group|holdings)\b'
    return re.sub(r'\s+', ' ', re.sub(LEGAL, '', s)).strip()

def sorted_tokens(s):
    toks = [t for t in s.split() if len(t) >= 2]
    return ' '.join(sorted(toks))

def spaceless(s):
    return re.sub(r'\s+', '', s)

missed = []
for qid, tids in gt_map.items():
    q_row = s1_dict.get(qid)
    if not q_row: continue
    q_nm = clean_name(q_row['business_name'])
    q_nm_strip = strip_suffixes(q_nm)
    q_nm_sort = sorted_tokens(q_nm_strip)
    q_nm_space = spaceless(q_nm_strip)
    q_nm_deleet = sorted_tokens(strip_suffixes(clean_name(deleet(q_row['business_name']))))
    
    q_addr = clean_name(q_row['business_address'])
    q_addr_sort = sorted_tokens(q_addr)
    
    for tid in tids:
        t_row = tgt_dict.get(tid)
        if not t_row: continue
        t_raw_nm = str(t_row['business_name'] or '')
        t_nm = clean_name(t_raw_nm)
        t_nm_strip = strip_suffixes(t_nm)
        t_nm_sort = sorted_tokens(t_nm_strip)
        t_nm_space = spaceless(t_nm_strip)
        t_nm_deleet = sorted_tokens(strip_suffixes(clean_name(deleet(t_raw_nm))))
        t_addr = clean_name(str(t_row['business_address'] or ''))
        t_addr_sort = sorted_tokens(t_addr)
        
        found = False
        if q_nm and t_nm and q_nm == t_nm: found = True
        elif q_nm_sort and t_nm_sort and q_nm_sort == t_nm_sort: found = True
        elif q_nm_space and t_nm_space and len(q_nm_space) >= 5 and (q_nm_space == t_nm_space or q_nm_space in t_nm_space or t_nm_space in q_nm_space): found = True
        elif q_nm_deleet and t_nm_deleet and q_nm_deleet == t_nm_deleet: found = True
        elif q_addr and t_addr and len(q_addr) >= 10 and q_addr == t_addr: found = True
        elif q_addr_sort and t_addr_sort and len(q_addr_sort) >= 12 and q_addr_sort == t_addr_sort: found = True
        else:
            q_toks = set(q_addr_sort.split())
            t_toks = set(t_addr_sort.split())
            overlap = len(q_toks & t_toks)
            if len(q_toks) >= 3 and len(t_toks) >= 3 and overlap / min(len(q_toks), len(t_toks)) >= 0.8:
                found = True
        
        if not found:
            missed.append((q_row, t_row))

print(f"Total missed in sample: {len(missed)}")
print("Sample of 15 missed pairs:")
for q, t in missed[:15]:
    print(f"\nQID: {q['entity_id']} | TID: {t['entity_id']} ({q['country']})")
    print(f"  S1:  Name='{q['business_name']}' | Addr='{q['business_address']}'")
    print(f"  Tgt: Name='{t['business_name']}' | Addr='{t['business_address']}'")
