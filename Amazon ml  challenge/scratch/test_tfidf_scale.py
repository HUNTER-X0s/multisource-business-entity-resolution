"""
scratch/test_tfidf_scale.py
Test scale and recall on 5k queries against full 762k target corpus.
"""
import time, re, sys, random
from collections import defaultdict
from pathlib import Path
import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import scipy.sparse as sp

ROOT = Path("z:/Amazon ML")
VAL_SAMPLE  = ROOT / "experiments" / "val_sample_50k.parquet"
FEAT_FILE   = ROOT / "experiments" / "phase2" / "val_50k_features.parquet"
S2_PATH     = ROOT / "dataset" / "raw" / "train" / "train_source2.tsv"
S3_PATH     = ROOT / "dataset" / "raw" / "train" / "train_source3.tsv"
S1_PATH     = ROOT / "dataset" / "raw" / "train" / "train_source1.tsv"

def flush(*args, **kwargs):
    print(*args, **kwargs, flush=True)

flush("Loading data...")
t0 = time.time()
val_df = pl.read_parquet(VAL_SAMPLE)
val_gt = {}
for row in val_df.iter_rows(named=True):
    qid = str(row["entity_id"])
    raw = str(row.get("matched_entity_ids", "") or "").strip()
    targets = set()
    if raw and raw not in ("nan", "None", ""):
        for t in raw.split(","):
            t = t.strip()
            if t:
                targets.add(t)
    val_gt[qid] = targets
val_query_list = sorted(val_gt.keys())[:5000] # test on 5k queries
all_pos_tids = set(tid for v in val_gt.values() for tid in v) # ALL 172k positive targets

feat_df = pl.read_parquet(FEAT_FILE)
baseline_cands = defaultdict(set)
for row in feat_df.select(["entity_id", "target_id"]).iter_rows(named=True):
    q = str(row["entity_id"])
    if q in val_gt:
        baseline_cands[q].add(str(row["target_id"]))

cols = ["entity_id", "business_name", "business_address", "country"]
s1_all = pl.read_csv(S1_PATH, separator="\t", quote_char=None, has_header=True, infer_schema_length=0, columns=cols)
s2 = pl.read_csv(S2_PATH, separator="\t", quote_char=None, has_header=True, infer_schema_length=0, columns=cols)
s3 = pl.read_csv(S3_PATH, separator="\t", quote_char=None, has_header=True, infer_schema_length=0, columns=cols)
targets_all = pl.concat([s2, s3])
val_s1 = s1_all.filter(pl.col("entity_id").is_in(val_query_list))
val_s1_rows = {str(r["entity_id"]): r for r in val_s1.iter_rows(named=True)}
target_rows = {str(r["entity_id"]): r for r in targets_all.iter_rows(named=True)}

all_tids = targets_all["entity_id"].to_list()
random.seed(42)
# Include all 172k positive targets + 600k random negatives = 772k targets!
neg_sample = set(random.sample(all_tids, min(600_000, len(all_tids))))
tfidf_tids = list(all_pos_tids | neg_sample)
flush(f"Targets corpus: {len(tfidf_tids):,} (172k pos + 600k neg), Queries: {len(val_query_list):,}")

def clean_text(r):
    nm = str(r.get("business_name", "") or "").lower()
    ad = str(r.get("business_address", "") or "").lower()
    return f"{nm} {ad}".strip()

tgt_texts = [clean_text(target_rows.get(tid, {})) for tid in tfidf_tids]
qry_texts = [clean_text(val_s1_rows.get(qid, {})) for qid in val_query_list]

flush("\nVectorizing corpus...")
t1 = time.time()
vec = TfidfVectorizer(
    analyzer='char_wb', ngram_range=(3, 4),
    max_features=150_000, sublinear_tf=True,
    min_df=3, max_df=0.03, strip_accents='unicode'
)
tgt_mat = vec.fit_transform(tgt_texts)
qry_mat = vec.transform(qry_texts)
tgt_mat_norm = normalize(tgt_mat, norm='l2', copy=False).tocsc()
qry_mat_norm = normalize(qry_mat, norm='l2', copy=False)
flush(f"Vectorized in {time.time()-t1:.2f}s. Matrix non-zeros: {tgt_mat.nnz:,}")

BATCH = 500
K = 60
t_search = time.time()
new_pos_found = 0
total_found = 0
tid_arr = np.array(tfidf_tids)

for b in range(0, len(val_query_list), BATCH):
    end = min(b + BATCH, len(val_query_list))
    q_batch = qry_mat_norm[b:end]
    scores = q_batch.dot(tgt_mat_norm.T) # CSR matrix of shape (BATCH x 772k)
    
    # Process each query's sparse scores
    for i in range(scores.shape[0]):
        row = scores.getrow(i)
        if row.nnz == 0:
            continue
        data = row.data
        indices = row.indices
        if len(data) > K:
            top_k_idx = np.argpartition(data, -K)[-K:]
            top_indices = indices[top_k_idx]
            top_scores = data[top_k_idx]
        else:
            top_indices = indices
            top_scores = data
        
        qid = val_query_list[b + i]
        gt_t = val_gt.get(qid, set())
        base_t = baseline_cands.get(qid, set())
        for idx, sc in zip(top_indices, top_scores):
            if sc > 0.15:
                tid = tid_arr[idx]
                if tid in gt_t:
                    total_found += 1
                    if tid not in base_t:
                        new_pos_found += 1
    
    if (b // BATCH) % 2 == 0:
        flush(f"  Processed {end}/{len(val_query_list)} queries in {time.time()-t_search:.1f}s...")

flush(f"\nCompleted {len(val_query_list)} queries against 772k targets in {time.time()-t_search:.2f}s!")
flush(f"Found {total_found} true matches, including {new_pos_found} NEW positive matches missed by baseline!")
