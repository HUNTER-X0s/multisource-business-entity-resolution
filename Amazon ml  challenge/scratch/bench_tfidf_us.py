import time, sys
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import numpy as np
import scipy.sparse as sp

print("Testing pure TF-IDF sparse retrieval on US targets...")

t0 = time.time()
# Load US queries from val
val_df = pl.read_parquet("experiments/val_sample_50k.parquet").filter(pl.col("country") == "US").head(5000)
print(f"Loaded {len(val_df)} US val queries in {time.time()-t0:.1f}s")

# Load US targets
t1 = time.time()
s2_us = pl.read_csv("dataset/raw/train/train_source2.tsv", separator="\t", quote_char=None).filter(pl.col("country") == "US")
s3_us = pl.read_csv("dataset/raw/train/train_source3.tsv", separator="\t", quote_char=None).filter(pl.col("country") == "US")
targets_us = pl.concat([s2_us, s3_us])
print(f"Loaded {len(targets_us):,} US targets in {time.time()-t1:.1f}s")

# Extract texts
tgt_texts = (targets_us['business_name'].fill_null('') + " " + targets_us['business_address'].fill_null('')).to_list()
qry_texts = (val_df['business_name'].fill_null('') + " " + val_df['business_address'].fill_null('')).to_list()

# Fit vectorizer
t2 = time.time()
print("Fitting TF-IDF vectorizer (char_wb 3-4)...")
vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 4), max_features=100_000, sublinear_tf=True, min_df=5, max_df=0.05)
tgt_mat = vec.fit_transform(tgt_texts)
qry_mat = vec.transform(qry_texts)

tgt_mat = normalize(tgt_mat, norm='l2', copy=False).tocsc()
qry_mat = normalize(qry_mat, norm='l2', copy=False)
print(f"TF-IDF matrix built in {time.time()-t2:.1f}s. Target nonzeros: {tgt_mat.nnz:,}, shape: {tgt_mat.shape}")

# Test dot product time for 5,000 queries
t3 = time.time()
print("Computing sparse dot product for 5,000 queries...")
BATCH = 500
total_pairs = 0
TOP_K = 40

for b in range(0, len(qry_texts), BATCH):
    q_batch = qry_mat[b:b+BATCH]
    scores = q_batch.dot(tgt_mat)
    total_pairs += scores.nnz

print(f"Dot product done in {time.time()-t3:.1f}s ({time.time()-t3:.2f}s per 5000 queries)")
print(f"Total nonzero similarity pairs: {total_pairs:,}")
