"""
scratch/build_target_tfidf_cache.py
===================================
Pre-computes and caches country-level char (3,4)-gram TF-IDF matrices
for the 9,969,589 test targets.

Countries: France (1.43M), US (3.82M), India (4.72M)
Outputs saved to: experiments/cache/tfidf/
  - tfidf_vec_{ctry}.pkl
  - tfidf_tgt_mat_{ctry}.npz
  - tfidf_tgt_ids_{ctry}.npy
"""
import time
import pickle
import numpy as np
import scipy.sparse as sp
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import pyarrow.parquet as pq

ROOT = Path("z:/Amazon ML")
CACHE_DIR = ROOT / "experiments" / "cache" / "tfidf"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
TARGETS_PARQUET = ROOT / "experiments" / "cache" / "test_targets_norm.parquet"

def flush(msg):
    print(msg, flush=True)

def build_country(ctry: str, t_df):
    out_vec = CACHE_DIR / f"tfidf_vec_{ctry}.pkl"
    out_mat = CACHE_DIR / f"tfidf_tgt_mat_{ctry}.npz"
    out_ids = CACHE_DIR / f"tfidf_tgt_ids_{ctry}.npy"
    
    if out_vec.exists() and out_mat.exists() and out_ids.exists():
        flush(f"[{ctry}] Already cached! Skipping.")
        return

    flush(f"\n[{ctry}] Building TF-IDF index for {len(t_df):,} targets...")
    t0 = time.time()
    
    names = t_df["clean_name"].fillna("").astype(str).tolist()
    addrs = t_df["std_address"].fillna("").astype(str).tolist()
    texts = [f"{n} {a}".strip() for n, a in zip(names, addrs)]
    t_ids = t_df["entity_id"].values.astype(str)
    
    flush(f"  Texts extracted in {time.time()-t0:.1f}s. Fitting TfidfVectorizer...")
    t1 = time.time()
    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 4),
        max_features=50_000,
        sublinear_tf=True,
        min_df=5,
        max_df=0.05,
        dtype=np.float32
    )
    tgt_mat = vec.fit_transform(texts)
    flush(f"  Fitted in {time.time()-t1:.1f}s | shape: {tgt_mat.shape} | nonzeros: {tgt_mat.nnz:,}")
    
    t2 = time.time()
    tgt_mat_norm = normalize(tgt_mat, norm="l2", copy=False).tocsc()
    flush(f"  Normalized & CSC converted in {time.time()-t2:.1f}s")
    
    t3 = time.time()
    with open(out_vec, "wb") as f:
        pickle.dump(vec, f)
    sp.save_npz(out_mat, tgt_mat_norm)
    np.save(out_ids, t_ids)
    flush(f"  Saved artifacts in {time.time()-t3:.1f}s -> {CACHE_DIR}")
    flush(f"  Total time for [{ctry}]: {time.time()-t0:.1f}s")

def main():
    flush("=" * 60)
    flush("PRE-COMPUTING TARGET TF-IDF MATRICES (Country-Sharded)")
    flush("=" * 60)
    
    t_start = time.time()
    flush("Loading test_targets_norm.parquet...")
    table = pq.read_table(str(TARGETS_PARQUET), columns=["entity_id", "clean_name", "std_address", "country"])
    df = table.to_pandas()
    flush(f"Loaded {len(df):,} targets in {time.time()-t_start:.1f}s")
    
    # Process in order of size: France, US, India
    for ctry in ["France", "US", "India"]:
        c_df = df[df["country"] == ctry]
        build_country(ctry, c_df)
        
    flush("\n" + "=" * 60)
    flush(f"ALL COUNTRY TF-IDF MATRICES BUILT & CACHED IN {time.time()-t_start:.1f}s!")
    flush("=" * 60)

if __name__ == "__main__":
    main()
