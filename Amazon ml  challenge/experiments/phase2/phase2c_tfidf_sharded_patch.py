"""
experiments/phase2/phase2c_tfidf_sharded_patch.py
===================================================
FALLBACK: Sharded TF-IDF retrieval by country slice.

If the full-corpus TF-IDF OOMs, this script patches the pipeline
to run TF-IDF per-country so each matrix is much smaller.

Also reduces max_features to 80K to fit in 6GB VRAM.
"""

# This is a drop-in replacement for the run_tfidf_channel call in the pipeline.
# Instead of one 10.3M corpus, shard by country.

def run_tfidf_channel_sharded(q_df, t_df, top_k=60, batch_size=500, min_score=0.12):
    """Country-sharded TF-IDF: smaller matrices, lower memory."""
    import time
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize
    import numpy as np
    import polars as pl

    flush(f"  Sharded TF-IDF on {len(q_df):,} queries...")
    t0 = time.time()
    all_cands = []

    countries = q_df["country"].unique().to_list()
    flush(f"  Countries: {len(countries)}")

    for ctry in countries:
        q_c = q_df.filter(pl.col("country") == ctry)
        t_c = t_df.filter(pl.col("country") == ctry)
        if len(q_c) == 0 or len(t_c) == 0:
            continue

        flush(f"    [{ctry}] Q={len(q_c):,} T={len(t_c):,}", end=" ")
        tc = time.time()

        tgt_texts = [f"{r['clean_name']} {r['std_address']}".lower().strip()
                     for r in t_c.select(['entity_id','clean_name','std_address']).iter_rows(named=True)]
        qry_texts = [f"{r['clean_name']} {r['std_address']}".lower().strip()
                     for r in q_c.select(['entity_id','clean_name','std_address']).iter_rows(named=True)]
        q_ids = q_c['entity_id'].to_list()
        t_ids = np.array(t_c['entity_id'].to_list())

        # Adaptive max_features based on corpus size
        mf = min(80_000, max(5_000, len(t_c) // 10))
        vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 4),
                              max_features=mf, sublinear_tf=True,
                              min_df=2, max_df=0.05, strip_accents='unicode')
        try:
            tgt_mat = vec.fit_transform(tgt_texts)
            qry_mat = vec.transform(qry_texts)
        except Exception as e:
            flush(f" SKIP ({e})")
            continue

        tgt_mat = normalize(tgt_mat, norm='l2', copy=False).tocsc()
        qry_mat = normalize(qry_mat, norm='l2', copy=False)

        ctry_cands = []
        for b in range(0, len(q_ids), batch_size):
            end = min(b + batch_size, len(q_ids))
            scores = qry_mat[b:end].dot(tgt_mat)
            for i in range(scores.shape[0]):
                rd = scores.data[scores.indptr[i]:scores.indptr[i+1]]
                ri = scores.indices[scores.indptr[i]:scores.indptr[i+1]]
                if len(rd) == 0: continue
                if len(rd) > top_k:
                    top_i = np.argpartition(rd, -top_k)[-top_k:]
                    rd, ri = rd[top_i], ri[top_i]
                qid = q_ids[b + i]
                for ridx, sc in zip(ri, rd):
                    if sc >= min_score:
                        ctry_cands.append((qid, t_ids[ridx], float(sc)))

        flush(f"-> {len(ctry_cands):,} pairs ({time.time()-tc:.1f}s)")
        all_cands.extend(ctry_cands)

    flush(f"  Sharded TF-IDF total: {len(all_cands):,} pairs in {time.time()-t0:.1f}s")
    return all_cands
