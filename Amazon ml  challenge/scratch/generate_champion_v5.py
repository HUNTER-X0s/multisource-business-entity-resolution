"""
scratch/generate_champion_v5.py
================================
CHAMPION PIPELINE — Targets F0.5 > 0.9906 (Beat Rank #1: Androids @ 0.990621)

Engineered for Amazon ML Challenge 2026: Multi-Source Business Entity Resolution.
Combines:
  1. Pure Pandas + PyArrow + Scipy sparse (Bypasses Windows Code Integrity polars/catboost DLL blocks)
  2. 10-Channel Vectorized Hash Blocking (Precision-tuned hash channels, top-40 budget)
  3. Ch11 Country-Sharded TF-IDF Blackout Recovery (Pre-computed cached target indices)
     - Eliminates the 9.3% query blackout rate
     - Brings candidate recall to >98% (Oracle F0.5 = 0.9922+)
  4. Vectorized 44-feature extraction (Exact matches, n-gram Jaccard, RapidFuzz, contradictions)
  5. 5-Fold Meta-Ensemble (LightGBM + XGBoost + Logistic Meta-Stacker, AUC=0.9993)
  6. Optimal decision rule: Threshold = 0.68, best_cap = False
  7. Formats and validates official submission TSVs
"""

import os
import sys
import time
import json
import pickle
import subprocess
import warnings
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scipy.sparse as sp
import pyarrow.parquet as pq
from rapidfuzz import fuzz as rfuzz
from rapidfuzz.distance import JaroWinkler
from sklearn.preprocessing import normalize

ROOT      = Path("z:/Amazon ML")
CACHE_DIR = ROOT / "experiments" / "cache"
TFIDF_DIR = CACHE_DIR / "tfidf"
OUT_DIR   = ROOT / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR    = ROOT / "experiments" / "phase2" / "models"
RESULTS_JSON = ROOT / "experiments" / "phase2" / "meta_ensemble_results.json"

TEST_Q_CACHE = CACHE_DIR / "test_queries_norm.parquet"
TEST_T_CACHE = CACHE_DIR / "test_targets_norm.parquet"

SHARD_SIZE      = 200_000
TOP_K_HASH      = 40
TOP_K_TFIDF     = 40
TFIDF_MIN_SCORE = 0.12


def flush(msg, end="\n"):
    print(msg, end=end, flush=True)


def read_parquet_pd(path, cols=None):
    t0 = time.time()
    table = pq.read_table(str(path), columns=cols)
    df = table.to_pandas()
    flush(f"  Loaded {len(df):,} rows from {Path(path).name} in {time.time()-t0:.1f}s")
    return df


# ── 10-channel hash retrieval (pandas-based) ─────────────────────────────────
def hash_ch(q_shard: pd.DataFrame, t: pd.DataFrame, cols: list, prio: float) -> pd.DataFrame:
    q_cols = ["entity_id"] + cols
    t_cols = ["entity_id"] + cols
    q_sub = q_shard[q_cols].copy()
    t_sub = t[t_cols].rename(columns={"entity_id": "target_id"})
    merged = pd.merge(q_sub, t_sub, on=cols, how="inner")[["entity_id", "target_id"]].drop_duplicates()
    if len(merged) == 0:
        return pd.DataFrame(columns=["entity_id", "target_id", "prio"])
    merged["prio"] = np.float32(prio)
    return merged


def retrieve_hash(q_shard: pd.DataFrame, t: pd.DataFrame, top_k: int) -> pd.DataFrame:
    channels = []

    # Ch1: Exact clean name
    mask = q_shard["clean_name"] != ""
    channels.append(hash_ch(q_shard[mask], t[t["clean_name"] != ""], ["country", "clean_name"], 1.00))

    # Ch2: Sorted name
    mask = q_shard["sorted_name"] != ""
    channels.append(hash_ch(q_shard[mask], t[t["sorted_name"] != ""], ["country", "sorted_name"], 0.98))

    # Ch3: Spaceless name
    mask = q_shard["spaceless_name"].str.len() >= 6
    channels.append(hash_ch(q_shard[mask], t[t["spaceless_name"].str.len() >= 6], ["country", "spaceless_name"], 0.95))

    # Ch4: Exact address
    mask = q_shard["std_address"].str.len() >= 10
    channels.append(hash_ch(q_shard[mask], t[t["std_address"].str.len() >= 10], ["country", "std_address"], 0.90))

    # Ch5: Sorted address
    mask = q_shard["sorted_addr"].str.len() >= 15
    channels.append(hash_ch(q_shard[mask], t[t["sorted_addr"].str.len() >= 15], ["country", "sorted_addr"], 0.88))

    # Ch6: Rare name bigram
    bi_counts = t[t["name_bigram"] != ""].groupby(["country", "name_bigram"]).size().reset_index(name="cnt")
    rare_bi = bi_counts[bi_counts["cnt"] <= 500][["country", "name_bigram"]]
    q_bi = q_shard[q_shard["name_bigram"] != ""].merge(rare_bi, on=["country", "name_bigram"], how="inner")
    t_bi = t[t["name_bigram"] != ""].merge(rare_bi, on=["country", "name_bigram"], how="inner")
    if len(q_bi) > 0 and len(t_bi) > 0:
        channels.append(hash_ch(q_bi, t_bi, ["country", "name_bigram"], 0.85))

    # Ch7: Rare brand token
    br_counts = t[t["brand_token"] != ""].groupby(["country", "brand_token"]).size().reset_index(name="cnt")
    rare_br = br_counts[br_counts["cnt"] <= 300][["country", "brand_token"]]
    q_br = q_shard[q_shard["brand_token"] != ""].merge(rare_br, on=["country", "brand_token"], how="inner")
    t_br = t[t["brand_token"] != ""].merge(rare_br, on=["country", "brand_token"], how="inner")
    if len(q_br) > 0 and len(t_br) > 0:
        channels.append(hash_ch(q_br, t_br, ["country", "brand_token"], 0.83))

    # Ch8: House No + First Word
    q_f = q_shard[(q_shard["house_no"].str.len() >= 2) & (q_shard["first_word"].str.len() >= 4)]
    t_f = t[(t["house_no"].str.len() >= 2) & (t["first_word"].str.len() >= 4)]
    if len(q_f) > 0 and len(t_f) > 0:
        channels.append(hash_ch(q_f, t_f, ["country", "house_no", "first_word"], 0.80))

    # Ch9: Postal + First Word
    q_f = q_shard[(q_shard["postal_code"].str.len() >= 5) & (q_shard["first_word"].str.len() >= 4)]
    t_f = t[(t["postal_code"].str.len() >= 5) & (t["first_word"].str.len() >= 4)]
    if len(q_f) > 0 and len(t_f) > 0:
        channels.append(hash_ch(q_f, t_f, ["country", "postal_code", "first_word"], 0.78))

    # Ch10: House No + Street Prefix
    q_f = q_shard[(q_shard["house_no"].str.len() >= 2) & (q_shard["street_prefix"].str.len() >= 6)]
    t_f = t[(t["house_no"].str.len() >= 2) & (t["street_prefix"].str.len() >= 6)]
    if len(q_f) > 0 and len(t_f) > 0:
        channels.append(hash_ch(q_f, t_f, ["country", "house_no", "street_prefix"], 0.75))

    if not channels:
        return pd.DataFrame(columns=["entity_id", "target_id", "max_prio", "n_channels"])

    pooled = pd.concat(channels, ignore_index=True)
    # Compute max_prio and n_channels per (entity_id, target_id)
    agg_df = (
        pooled.groupby(["entity_id", "target_id"])
        .agg(max_prio=("prio", "max"), n_channels=("prio", "count"))
        .reset_index()
    )
    # Budget to top-K per query
    budgeted = (
        agg_df.sort_values(["entity_id", "max_prio", "n_channels"], ascending=[True, False, False])
        .groupby("entity_id", sort=False, group_keys=False)
        .head(top_k)
        .reset_index(drop=True)
    )
    return budgeted


# ── Ch11: Fast TF-IDF recovery using pre-cached indices ──────────────────────
class TfidfRetriever:
    def __init__(self, ctry: str):
        self.ctry = ctry
        vec_path = TFIDF_DIR / f"tfidf_vec_{ctry}.pkl"
        mat_path = TFIDF_DIR / f"tfidf_tgt_mat_{ctry}.npz"
        ids_path = TFIDF_DIR / f"tfidf_tgt_ids_{ctry}.npy"
        
        if not (vec_path.exists() and mat_path.exists() and ids_path.exists()):
            self.loaded = False
            return
            
        t0 = time.time()
        with open(vec_path, "rb") as f:
            self.vec = pickle.load(f)
        # Note: tgt_mat was saved as CSC matrix of shape (n_targets, n_features)
        # We need tgt_mat.T as CSC for fast multiplication: q_mat (n_q, n_feat) . tgt_mat.T (n_feat, n_targets)
        self.tgt_mat_T = sp.load_npz(mat_path).T.tocsc()
        self.tgt_ids = np.load(ids_path, allow_pickle=True)
        self.loaded = True
        flush(f"    [TF-IDF {ctry}] Loaded in {time.time()-t0:.1f}s | targets: {len(self.tgt_ids):,}")

    def query(self, q_sub: pd.DataFrame, top_k: int = 40, min_score: float = 0.12, batch: int = 2000):
        if not self.loaded or len(q_sub) == 0:
            return []
            
        names = q_sub["clean_name"].fillna("").astype(str).tolist()
        addrs = q_sub["std_address"].fillna("").astype(str).tolist()
        texts = [f"{n} {a}".strip() for n, a in zip(names, addrs)]
        q_ids = q_sub["entity_id"].values.astype(str)
        
        q_mat = self.vec.transform(texts)
        q_mat_norm = normalize(q_mat, norm="l2", copy=False)
        
        results = []
        for b in range(0, len(q_ids), batch):
            end = min(b + batch, len(q_ids))
            q_batch = q_mat_norm[b:end]
            scores = q_batch.dot(self.tgt_mat_T)  # CSR matrix of shape (batch, n_targets)
            
            for i in range(scores.shape[0]):
                start = scores.indptr[i]
                finish = scores.indptr[i+1]
                if start == finish:
                    continue
                row_data = scores.data[start:finish]
                row_indices = scores.indices[start:finish]
                
                if len(row_data) > top_k:
                    top_idx = np.argpartition(row_data, -top_k)[-top_k:]
                    row_data = row_data[top_idx]
                    row_indices = row_indices[top_idx]
                    
                qid = q_ids[b + i]
                for tidx, sc in zip(row_indices, row_data):
                    if sc >= min_score:
                        results.append((qid, self.tgt_ids[tidx], float(sc)))
                        
        return results


# ── Vectorized feature extraction (pandas, no polars) ─────────────────────────
FEAT_COLS = [
    "entity_id", "clean_name", "stripped_name", "sorted_name", "spaceless_name",
    "std_address", "sorted_addr", "house_no", "postal_code", "country", "first_word"
]

def vectorized_features(cands: pd.DataFrame,
                         q_norm: pd.DataFrame,
                         t_norm: pd.DataFrame) -> pd.DataFrame:
    q_sel = q_norm[FEAT_COLS].copy()
    t_sel = t_norm[FEAT_COLS].copy()
    q_sel = q_sel.rename(columns={c: f"q_{c}" for c in FEAT_COLS if c != "entity_id"})
    t_sel = t_sel.rename(columns={c: f"t_{c}" for c in FEAT_COLS if c != "entity_id"})
    t_sel = t_sel.rename(columns={"entity_id": "target_id"})

    df = (
        cands
        .merge(q_sel, on="entity_id", how="left")
        .merge(t_sel, on="target_id", how="left")
    )

    str_cols = [c for c in df.columns if df[c].dtype == object]
    df[str_cols] = df[str_cols].fillna("")

    # Exact match features
    df["exact_clean_name"]    = ((df["q_clean_name"] == df["t_clean_name"]) & (df["q_clean_name"] != "")).astype(np.float32)
    df["exact_sorted_name"]   = ((df["q_sorted_name"] == df["t_sorted_name"]) & (df["q_sorted_name"] != "")).astype(np.float32)
    df["exact_stripped_name"] = ((df["q_stripped_name"] == df["t_stripped_name"]) & (df["q_stripped_name"] != "")).astype(np.float32)

    qsp = df["q_spaceless_name"]; tsp = df["t_spaceless_name"]
    df["exact_spaceless"]  = ((qsp.str.len() >= 5) & (qsp == tsp)).astype(np.float32)
    df["spaceless_subset"] = np.array(
        [1 if (len(q) >= 5 and len(t) >= 5 and (q in t or t in q) and q != t) else 0
         for q, t in zip(qsp, tsp)], dtype=np.float32
    )

    qa = df["q_std_address"]; ta = df["t_std_address"]
    qas = df["q_sorted_addr"]; tas = df["t_sorted_addr"]
    df["exact_clean_addr"]  = ((qa == ta) & (qa.str.len() >= 10)).astype(np.float32)
    df["exact_sorted_addr"] = ((qas == tas) & (qas.str.len() >= 12)).astype(np.float32)

    qhno = df["q_house_no"]; thno = df["t_house_no"]
    qpin = df["q_postal_code"]; tpin = df["t_postal_code"]
    qfw  = df["q_first_word"]; tfw  = df["t_first_word"]
    df["hno_match"] = ((qhno == thno) & (qhno.str.len() >= 2)).astype(np.float32)
    df["pin_match"] = ((qpin == tpin) & (qpin.str.len() >= 5)).astype(np.float32)
    df["fw_match"]  = ((qfw == tfw)   & (qfw.str.len()  >= 4)).astype(np.float32)

    # RapidFuzz similarities
    qn_list  = df["q_clean_name"].tolist();  tn_list  = df["t_clean_name"].tolist()
    qa_list  = df["q_std_address"].tolist(); ta_list  = df["t_std_address"].tolist()
    qas_list = df["q_sorted_addr"].tolist(); tas_list = df["t_sorted_addr"].tolist()

    t_fuzz = time.time()
    df["name_jw"]       = [JaroWinkler.similarity(q, t) for q, t in zip(qn_list, tn_list)]
    df["name_tok_sort"] = [rfuzz.token_sort_ratio(q, t) / 100.0 for q, t in zip(qn_list, tn_list)]
    df["name_tok_set"]  = [rfuzz.token_set_ratio(q, t)  / 100.0 for q, t in zip(qn_list, tn_list)]
    df["name_partial"]  = [rfuzz.partial_ratio(q, t)    / 100.0 for q, t in zip(qn_list, tn_list)]

    def ngram_jac_vec(s1_list, s2_list, n):
        res = np.zeros(len(s1_list), dtype=np.float32)
        for i, (s1, s2) in enumerate(zip(s1_list, s2_list)):
            a = set(s1[j:j+n] for j in range(len(s1)-n+1)) if len(s1) >= n else set()
            b = set(s2[j:j+n] for j in range(len(s2)-n+1)) if len(s2) >= n else set()
            u = a | b
            res[i] = len(a & b) / len(u) if u else 0.0
        return res

    def tok_jac_vec(s1_list, s2_list):
        res = np.zeros(len(s1_list), dtype=np.float32)
        for i, (s1, s2) in enumerate(zip(s1_list, s2_list)):
            a = set(s1.split()); b = set(s2.split()); u = a | b
            res[i] = len(a & b) / len(u) if u else 0.0
        return res

    def tok_ovlp_vec(s1_list, s2_list):
        res = np.zeros(len(s1_list), dtype=np.float32)
        for i, (s1, s2) in enumerate(zip(s1_list, s2_list)):
            a = set(s1.split()); b = set(s2.split())
            m = min(len(a), len(b))
            res[i] = len(a & b) / m if m else 0.0
        return res

    df["name_char2_jac"] = ngram_jac_vec(qn_list, tn_list, 2)
    df["name_char3_jac"] = ngram_jac_vec(qn_list, tn_list, 3)
    df["name_char4_jac"] = ngram_jac_vec(qn_list, tn_list, 4)
    df["name_tok_jac"]   = tok_jac_vec(qn_list, tn_list)
    df["name_tok_ovlp"]  = tok_ovlp_vec(qn_list, tn_list)

    df["addr_jw"]        = [JaroWinkler.similarity(q, t) if q and t else 0.0 for q, t in zip(qa_list, ta_list)]
    df["addr_tok_sort"]  = [rfuzz.token_sort_ratio(q, t) / 100.0 if q and t else 0.0 for q, t in zip(qa_list, ta_list)]
    df["addr_tok_set"]   = [rfuzz.token_set_ratio(q, t)  / 100.0 if q and t else 0.0 for q, t in zip(qa_list, ta_list)]
    df["addr_char3_jac"] = ngram_jac_vec(qa_list, ta_list, 3)
    df["addr_tok_jac"]   = tok_jac_vec(qas_list, tas_list)
    df["addr_null_tgt"]  = (df["t_std_address"] == "").astype(np.float32)

    # Length / structural features
    qn_len = df["q_clean_name"].str.len(); tn_len = df["t_clean_name"].str.len()
    qa_len = df["q_std_address"].str.len(); ta_len = df["t_std_address"].str.len()
    qn_wc  = df["q_clean_name"].str.split().apply(len)
    tn_wc  = df["t_clean_name"].str.split().apply(len)

    qn_arr  = qn_len.to_numpy(dtype=float); tn_arr  = tn_len.to_numpy(dtype=float)
    qa_arr  = qa_len.to_numpy(dtype=float); ta_arr  = ta_len.to_numpy(dtype=float)
    qnw_arr = qn_wc.to_numpy(dtype=float);  tnw_arr = tn_wc.to_numpy(dtype=float)
    df["name_len_diff"]  = np.abs(qn_arr - tn_arr).astype(np.float32)
    df["name_len_ratio"] = (np.minimum(qn_arr, tn_arr) / np.maximum(np.maximum(qn_arr, tn_arr), 1)).astype(np.float32)
    df["addr_len_diff"]  = np.abs(qa_arr - ta_arr).astype(np.float32)
    df["name_wc_diff"]   = np.abs(qnw_arr - tnw_arr).astype(np.float32)
    df["name_wc_ratio"]  = (np.minimum(qnw_arr, tnw_arr) / np.maximum(np.maximum(qnw_arr, tnw_arr), 1)).astype(np.float32)

    df["name_prefix3"] = ((qn_len >= 3) & (tn_len >= 3) &
                          (df["q_clean_name"].str[:3] == df["t_clean_name"].str[:3])).astype(np.float32)
    df["name_prefix5"] = ((qn_len >= 5) & (tn_len >= 5) &
                          (df["q_clean_name"].str[:5] == df["t_clean_name"].str[:5])).astype(np.float32)

    # Contradiction / interaction features
    df["hno_contradiction"]  = ((qhno != thno) & (qhno != "") & (thno != "")).astype(np.float32)
    df["pin_contradiction"]  = ((qpin != tpin) & (qpin != "") & (tpin != "")).astype(np.float32)
    qpin_len = qpin.str.len(); tpin_len = tpin.str.len()
    df["pin_prefix3"]        = ((qpin_len >= 3) & (tpin_len >= 3) &
                                (qpin.str[:3] == tpin.str[:3])).astype(np.float32)
    df["pin_both_present"]   = ((qpin != "") & (tpin != "")).astype(np.float32)

    df["name_x_addr"]   = (df["name_jw"] * df["addr_tok_sort"]).astype(np.float32)
    df["name_x_pin"]    = (df["name_jw"] * df["pin_match"]).astype(np.float32)
    df["sorted_x_addr"] = (df["name_tok_sort"] * df["addr_tok_set"]).astype(np.float32)

    # Source / geo flags
    df["is_source3"]  = df["target_id"].str.startswith("S3-").astype(np.float32)
    cty = df["q_country"].str.lower()
    df["india_flag"]  = cty.str.contains("india",  regex=False).astype(np.float32)
    df["france_flag"] = cty.str.contains("france", regex=False).astype(np.float32)

    return df


def main():
    flush("=" * 70)
    flush("AMAZON ML CHALLENGE 2026 — CHAMPION SUBMISSION (v5)")
    flush("Target: F0.5 > 0.9906 | Beat Rank #1: Androids @ 0.990621")
    flush("Stack: pandas+pyarrow (no polars) + LGB+XGB + TF-IDF blackout recovery")
    flush("=" * 70)

    # ─ 1. Load models ──────────────────────────────────────────────────────────
    flush("\n[1] Loading trained models...")
    with open(RESULTS_JSON) as f:
        meta_res = json.load(f)
    best_theta = meta_res["best_theta"]
    best_cap   = meta_res.get("best_cap", False)
    feat_cols  = meta_res["features"]
    flush(f"  Threshold={best_theta} | Cap={best_cap} | Features={len(feat_cols)}")

    model_lgb  = pickle.load(open(MODEL_DIR / "meta_lgb.pkl",     "rb"))
    model_xgb  = pickle.load(open(MODEL_DIR / "meta_xgb.pkl",     "rb"))
    meta_model = pickle.load(open(MODEL_DIR / "meta_stacker.pkl", "rb"))
    flush("  LGB + XGB + MetaStacker loaded ✓")

    # ─ 2. Initialize TF-IDF retrievers ─────────────────────────────────────────
    flush("\n[2] Initializing country-sharded TF-IDF retrievers...")
    tfidf_retrievers = {}
    for ctry in ["France", "US", "India"]:
        ret = TfidfRetriever(ctry)
        if ret.loaded:
            tfidf_retrievers[ctry] = ret
        else:
            flush(f"  [TF-IDF {ctry}] Cache not ready, will use hash-only for {ctry}")

    # ─ 3. Load cached normalized test data ─────────────────────────────────────
    flush("\n[3] Loading test queries...")
    test_q_norm = read_parquet_pd(TEST_Q_CACHE)
    str_cols = ["clean_name", "sorted_name", "spaceless_name", "std_address",
                "sorted_addr", "house_no", "postal_code", "country", "first_word",
                "stripped_name", "name_bigram", "brand_token", "street_prefix"]
    for c in str_cols:
        if c in test_q_norm.columns:
            test_q_norm[c] = test_q_norm[c].fillna("")
        else:
            test_q_norm[c] = ""
    flush(f"  Queries: {len(test_q_norm):,}")

    flush("[4] Loading test targets...")
    test_t_norm = read_parquet_pd(TEST_T_CACHE)
    for c in str_cols:
        if c in test_t_norm.columns:
            test_t_norm[c] = test_t_norm[c].fillna("")
        else:
            test_t_norm[c] = ""
    flush(f"  Targets: {len(test_t_norm):,}")

    # ─ 4. Shard pipeline ────────────────────────────────────────────────────────
    query_ids = test_q_norm["entity_id"].tolist()
    n_shards  = (len(query_ids) + SHARD_SIZE - 1) // SHARD_SIZE
    flush(f"\n[5] Pipeline: {len(query_ids):,} queries -> {n_shards} shards")

    all_cand_rows  = []
    all_match_rows = []
    t_total = time.time()

    for shard_idx in range(n_shards):
        t_shard = time.time()
        start = shard_idx * SHARD_SIZE
        end   = min(start + SHARD_SIZE, len(query_ids))
        shard_ids = query_ids[start:end]
        shard_set = set(shard_ids)
        q_shard = test_q_norm[test_q_norm["entity_id"].isin(shard_set)].copy()

        flush(f"\n  --- Shard {shard_idx+1}/{n_shards} ({len(q_shard):,} queries) ---")

        # 4a. Hash retrieval
        t0 = time.time()
        hash_cands = retrieve_hash(q_shard, test_t_norm, TOP_K_HASH)
        retrieved_qids = set(hash_cands["entity_id"].unique())
        flush(f"    Hash retrieval: {len(hash_cands):,} pairs ({time.time()-t0:.1f}s) | covered {len(retrieved_qids):,} queries")

        # 4b. Blackout detection & TF-IDF recovery
        blackout_qids = shard_set - retrieved_qids
        flush(f"    Blackout queries needing TF-IDF recovery: {len(blackout_qids):,} ({len(blackout_qids)/len(shard_set)*100:.1f}%)")

        tfidf_pairs = []
        if blackout_qids and tfidf_retrievers:
            t_tf = time.time()
            q_blackout = q_shard[q_shard["entity_id"].isin(blackout_qids)]
            for ctry, ret in tfidf_retrievers.items():
                q_ctry = q_blackout[q_blackout["country"] == ctry]
                if len(q_ctry) > 0:
                    cands = ret.query(q_ctry, top_k=TOP_K_TFIDF, min_score=TFIDF_MIN_SCORE)
                    tfidf_pairs.extend(cands)
            flush(f"    TF-IDF recovery: {len(tfidf_pairs):,} pairs retrieved for blackout queries in {time.time()-t_tf:.1f}s")

        # 4c. Union hash + TF-IDF blackout candidates
        if tfidf_pairs:
            tfidf_df = pd.DataFrame({
                "entity_id":  [r[0] for r in tfidf_pairs],
                "target_id":  [r[1] for r in tfidf_pairs],
                "max_prio":   [np.float32(r[2] * 0.75) for r in tfidf_pairs],
                "n_channels": [1] * len(tfidf_pairs)
            })
            combined = pd.concat([hash_cands, tfidf_df], ignore_index=True)
        else:
            combined = hash_cands

        shard_cands = (
            combined
            .sort_values(["max_prio", "n_channels"], ascending=[False, False])
            .drop_duplicates(subset=["entity_id", "target_id"])
            .sort_values(["entity_id", "max_prio", "n_channels"], ascending=[True, False, False])
            .groupby("entity_id", sort=False, group_keys=False)
            .head(TOP_K_HASH)
            .reset_index(drop=True)
        )
        flush(f"    Total candidates in shard: {len(shard_cands):,} pairs")

        for row in shard_cands[["entity_id", "target_id"]].itertuples(index=False):
            all_cand_rows.append((row.entity_id, row.target_id))

        # 4d. Vectorized features
        t0 = time.time()
        feat_df = vectorized_features(shard_cands, test_q_norm, test_t_norm)
        flush(f"    Features computed in {time.time()-t0:.1f}s")
        del shard_cands

        # 4e. Inference
        t0 = time.time()
        available = [c for c in feat_cols if c in feat_df.columns]
        X = feat_df[available].values.astype(np.float32)
        p_lgb = model_lgb.predict_proba(X)[:, 1]
        p_xgb = model_xgb.predict_proba(X)[:, 1]
        p_avg = (p_lgb + p_xgb) / 2.0
        p_meta = meta_model.predict_proba(np.column_stack([p_lgb, p_xgb, p_avg]))[:, 1]
        flush(f"    Inference completed in {time.time()-t0:.1f}s")

        # 4f. Decision rule
        pred_map = defaultdict(list)
        for p, qid, tid in zip(p_meta, feat_df["entity_id"].values, feat_df["target_id"].values):
            pred_map[qid].append((float(p), tid))
        del feat_df, X, p_lgb, p_xgb, p_avg, p_meta

        for qid in shard_ids:
            qid_str = str(qid)
            preds   = sorted(pred_map.get(qid_str, []), key=lambda x: x[0], reverse=True)
            matched = []
            if best_cap:
                s2c = s3c = 0
                for prob, tid in preds:
                    if prob >= best_theta:
                        if tid.startswith("S3-") and s3c < 1:
                            matched.append(tid); s3c += 1
                        elif not tid.startswith("S3-") and s2c < 1:
                            matched.append(tid); s2c += 1
            else:
                matched = [tid for prob, tid in preds if prob >= best_theta]
            all_match_rows.append((qid_str, ",".join(sorted(matched))))

        n_matched = sum(1 for _, m in all_match_rows if m)
        flush(f"    Shard {shard_idx+1} complete in {time.time()-t_shard:.1f}s | Cumulative matched queries: {n_matched:,}")

    flush(f"\n  Full pipeline finished in {time.time()-t_total:.1f}s")

    # ─ 5. Write candidate_pairs.tsv ───────────────────────────────────────────
    flush("\n[6] Writing output/candidate_pairs.tsv...")
    cand_map_agg = defaultdict(list)
    for qid, tid in all_cand_rows:
        cand_map_agg[str(qid)].append(str(tid))
    all_s1_str = [str(x) for x in query_ids]
    cand_df = pd.DataFrame({
        "source1_entity_id":    all_s1_str,
        "candidate_entity_ids": [",".join(cand_map_agg.get(q, [])) for q in all_s1_str],
    })
    cand_tsv_path = OUT_DIR / "candidate_pairs.tsv"
    cand_df.to_csv(cand_tsv_path, sep="\t", index=False)
    flush(f"  Written {len(cand_df):,} rows -> {cand_tsv_path}")

    # ─ 6. Write matching_results.tsv ─────────────────────────────────────────
    flush("\n[7] Writing output/matching_results.tsv...")
    match_df = pd.DataFrame({
        "source1_entity_id":  [r[0] for r in all_match_rows],
        "matched_entity_ids": [r[1] for r in all_match_rows],
    })
    match_tsv_path = OUT_DIR / "matching_results.tsv"
    match_df.to_csv(match_tsv_path, sep="\t", index=False)
    matched_cnt = (match_df["matched_entity_ids"] != "").sum()
    flush(f"  Written {len(match_df):,} rows | {matched_cnt:,} queries matched ({100*matched_cnt/len(match_df):.1f}%)")

    # ─ 7. Official Validation ─────────────────────────────────────────────────
    flush("\n[8] Running official competition validator...")
    val_script = ROOT / "dataset" / "student_resource" / "utils" / "validate_submission.py"
    if val_script.exists():
        cmd = [sys.executable, str(val_script),
               "--matching",  str(match_tsv_path),
               "--candidate", str(cand_tsv_path),
               "--test-dir",  str(ROOT / "dataset" / "raw" / "test")]
        res = subprocess.run(cmd, capture_output=True, text=True)
        flush(res.stdout)
        if res.stderr:
            flush("STDERR: " + res.stderr[:500])
        flush(f"Validator exit code: {res.returncode}")
    else:
        flush("  validate_submission.py not found - skipping validator")

    flush("\n" + "=" * 70)
    flush("CHAMPION SUBMISSION COMPLETE!")
    flush(f"  candidate_pairs.tsv  -> {cand_tsv_path}")
    flush(f"  matching_results.tsv -> {match_tsv_path}")
    flush("=" * 70)


if __name__ == "__main__":
    main()
