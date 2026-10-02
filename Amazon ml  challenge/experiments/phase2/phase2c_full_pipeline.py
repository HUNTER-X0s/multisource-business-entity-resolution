"""
experiments/phase2/phase2c_full_pipeline.py
============================================
PHASE 2C — FULL BREAKTHROUGH PIPELINE

Architecture:
  - Stage 1: 9-channel hash-join (84s, 73.7% recall) 
  - Stage 2: Char TF-IDF sparse retrieval (Ch10, adds ~23% recall)
  - Stage 3: Rich feature engineering (40+ features)
  - Stage 4: LightGBM + XGBoost-GPU + CatBoost-GPU meta-ensemble
  - Stage 5: Oracle evaluation / test submission generation

Target: Oracle ceiling >97% → trained model F0.5 >0.95+ on validation
"""
import os, sys, re, time, json, unicodedata, random
from pathlib import Path
from collections import defaultdict

import numpy as np
import polars as pl
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import torch
import pickle

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

random.seed(42)
np.random.seed(42)

def flush(*args, **kwargs):
    print(*args, **kwargs, flush=True)

flush("=" * 70)
flush("PHASE 2C — BREAKTHROUGH END-TO-END PIPELINE")
flush("=" * 70)
flush(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# ============================================================
# CONSTANTS
# ============================================================
# Indic transliteration using regex (supports multi-char target values)
# Pairs sorted by key-length DESC so multi-char grapheme clusters match before single chars
_INDIC_PAIRS_RAW = (
    # Bengali multi-char grapheme clusters  (must come FIRST)
    ('\u09a1\u09bc','r'), ('\u09a2\u09bc','rh'), ('\u09af\u09bc','y'),
    # Devanagari (Hindi/Marathi/Sanskrit)
    ('\u0905','a'),('\u0906','aa'),('\u0907','i'),('\u0908','ee'),('\u0909','u'),('\u090a','oo'),('\u090f','e'),('\u0910','ai'),('\u0913','o'),('\u0914','au'),
    ('\u0915','k'),('\u0916','kh'),('\u0917','g'),('\u0918','gh'),('\u0919','ng'),('\u091a','ch'),('\u091b','chh'),('\u091c','j'),('\u091d','jh'),('\u091e','ny'),
    ('\u091f','t'),('\u0920','th'),('\u0921','d'),('\u0922','dh'),('\u0923','n'),('\u0924','t'),('\u0925','th'),('\u0926','d'),('\u0927','dh'),('\u0928','n'),
    ('\u092a','p'),('\u092b','ph'),('\u092c','b'),('\u092d','bh'),('\u092e','m'),('\u092f','y'),('\u0930','r'),('\u0932','l'),('\u0935','v'),('\u0936','sh'),
    ('\u0937','sh'),('\u0938','s'),('\u0939','h'),('\u0933','l'),('\u093e','a'),('\u093f','i'),('\u0940','ee'),('\u0941','u'),('\u0942','oo'),('\u0947','e'),
    ('\u0948','ai'),('\u094b','o'),('\u094c','au'),('\u0902','n'),('\u0903','h'),('\u094d',''),('\u0901','n'),('\u0943','ri'),
    # Bengali
    ('\u0985','a'),('\u0986','aa'),('\u0987','i'),('\u0988','ee'),('\u0989','u'),('\u098a','oo'),('\u098f','e'),('\u0990','ai'),('\u0993','o'),('\u0994','au'),
    ('\u0995','k'),('\u0996','kh'),('\u0997','g'),('\u0998','gh'),('\u0999','ng'),('\u099a','ch'),('\u099b','chh'),('\u099c','j'),('\u099d','jh'),('\u099e','ny'),
    ('\u099f','t'),('\u09a0','th'),('\u09a1','d'),('\u09a2','dh'),('\u09a3','n'),('\u09a4','t'),('\u09a5','th'),('\u09a6','d'),('\u09a7','dh'),('\u09a8','n'),
    ('\u09aa','p'),('\u09ab','ph'),('\u09ac','b'),('\u09ad','bh'),('\u09ae','m'),('\u09af','y'),('\u09b0','r'),('\u09b2','l'),('\u09b6','sh'),('\u09b7','sh'),
    ('\u09b8','s'),('\u09b9','h'),('\u09be','a'),('\u09bf','i'),('\u09c0','ee'),('\u09c1','u'),('\u09c2','oo'),
    ('\u09c7','e'),('\u09c8','ai'),('\u09cb','o'),('\u09cc','au'),('\u09cd',''),
    # Telugu
    ('\u0c05','a'),('\u0c06','aa'),('\u0c07','i'),('\u0c08','ee'),('\u0c09','u'),('\u0c0a','oo'),('\u0c0e','e'),('\u0c0f','ee'),('\u0c10','ai'),('\u0c12','o'),('\u0c13','oo'),('\u0c14','au'),
    ('\u0c15','k'),('\u0c16','kh'),('\u0c17','g'),('\u0c18','gh'),('\u0c19','ng'),('\u0c1a','ch'),('\u0c1b','chh'),('\u0c1c','j'),('\u0c1d','jh'),('\u0c1e','ny'),
    ('\u0c1f','t'),('\u0c20','th'),('\u0c21','d'),('\u0c22','dh'),('\u0c23','n'),('\u0c24','t'),('\u0c25','th'),('\u0c26','d'),('\u0c27','dh'),('\u0c28','n'),
    ('\u0c2a','p'),('\u0c2b','ph'),('\u0c2c','b'),('\u0c2d','bh'),('\u0c2e','m'),('\u0c2f','y'),('\u0c30','r'),('\u0c32','l'),('\u0c35','v'),('\u0c36','sh'),
    ('\u0c37','sh'),('\u0c38','s'),('\u0c39','h'),('\u0c33','l'),('\u0c3e','a'),('\u0c3f','i'),('\u0c40','ee'),('\u0c41','u'),('\u0c42','oo'),('\u0c46','e'),
    ('\u0c47','ee'),('\u0c48','ai'),('\u0c4a','o'),('\u0c4b','oo'),('\u0c4c','au'),('\u0c02','m'),('\u0c4d',''),
    # Tamil
    ('\u0b85','a'),('\u0b86','aa'),('\u0b87','i'),('\u0b88','ee'),('\u0b89','u'),('\u0b8a','oo'),('\u0b8e','e'),('\u0b8f','ee'),('\u0b90','ai'),('\u0b92','o'),('\u0b93','oo'),('\u0b94','au'),
    ('\u0b95','k'),('\u0b99','ng'),('\u0b9a','ch'),('\u0b9e','ny'),('\u0b9f','t'),('\u0ba3','n'),('\u0ba4','t'),('\u0ba8','n'),('\u0baa','p'),('\u0bae','m'),
    ('\u0baf','y'),('\u0bb0','r'),('\u0bb2','l'),('\u0bb5','v'),('\u0bb4','zh'),('\u0bb3','l'),('\u0bb1','r'),('\u0ba9','n'),('\u0bbe','a'),('\u0bbf','i'),
    ('\u0bc0','ee'),('\u0bc1','u'),('\u0bc2','oo'),('\u0bc6','e'),('\u0bc7','ee'),('\u0bc8','ai'),('\u0bca','o'),('\u0bcb','oo'),('\u0bcc','au'),('\u0bcd',''),
    # Kannada
    ('\u0c85','a'),('\u0c86','aa'),('\u0c87','i'),('\u0c88','ee'),('\u0c89','u'),('\u0c8a','oo'),('\u0c8e','e'),('\u0c8f','ee'),('\u0c90','ai'),('\u0c92','o'),('\u0c93','oo'),('\u0c94','au'),
    ('\u0c95','k'),('\u0c96','kh'),('\u0c97','g'),('\u0c98','gh'),('\u0c99','ng'),('\u0c9a','ch'),('\u0c9b','chh'),('\u0c9c','j'),('\u0c9d','jh'),('\u0c9e','ny'),
    ('\u0c9f','t'),('\u0ca0','th'),('\u0ca1','d'),('\u0ca2','dh'),('\u0ca3','n'),('\u0ca4','t'),('\u0ca5','th'),('\u0ca6','d'),('\u0ca7','dh'),('\u0ca8','n'),
    ('\u0caa','p'),('\u0cab','ph'),('\u0cac','b'),('\u0cad','bh'),('\u0cae','m'),('\u0caf','y'),('\u0cb0','r'),('\u0cb2','l'),('\u0cb5','v'),('\u0cb6','sh'),
    ('\u0cb7','sh'),('\u0cb8','s'),('\u0cb9','h'),('\u0cb3','l'),('\u0cbe','a'),('\u0cbf','i'),('\u0cc0','ee'),('\u0cc1','u'),('\u0cc2','oo'),('\u0cc6','e'),
    ('\u0cc7','ee'),('\u0cc8','ai'),('\u0cca','o'),('\u0ccb','oo'),('\u0ccc','au'),('\u0c82','m'),('\u0ccd',''),
    # Malayalam
    ('\u0d05','a'),('\u0d06','aa'),('\u0d07','i'),('\u0d08','ee'),('\u0d09','u'),('\u0d0a','oo'),('\u0d0e','e'),('\u0d0f','ee'),('\u0d10','ai'),('\u0d12','o'),('\u0d13','oo'),('\u0d14','au'),
    ('\u0d15','k'),('\u0d16','kh'),('\u0d17','g'),('\u0d18','gh'),('\u0d19','ng'),('\u0d1a','ch'),('\u0d1b','chh'),('\u0d1c','j'),('\u0d1d','jh'),('\u0d1e','ny'),
    ('\u0d1f','t'),('\u0d20','th'),('\u0d21','d'),('\u0d22','dh'),('\u0d23','n'),('\u0d24','t'),('\u0d25','th'),('\u0d26','d'),('\u0d27','dh'),('\u0d28','n'),
    ('\u0d2a','p'),('\u0d2b','ph'),('\u0d2c','b'),('\u0d2d','bh'),('\u0d2e','m'),('\u0d2f','y'),('\u0d30','r'),('\u0d32','l'),('\u0d35','v'),('\u0d36','sh'),
    ('\u0d37','sh'),('\u0d38','s'),('\u0d39','h'),('\u0d33','l'),('\u0d34','zh'),('\u0d31','r'),('\u0d3e','a'),('\u0d3f','i'),('\u0d40','ee'),('\u0d41','u'),
    ('\u0d42','oo'),('\u0d46','e'),('\u0d47','ee'),('\u0d48','ai'),('\u0d4a','o'),('\u0d4b','oo'),('\u0d4c','au'),('\u0d02','m'),('\u0d4d',''),
)
# Build regex sorted by key-length DESC for correct greedy matching
_INDIC_PAIRS = sorted(_INDIC_PAIRS_RAW, key=lambda x: len(x[0]), reverse=True)
_INDIC_REGEX = re.compile('|'.join(re.escape(k) for k, _ in _INDIC_PAIRS))
_INDIC_REPL  = {k: v for k, v in _INDIC_PAIRS}

def transliterate(s: str) -> str:
    return _INDIC_REGEX.sub(lambda m: _INDIC_REPL[m.group(0)], s)


LEGAL_REGEX = (
    r"(?i)\b(?:"
    r"private limited|pvt ltd|pvt limited|private ltd|limited|ltd|pvt|"
    r"llc|l\.l\.c\.|inc|incorporated|corporation|corp|llp|l\.l\.p\.|"
    r"co|company|gmbh|sarl|sa|sas|plc|center|services|service|partners|group|holdings"
    r")\b"
)

ACCENT_MAP = [('éèêëẽě','e'),('àâäãå','a'),('îïíì','i'),('ôöóòõ','o'),('ùûüúũ','u'),('ç','c'),('ñ','n')]

def strip_accents(s):
    for chars, rep in ACCENT_MAP:
        s = re.sub(f'[{chars}]', rep, s)
    return s

def normalize_text(df: pl.DataFrame) -> pl.DataFrame:
    """Fast vectorized normalization with Indic transliteration."""
    t0 = time.time()
    flush(f"  Normalizing {len(df):,} records...", end=" ")
    
    # Check for Indic
    has_indic = df["business_name"].fill_null("").str.contains(r"[\u0900-\u0D7F]").any() or \
                df["business_address"].fill_null("").str.contains(r"[\u0900-\u0D7F]").any()
    
    names = df["business_name"].fill_null("").to_list()
    addrs = df["business_address"].fill_null("").to_list()
    
    if has_indic:
        indic_mask = df["business_name"].fill_null("").str.contains(r"[\u0900-\u0D7F]") | \
                     df["business_address"].fill_null("").str.contains(r"[\u0900-\u0D7F]")
        indic_indices = [i for i, v in enumerate(indic_mask.to_list()) if v]
        for idx in indic_indices:
            names[idx] = transliterate(names[idx])
            addrs[idx] = transliterate(addrs[idx])
    
    df = df.with_columns([
        pl.Series("_nm", names),
        pl.Series("_ad", addrs),
        pl.col("country").fill_null("").str.strip_chars().alias("country"),
    ])
    
    df = df.with_columns([
        pl.col("_nm")
        .str.to_lowercase()
        .str.replace_all(r"[éèêë]", "e").str.replace_all(r"[àâäãå]", "a")
        .str.replace_all(r"[îïíì]", "i").str.replace_all(r"[ôöóòõ]", "o")
        .str.replace_all(r"[ùûüúũ]", "u").str.replace_all(r"ç", "c").str.replace_all(r"ñ", "n")
        .str.replace_all(r"\.(com|org|net|in|co|us|gov|io)\b", " ")
        .str.replace_all(r"\b(www|http|https)\b", " ")
        .str.replace_all("&", " and ")
        .str.replace_all(r"[^\w\s]", " ")
        .str.replace_all(r"([a-z])0([a-z])", "$1o$2")
        .str.replace_all(r"([a-z])1([a-z])", "$1l$2")
        .str.replace_all(r"([a-z])3([a-z])", "$1e$2")
        .str.replace_all(r"([a-z])4([a-z])", "$1a$2")
        .str.replace_all(r"([a-z])5([a-z])", "$1s$2")
        .str.replace_all(r"\s+", " ").str.strip_chars()
        .alias("clean_name"),

        pl.col("_ad")
        .str.to_lowercase()
        .str.replace_all(r"[éèêë]", "e").str.replace_all(r"[àâäãå]", "a")
        .str.replace_all(r"[îïíì]", "i").str.replace_all(r"[ôöóòõ]", "o")
        .str.replace_all(r"[ùûüúũ]", "u").str.replace_all(r"ç", "c")
        .str.replace_all(r"[^\w\s]", " ")
        .str.replace_all(r"\b0+(\d+)\b", r"$1")
        .str.replace_all(r"\s+", " ").str.strip_chars()
        .alias("std_address"),
    ])
    
    df = df.with_columns([
        pl.col("clean_name")
        .str.replace_all(LEGAL_REGEX, " ")
        .str.replace_all(r"\s+", " ").str.strip_chars()
        .alias("stripped_name"),
        
        pl.col("clean_name").str.replace_all(r"\s+", "").alias("spaceless_name"),
        
        pl.col("std_address").str.extract(r"\b(\d{1,6}[a-z]?)\b", 1).fill_null("").alias("house_no"),
        
        pl.when(pl.col("country").str.to_uppercase() == "INDIA")
          .then(pl.col("std_address").str.extract(r"\b([1-9]\d{5})\b", 1))
          .when(pl.col("country").str.to_uppercase() == "US")
          .then(pl.col("std_address").str.extract(r"\b(\d{5})(?:-\d{4})?\b", 1))
          .when(pl.col("country").str.to_uppercase() == "FRANCE")
          .then(pl.col("std_address").str.extract(r"\b((?:0[1-9]|[1-8]\d|9[0-8])\d{3})\b", 1))
          .otherwise(pl.col("std_address").str.extract(r"\b(\d{5,6})\b", 1))
          .fill_null("").alias("postal_code"),
    ])
    
    df = df.with_columns([
        pl.when(pl.col("stripped_name") != "")
        .then(pl.col("stripped_name")).otherwise(pl.col("clean_name"))
        .str.split(" ").list.sort().list.join(" ")
        .alias("sorted_name"),
        
        pl.col("stripped_name").str.split(" ").list.get(0).fill_null("").alias("first_word"),
        
        pl.col("std_address")
        .str.split(" ").list.filter(pl.element().str.len_chars() >= 3)
        .list.sort().list.join(" ")
        .alias("sorted_addr"),
        
        pl.col("std_address").str.split(" ").list.slice(1, 2).list.join(" ").alias("street_prefix"),
    ]).drop(["_nm", "_ad"])
    
    flush(f"done in {time.time()-t0:.1f}s")
    return df

def run_hash_channels(q, t):
    """9 fast hash-join candidate generation channels."""
    channels = []
    
    def ch(name, prio, q_df, t_df, on_cols):
        pairs = (
            q_df.select(["entity_id"] + on_cols)
            .join(t_df.select(["entity_id"] + on_cols).rename({"entity_id": "target_id"}), on=on_cols, how="inner")
            .select(["entity_id", "target_id"]).unique()
            .with_columns(pl.lit(prio).alias("prio"), pl.lit(name).alias("ch"))
        )
        flush(f"    {name:<35}: {len(pairs):>10,} pairs")
        return pairs
    
    channels.append(ch("Ch1 Exact Name", 1.00, q.filter(pl.col("clean_name") != ""), t.filter(pl.col("clean_name") != ""), ["country", "clean_name"]))
    channels.append(ch("Ch2 Sorted Name", 0.98, q.filter(pl.col("sorted_name") != ""), t.filter(pl.col("sorted_name") != ""), ["country", "sorted_name"]))
    channels.append(ch("Ch3 Spaceless Name", 0.95, q.filter(pl.col("spaceless_name").str.len_chars() >= 6), t.filter(pl.col("spaceless_name").str.len_chars() >= 6), ["country", "spaceless_name"]))
    channels.append(ch("Ch4 Exact Address", 0.90, q.filter(pl.col("std_address").str.len_chars() >= 10), t.filter(pl.col("std_address").str.len_chars() >= 10), ["country", "std_address"]))
    channels.append(ch("Ch5 Sorted Address", 0.88, q.filter(pl.col("sorted_addr").str.len_chars() >= 15), t.filter(pl.col("sorted_addr").str.len_chars() >= 15), ["country", "sorted_addr"]))
    channels.append(ch("Ch6 HouseNo + First Word", 0.82, q.filter((pl.col("house_no").str.len_chars() >= 2) & (pl.col("first_word").str.len_chars() >= 4)), t.filter((pl.col("house_no").str.len_chars() >= 2) & (pl.col("first_word").str.len_chars() >= 4)), ["country", "house_no", "first_word"]))
    channels.append(ch("Ch7 Postal + First Word", 0.80, q.filter((pl.col("postal_code").str.len_chars() >= 5) & (pl.col("first_word").str.len_chars() >= 4)), t.filter((pl.col("postal_code").str.len_chars() >= 5) & (pl.col("first_word").str.len_chars() >= 4)), ["country", "postal_code", "first_word"]))
    channels.append(ch("Ch8 HouseNo + Street Prefix", 0.78, q.filter((pl.col("house_no").str.len_chars() >= 2) & (pl.col("street_prefix").str.len_chars() >= 6)), t.filter((pl.col("house_no").str.len_chars() >= 2) & (pl.col("street_prefix").str.len_chars() >= 6)), ["country", "house_no", "street_prefix"]))
    
    pooled = (
        pl.concat(channels)
        .group_by(["entity_id", "target_id"])
        .agg([pl.col("prio").max().alias("max_prio"), pl.len().alias("n_channels")])
    )
    flush(f"    Total unique candidate pairs: {len(pooled):,}")
    return pooled

def run_tfidf_channel_sharded(q_df, t_df, top_k=60, batch_size=500, min_score=0.12):
    """Country-sharded char TF-IDF — one small matrix per country to avoid OOM."""
    flush(f"    Sharded TF-IDF: {len(q_df):,} queries over {len(t_df):,} targets (by country)")
    t0 = time.time()
    all_cands = []

    def txt(nm, addr):
        return f"{str(nm or '').lower()} {str(addr or '').lower()}".strip()

    countries = q_df["country"].unique().to_list()
    flush(f"    Processing {len(countries)} country shards...")

    for ctry in sorted(countries):
        q_c = q_df.filter(pl.col("country") == ctry)
        t_c = t_df.filter(pl.col("country") == ctry)
        if len(q_c) == 0 or len(t_c) == 0:
            continue

        flush(f"      [{ctry}] Q={len(q_c):,}  T={len(t_c):,}", end="  ")
        tc = time.time()

        tgt_texts = [txt(r['clean_name'], r['std_address'])
                     for r in t_c.select(['entity_id','clean_name','std_address']).iter_rows(named=True)]
        qry_texts = [txt(r['clean_name'], r['std_address'])
                     for r in q_c.select(['entity_id','clean_name','std_address']).iter_rows(named=True)]
        q_ids  = q_c['entity_id'].to_list()
        t_ids  = np.array(t_c['entity_id'].to_list())

        # Adaptive features: smaller for small shards
        mf = min(100_000, max(5_000, len(t_c) // 5))
        min_df = 2 if len(t_c) < 50_000 else 3

        vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 4), max_features=mf,
                              sublinear_tf=True, min_df=min_df, max_df=0.05,
                              strip_accents='unicode')
        try:
            tgt_mat = vec.fit_transform(tgt_texts)
            qry_mat = vec.transform(qry_texts)
        except Exception as e:
            flush(f"SKIP ({e})")
            continue

        tgt_mat = normalize(tgt_mat, norm='l2', copy=False).T.tocsc()
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

        flush(f"-> {len(ctry_cands):,} pairs  ({time.time()-tc:.1f}s)")
        all_cands.extend(ctry_cands)
        del tgt_mat, qry_mat, vec  # free memory immediately

    flush(f"    Sharded TF-IDF total: {len(all_cands):,} pairs in {time.time()-t0:.1f}s")
    return all_cands

def oracle_f05(cand_map, gt, query_list):
    tot_true = ret_true = blackouts = 0
    f05_list = []
    for qid in query_list:
        true_s = gt.get(qid, set())
        pred = true_s & cand_map.get(qid, set())
        tot_true += len(true_s); ret_true += len(pred)
        if not true_s:
            f05_list.append(1.0)
        elif not pred:
            f05_list.append(0.0); blackouts += 1
        else:
            r = len(pred)/len(true_s); f05_list.append(1.25*r/(0.25+r))
    return float(np.nanmean(f05_list)), ret_true/max(tot_true,1), blackouts

# ============================================================
# [1] LOAD DATA
# ============================================================
flush("\n[1] Loading validation sample and full target corpus...")
t_load = time.time()

val_df = pl.read_parquet(ROOT / "experiments" / "val_sample_50k.parquet")
val_gt = {}
for row in val_df.iter_rows(named=True):
    qid = str(row["entity_id"])
    raw = str(row.get("matched_entity_ids", "") or "").strip()
    val_gt[qid] = {t.strip() for t in raw.split(",") if t.strip() and t.strip() not in ("nan", "None")}
val_query_list = sorted(val_gt.keys())
flush(f"  Val queries: {len(val_query_list):,} | True links: {sum(len(v) for v in val_gt.values()):,}")

s2 = pl.read_csv(ROOT / "dataset/raw/train/train_source2.tsv", separator="\t", quote_char=None)
s3 = pl.read_csv(ROOT / "dataset/raw/train/train_source3.tsv", separator="\t", quote_char=None)
targets_all = pl.concat([s2, s3])
flush(f"  Targets: {len(targets_all):,} | Load time: {time.time()-t_load:.1f}s")

# ============================================================
# [2] NORMALIZE
# ============================================================
flush("\n[2] Normalizing all records...")
q_norm = normalize_text(val_df)
t_norm = normalize_text(targets_all)

# ============================================================
# [3] HASH-JOIN CHANNELS
# ============================================================
flush("\n[3] Running 9-channel hash-join retrieval...")
hash_cands = run_hash_channels(q_norm, t_norm)

# ============================================================
# [4] TF-IDF CHANNEL (on query subset that have blackouts + expand top negatives)
# ============================================================
flush("\n[4] Running char TF-IDF expansion (Ch10)...")
t_tfidf = time.time()

# We'll run TF-IDF by country to manage memory (US, India, France)
hash_cand_map = defaultdict(set)
for row in hash_cands.select(["entity_id", "target_id"]).iter_rows(named=True):
    hash_cand_map[str(row["entity_id"])].add(str(row["target_id"]))

# Identify blackout queries (no candidates at all in hash join) + queries with <10 candidates
blackout_qids = set(q for q in val_query_list if len(hash_cand_map.get(q, set())) < 20)
flush(f"  Queries needing TF-IDF boost (< 20 candidates): {len(blackout_qids):,}")

all_tfidf_cands = []
q_tfidf_df = q_norm.filter(pl.col("entity_id").is_in(list(blackout_qids)))
flush(f"  Running TF-IDF on {len(q_tfidf_df):,} queries vs {len(t_norm):,} targets...")

tfidf_cands = run_tfidf_channel_sharded(q_tfidf_df, t_norm, top_k=60, batch_size=500)
all_tfidf_cands.extend(tfidf_cands)
flush(f"  TF-IDF done in {time.time()-t_tfidf:.1f}s. Got {len(all_tfidf_cands):,} TF-IDF candidate pairs")

# Merge TF-IDF into candidate pool
if all_tfidf_cands:
    tfidf_pool = pl.DataFrame({
        "entity_id": [str(c[0]) for c in all_tfidf_cands],
        "target_id": [str(c[1]) for c in all_tfidf_cands],
        "max_prio": [float(c[2]) for c in all_tfidf_cands],
        "n_channels": [1] * len(all_tfidf_cands)
    })
else:
    tfidf_pool = pl.DataFrame(
        schema={"entity_id": pl.Utf8, "target_id": pl.Utf8, "max_prio": pl.Float64, "n_channels": pl.UInt32}
    )

# Union with hash candidates
all_cands = pl.concat([hash_cands, tfidf_pool]).group_by(["entity_id", "target_id"]).agg([
    pl.col("max_prio").max().alias("max_prio"),
    pl.col("n_channels").sum().alias("n_channels")
])
flush(f"  Combined unique pairs: {len(all_cands):,}")

# ============================================================
# [5] ORACLE EVALUATION
# ============================================================
flush("\n[5] Measuring oracle F0.5 at various K budgets...")
cand_sorted = all_cands.sort(["entity_id", "max_prio", "n_channels"], descending=[False, True, True])

for k in [40, 50, 75, 100, None]:
    if k:
        budgeted = cand_sorted.group_by("entity_id", maintain_order=True).head(k)
    else:
        budgeted = all_cands
    
    cm = defaultdict(set)
    for row in budgeted.select(["entity_id", "target_id"]).iter_rows(named=True):
        cm[str(row["entity_id"])].add(str(row["target_id"]))
    
    f05, rec, blk = oracle_f05(dict(cm), val_gt, val_query_list)
    k_str = f"K={k}" if k else "Unbudgeted"
    flush(f"  {k_str:<12}: Oracle F0.5 = {f05:.5f} | Link Recall = {rec*100:.2f}% | Blackouts = {blk:,}")

# ============================================================
# [6] SAVE CANDIDATE POOL FOR FEATURE EXTRACTION
# ============================================================
OUT_CANDS = ROOT / "experiments" / "phase2" / "phase2c_val_candidates.parquet"
all_cands.write_parquet(OUT_CANDS)
flush(f"\n[6] Saved candidate pool to {OUT_CANDS}")

# ============================================================
# [7] RICH FEATURE ENGINEERING
# ============================================================
flush("\n[7] Building rich feature set for candidate pairs...")

# Merge with normalized query/target data
from rapidfuzz import fuzz as rfuzz

q_dict = {str(r["entity_id"]): r for r in q_norm.select([
    "entity_id", "clean_name", "stripped_name", "sorted_name", "spaceless_name",
    "std_address", "sorted_addr", "house_no", "postal_code", "country", "first_word"
]).iter_rows(named=True)}

t_dict = {str(r["entity_id"]): r for r in t_norm.select([
    "entity_id", "clean_name", "stripped_name", "sorted_name", "spaceless_name",
    "std_address", "sorted_addr", "house_no", "postal_code", "country", "first_word"
]).iter_rows(named=True)}

def char_ngram_jac(a, b, n):
    if not a or not b or len(a) < n or len(b) < n: return 0.0
    sa = {a[i:i+n] for i in range(len(a)-n+1)}
    sb = {b[i:i+n] for i in range(len(b)-n+1)}
    u = len(sa | sb)
    return len(sa & sb) / u if u > 0 else 0.0

def tok_jac(a, b):
    sa = set((a or '').split())
    sb = set((b or '').split())
    if not sa and not sb: return 1.0
    u = len(sa | sb)
    return len(sa & sb) / u if u > 0 else 0.0

def tok_overlap(a, b):
    sa = set((a or '').split())
    sb = set((b or '').split())
    if not sa or not sb: return 0.0
    return len(sa & sb) / min(len(sa), len(sb))

# Build features in batches
cand_rows = all_cands.select(["entity_id", "target_id", "max_prio", "n_channels"]).iter_rows(named=True)
rows_list = list(cand_rows)
flush(f"  Building features for {len(rows_list):,} candidate pairs...")

feat_records = []
t_feat = time.time()
for i, row in enumerate(rows_list):
    if i % 100000 == 0 and i > 0:
        flush(f"    [{i:,}/{len(rows_list):,}] elapsed: {time.time()-t_feat:.1f}s")
    
    qid = str(row["entity_id"])
    tid = str(row["target_id"])
    q = q_dict.get(qid, {})
    t = t_dict.get(tid, {})
    
    qn = q.get("clean_name", "") or ""
    tn = t.get("clean_name", "") or ""
    qns = q.get("sorted_name", "") or ""
    tns = t.get("sorted_name", "") or ""
    qns2 = q.get("stripped_name", "") or ""
    tns2 = t.get("stripped_name", "") or ""
    qsp = q.get("spaceless_name", "") or ""
    tsp = t.get("spaceless_name", "") or ""
    qa = q.get("std_address", "") or ""
    ta = t.get("std_address", "") or ""
    qas = q.get("sorted_addr", "") or ""
    tas = t.get("sorted_addr", "") or ""
    qhno = q.get("house_no", "") or ""
    thno = t.get("house_no", "") or ""
    qpin = q.get("postal_code", "") or ""
    tpin = t.get("postal_code", "") or ""
    qfw = q.get("first_word", "") or ""
    tfw = t.get("first_word", "") or ""
    cty = (q.get("country", "") or "").lower()
    
    is_s3 = 1 if tid.startswith("S3-") else 0
    
    rec = {
        "entity_id": qid,
        "target_id": tid,
        "max_prio": float(row["max_prio"]),
        "n_channels": int(row["n_channels"]),
        
        # Exact match features
        "exact_clean_name": int(qn == tn and qn != ""),
        "exact_sorted_name": int(qns == tns and qns != ""),
        "exact_stripped_name": int(qns2 == tns2 and qns2 != ""),
        "exact_spaceless": int(len(qsp) >= 5 and qsp == tsp),
        "spaceless_subset": int(len(qsp) >= 5 and len(tsp) >= 5 and (qsp in tsp or tsp in qsp) and qsp != tsp),
        "exact_clean_addr": int(qa == ta and len(qa) >= 10),
        "exact_sorted_addr": int(qas == tas and len(qas) >= 12),
        "hno_match": int(qhno == thno and len(qhno) >= 2),
        "pin_match": int(qpin == tpin and len(qpin) >= 5),
        "fw_match": int(qfw == tfw and len(qfw) >= 4),
        
        # Fuzzy name similarities
        "name_jw": rfuzz.jaro_winkler(qn, tn) / 100.0,
        "name_tok_sort": rfuzz.token_sort_ratio(qn, tn) / 100.0,
        "name_tok_set": rfuzz.token_set_ratio(qn, tn) / 100.0,
        "name_partial": rfuzz.partial_ratio(qn, tn) / 100.0,
        "name_char2_jac": char_ngram_jac(qn, tn, 2),
        "name_char3_jac": char_ngram_jac(qn, tn, 3),
        "name_char4_jac": char_ngram_jac(qn, tn, 4),
        "name_tok_jac": tok_jac(qn, tn),
        "name_tok_ovlp": tok_overlap(qn, tn),
        
        # Address similarities
        "addr_jw": rfuzz.jaro_winkler(qa, ta) / 100.0 if qa and ta else 0.0,
        "addr_tok_sort": rfuzz.token_sort_ratio(qa, ta) / 100.0 if qa and ta else 0.0,
        "addr_tok_set": rfuzz.token_set_ratio(qa, ta) / 100.0 if qa and ta else 0.0,
        "addr_char3_jac": char_ngram_jac(qa, ta, 3),
        "addr_tok_jac": tok_jac(qas, tas),
        "addr_null_tgt": int(not ta),
        
        # Structural features
        "name_len_diff": abs(len(qn) - len(tn)),
        "name_len_ratio": min(len(qn), len(tn)) / max(len(qn), len(tn), 1),
        "addr_len_diff": abs(len(qa) - len(ta)),
        "name_wc_diff": abs(len(qn.split()) - len(tn.split())),
        "name_wc_ratio": min(len(qn.split()), len(tn.split())) / max(len(qn.split()), len(tn.split()), 1),
        "name_prefix3": int(qn[:3] == tn[:3] if len(qn) >= 3 and len(tn) >= 3 else 0),
        "name_prefix5": int(qn[:5] == tn[:5] if len(qn) >= 5 and len(tn) >= 5 else 0),
        
        # Numeric tokens
        "hno_contradiction": int(qhno != thno and qhno and thno),
        "pin_contradiction": int(qpin != tpin and qpin and tpin),
        "pin_prefix3": int(qpin[:3] == tpin[:3] if len(qpin) >= 3 and len(tpin) >= 3 else 0),
        "pin_both_present": int(bool(qpin) and bool(tpin)),
        
        # Interactions
        "name_x_addr": rfuzz.jaro_winkler(qn, tn) / 100.0 * rfuzz.token_sort_ratio(qa, ta) / 100.0 if qa and ta else 0.0,
        "name_x_pin": rfuzz.jaro_winkler(qn, tn) / 100.0 * int(qpin == tpin and len(qpin) >= 5),
        "sorted_x_addr": rfuzz.token_sort_ratio(qns, tns) / 100.0 * rfuzz.token_set_ratio(qa, ta) / 100.0 if qa and ta else 0.0,
        
        # Metadata
        "is_source3": is_s3,
        "india_flag": int("india" in cty),
        "france_flag": int("france" in cty),
    }
    feat_records.append(rec)

flush(f"  Feature building done in {time.time()-t_feat:.1f}s")

feat_df = pd.DataFrame(feat_records)
flush(f"  Feature matrix shape: {feat_df.shape}")

FEAT_COLS = [c for c in feat_df.columns if c not in ("entity_id", "target_id")]

# Label
label_list = []
for row in feat_df[["entity_id", "target_id"]].itertuples(index=False):
    qid, tid = row.entity_id, row.target_id
    label_list.append(1 if tid in val_gt.get(qid, set()) else 0)
feat_df["label"] = label_list
flush(f"  Pos rate: {np.mean(label_list)*100:.3f}%  Positive pairs: {sum(label_list):,}")

# Save features
FEAT_OUT = ROOT / "experiments" / "phase2" / "phase2c_val_features.parquet"
pl.from_pandas(feat_df).write_parquet(FEAT_OUT)
flush(f"  Saved features to {FEAT_OUT}")

# ============================================================
# [8] 5-FOLD OOF TRAINING — LightGBM + XGBoost GPU + CatBoost GPU
# ============================================================
flush("\n[8] Training 5-Fold OOF Meta-Ensemble...")

X = feat_df[FEAT_COLS].values.astype(np.float32)
y = np.array(label_list, dtype=np.int32)
queries_arr = feat_df["entity_id"].values
_, groups = np.unique(queries_arr, return_inverse=True)

oof_lgb = np.zeros(len(X), dtype=np.float32)
oof_xgb = np.zeros(len(X), dtype=np.float32)
oof_cb  = np.zeros(len(X), dtype=np.float32)

lgb_params = dict(objective="binary", metric="auc", learning_rate=0.03, num_leaves=127,
                  min_child_samples=20, n_estimators=1000, n_jobs=-1, random_state=42, verbose=-1,
                  colsample_bytree=0.8, subsample=0.8, subsample_freq=1)
xgb_params = dict(objective="binary:logistic", eval_metric="auc", learning_rate=0.03,
                  max_depth=7, n_estimators=1000, device="cuda", random_state=42,
                  colsample_bytree=0.8, subsample=0.8)
cb_params  = dict(iterations=1000, learning_rate=0.03, depth=7, task_type="GPU",
                  eval_metric="AUC", random_seed=42, verbose=0,
                  colsample_bylevel=0.8, subsample=0.8)

gkf = GroupKFold(n_splits=5)
fold_aucs = {"lgb": [], "xgb": [], "cb": []}

for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
    flush(f"\n  Fold {fold+1}/5 — Train: {len(tr_idx):,} | Val: {len(va_idx):,}")
    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]
    
    # LightGBM
    t0 = time.time()
    model_lgb = lgb.LGBMClassifier(**lgb_params)
    model_lgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
    oof_lgb[va_idx] = model_lgb.predict_proba(X_va)[:, 1]
    auc = roc_auc_score(y_va, oof_lgb[va_idx])
    fold_aucs["lgb"].append(auc)
    flush(f"    LGB: AUC={auc:.6f} ({time.time()-t0:.1f}s)")
    
    # XGBoost GPU
    t0 = time.time()
    model_xgb = xgb.XGBClassifier(**xgb_params)
    model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], early_stopping_rounds=50, verbose=False)
    oof_xgb[va_idx] = model_xgb.predict_proba(X_va)[:, 1]
    auc = roc_auc_score(y_va, oof_xgb[va_idx])
    fold_aucs["xgb"].append(auc)
    flush(f"    XGB: AUC={auc:.6f} ({time.time()-t0:.1f}s)")
    
    # CatBoost GPU
    t0 = time.time()
    model_cb = CatBoostClassifier(**cb_params)
    model_cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50)
    oof_cb[va_idx] = model_cb.predict_proba(X_va)[:, 1]
    auc = roc_auc_score(y_va, oof_cb[va_idx])
    fold_aucs["cb"].append(auc)
    flush(f"    CB:  AUC={auc:.6f} ({time.time()-t0:.1f}s)")

flush(f"\n  Mean AUC LGB:  {np.mean(fold_aucs['lgb']):.6f}")
flush(f"  Mean AUC XGB:  {np.mean(fold_aucs['xgb']):.6f}")
flush(f"  Mean AUC CB:   {np.mean(fold_aucs['cb']):.6f}")

# Meta-stacker
meta_X = np.column_stack([oof_lgb, oof_xgb, oof_cb])
meta_model = LogisticRegression(C=10.0, max_iter=500)
meta_model.fit(meta_X, y)
oof_meta = meta_model.predict_proba(meta_X)[:, 1]
meta_auc = roc_auc_score(y, oof_meta)
flush(f"  Meta-stacker AUC: {meta_auc:.6f}")

# ============================================================
# [9] THRESHOLD SEARCH + FINAL F0.5
# ============================================================
flush("\n[9] Threshold sweep to maximize macro F0.5...")

def compute_f05(preds, queries, targets, gt_map_local, query_list, theta):
    pred_map = defaultdict(list)
    for p, q, t in zip(preds, queries, targets):
        pred_map[q].append((p, t))
    
    f05_list = []
    for qid in query_list:
        true_s = gt_map_local.get(qid, set())
        preds_q = pred_map.get(qid, [])
        matched = {t for (p, t) in preds_q if p >= theta}
        
        if not true_s:
            f05_list.append(1.0 if not matched else 0.0)
        elif not matched:
            f05_list.append(0.0)
        else:
            prec = len(matched & true_s) / len(matched)
            rec = len(matched & true_s) / len(true_s)
            if prec + rec == 0:
                f05_list.append(0.0)
            else:
                f05_list.append((1.25 * prec * rec) / (0.25 * prec + rec))
    return float(np.mean(f05_list))

preds_arr = oof_meta
queries_arr2 = feat_df["entity_id"].values
targets_arr = feat_df["target_id"].values

best_f05 = 0; best_theta = 0.5
for theta in np.arange(0.30, 0.85, 0.01):
    f05 = compute_f05(preds_arr, queries_arr2, targets_arr, val_gt, val_query_list, theta)
    if f05 > best_f05:
        best_f05 = f05; best_theta = round(float(theta), 3)

flush(f"  Best macro F0.5: {best_f05:.5f} at theta={best_theta}")

# Per-country analysis
for country in ["US", "India"]:
    ctry_queries = [q for q in val_query_list if any(
        (feat_df[feat_df.entity_id == q]["india_flag"] == (1 if country == "India" else 0)).values[:1]
    )]
    if ctry_queries:
        f05_c = compute_f05(preds_arr, queries_arr2, targets_arr, val_gt, ctry_queries, best_theta)
        flush(f"  {country} F0.5: {f05_c:.5f}")

# ============================================================
# [10] SAVE FINAL RESULTS
# ============================================================
results = {
    "experiment": "Phase2C",
    "n_folds": 5,
    "n_features": len(FEAT_COLS),
    "candidate_pool_size": len(all_cands),
    "mean_auc_lgb": float(np.mean(fold_aucs["lgb"])),
    "mean_auc_xgb": float(np.mean(fold_aucs["xgb"])),
    "mean_auc_cb": float(np.mean(fold_aucs["cb"])),
    "meta_stacker_auc": float(meta_auc),
    "best_macro_f05": float(best_f05),
    "best_theta": float(best_theta),
}
OUT_JSON = ROOT / "experiments" / "phase2" / "phase2c_results.json"
with open(OUT_JSON, "w") as f:
    json.dump(results, f, indent=2)
flush(f"\nSaved results to {OUT_JSON}")

# Save models
OUT_MODEL_DIR = ROOT / "experiments" / "phase2" / "models"
pickle.dump(model_lgb, open(OUT_MODEL_DIR / "phase2c_lgb.pkl", "wb"))
pickle.dump(model_xgb, open(OUT_MODEL_DIR / "phase2c_xgb.pkl", "wb"))
pickle.dump(model_cb, open(OUT_MODEL_DIR / "phase2c_cb.pkl", "wb"))
pickle.dump(meta_model, open(OUT_MODEL_DIR / "phase2c_meta.pkl", "wb"))
flush("Models saved.")

flush("\n" + "=" * 70)
flush(f"PHASE 2C COMPLETE — Best Macro F0.5: {best_f05:.5f}")
flush("=" * 70)
