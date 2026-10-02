"""
experiments/phase2/phase2c_generate_submission.py
==================================================
PHASE 2C — TEST SUBMISSION GENERATOR

Uses the trained Phase 2C ensemble models to predict on the test set.
Applies the same 8-channel hash-join + TF-IDF + feature engineering pipeline,
then uses the saved LightGBM / XGBoost / CatBoost + meta-stacker models.

Output: submissions/phase2c_submission.csv
"""
import os, sys, re, time, json, pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
import polars as pl
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from rapidfuzz import fuzz as rfuzz

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Import the normalization + feature functions from the pipeline module
sys.path.insert(0, str(ROOT / "experiments" / "phase2"))

def flush(*args, **kwargs):
    print(*args, **kwargs, flush=True)

flush("=" * 70)
flush("PHASE 2C — TEST SUBMISSION GENERATOR")
flush("=" * 70)

MODEL_DIR = ROOT / "experiments" / "phase2" / "models"
OUT_DIR   = ROOT / "submissions"
OUT_DIR.mkdir(exist_ok=True)

# ============================================================
# LOAD MODELS
# ============================================================
flush("[0] Loading trained ensemble models...")
model_lgb  = pickle.load(open(MODEL_DIR / "phase2c_lgb.pkl", "rb"))
model_xgb  = pickle.load(open(MODEL_DIR / "phase2c_xgb.pkl", "rb"))
model_cb   = pickle.load(open(MODEL_DIR / "phase2c_cb.pkl", "rb"))
meta_model = pickle.load(open(MODEL_DIR / "phase2c_meta.pkl", "rb"))
results    = json.load(open(ROOT / "experiments" / "phase2" / "phase2c_results.json"))
THETA      = results["best_theta"]
FEAT_COLS  = model_lgb.feature_name_
flush(f"  Models loaded. Threshold: {THETA}  Features: {len(FEAT_COLS)}")

# ============================================================
# COPY HELPERS FROM PIPELINE (inline to avoid import issues)
# ============================================================
_INDIC_PAIRS_RAW = (
    ('\u09a1\u09bc','r'), ('\u09a2\u09bc','rh'), ('\u09af\u09bc','y'),
    ('\u0905','a'),('\u0906','aa'),('\u0907','i'),('\u0908','ee'),('\u0909','u'),('\u090a','oo'),('\u090f','e'),('\u0910','ai'),('\u0913','o'),('\u0914','au'),
    ('\u0915','k'),('\u0916','kh'),('\u0917','g'),('\u0918','gh'),('\u0919','ng'),('\u091a','ch'),('\u091b','chh'),('\u091c','j'),('\u091d','jh'),('\u091e','ny'),
    ('\u091f','t'),('\u0920','th'),('\u0921','d'),('\u0922','dh'),('\u0923','n'),('\u0924','t'),('\u0925','th'),('\u0926','d'),('\u0927','dh'),('\u0928','n'),
    ('\u092a','p'),('\u092b','ph'),('\u092c','b'),('\u092d','bh'),('\u092e','m'),('\u092f','y'),('\u0930','r'),('\u0932','l'),('\u0935','v'),('\u0936','sh'),
    ('\u0937','sh'),('\u0938','s'),('\u0939','h'),('\u0933','l'),('\u093e','a'),('\u093f','i'),('\u0940','ee'),('\u0941','u'),('\u0942','oo'),('\u0947','e'),
    ('\u0948','ai'),('\u094b','o'),('\u094c','au'),('\u0902','n'),('\u0903','h'),('\u094d',''),('\u0901','n'),('\u0943','ri'),
    ('\u0985','a'),('\u0986','aa'),('\u0987','i'),('\u0988','ee'),('\u0989','u'),('\u098a','oo'),('\u098f','e'),('\u0990','ai'),('\u0993','o'),('\u0994','au'),
    ('\u0995','k'),('\u0996','kh'),('\u0997','g'),('\u0998','gh'),('\u0999','ng'),('\u099a','ch'),('\u099b','chh'),('\u099c','j'),('\u099d','jh'),('\u099e','ny'),
    ('\u099f','t'),('\u09a0','th'),('\u09a1','d'),('\u09a2','dh'),('\u09a3','n'),('\u09a4','t'),('\u09a5','th'),('\u09a6','d'),('\u09a7','dh'),('\u09a8','n'),
    ('\u09aa','p'),('\u09ab','ph'),('\u09ac','b'),('\u09ad','bh'),('\u09ae','m'),('\u09af','y'),('\u09b0','r'),('\u09b2','l'),('\u09b6','sh'),('\u09b7','sh'),
    ('\u09b8','s'),('\u09b9','h'),('\u09be','a'),('\u09bf','i'),('\u09c0','ee'),('\u09c1','u'),('\u09c2','oo'),
    ('\u09c7','e'),('\u09c8','ai'),('\u09cb','o'),('\u09cc','au'),('\u09cd',''),
    ('\u0c05','a'),('\u0c06','aa'),('\u0c07','i'),('\u0c08','ee'),('\u0c09','u'),('\u0c0a','oo'),('\u0c0e','e'),('\u0c0f','ee'),('\u0c10','ai'),('\u0c12','o'),('\u0c13','oo'),('\u0c14','au'),
    ('\u0c15','k'),('\u0c16','kh'),('\u0c17','g'),('\u0c18','gh'),('\u0c19','ng'),('\u0c1a','ch'),('\u0c1b','chh'),('\u0c1c','j'),('\u0c1d','jh'),('\u0c1e','ny'),
    ('\u0c1f','t'),('\u0c20','th'),('\u0c21','d'),('\u0c22','dh'),('\u0c23','n'),('\u0c24','t'),('\u0c25','th'),('\u0c26','d'),('\u0c27','dh'),('\u0c28','n'),
    ('\u0c2a','p'),('\u0c2b','ph'),('\u0c2c','b'),('\u0c2d','bh'),('\u0c2e','m'),('\u0c2f','y'),('\u0c30','r'),('\u0c32','l'),('\u0c35','v'),('\u0c36','sh'),
    ('\u0c37','sh'),('\u0c38','s'),('\u0c39','h'),('\u0c33','l'),('\u0c3e','a'),('\u0c3f','i'),('\u0c40','ee'),('\u0c41','u'),('\u0c42','oo'),('\u0c46','e'),
    ('\u0c47','ee'),('\u0c48','ai'),('\u0c4a','o'),('\u0c4b','oo'),('\u0c4c','au'),('\u0c02','m'),('\u0c4d',''),
    ('\u0b85','a'),('\u0b86','aa'),('\u0b87','i'),('\u0b88','ee'),('\u0b89','u'),('\u0b8a','oo'),('\u0b8e','e'),('\u0b8f','ee'),('\u0b90','ai'),('\u0b92','o'),('\u0b93','oo'),('\u0b94','au'),
    ('\u0b95','k'),('\u0b99','ng'),('\u0b9a','ch'),('\u0b9e','ny'),('\u0b9f','t'),('\u0ba3','n'),('\u0ba4','t'),('\u0ba8','n'),('\u0baa','p'),('\u0bae','m'),
    ('\u0baf','y'),('\u0bb0','r'),('\u0bb2','l'),('\u0bb5','v'),('\u0bb4','zh'),('\u0bb3','l'),('\u0bb1','r'),('\u0ba9','n'),('\u0bbe','a'),('\u0bbf','i'),
    ('\u0bc0','ee'),('\u0bc1','u'),('\u0bc2','oo'),('\u0bc6','e'),('\u0bc7','ee'),('\u0bc8','ai'),('\u0bca','o'),('\u0bcb','oo'),('\u0bcc','au'),('\u0bcd',''),
    ('\u0c85','a'),('\u0c86','aa'),('\u0c87','i'),('\u0c88','ee'),('\u0c89','u'),('\u0c8a','oo'),('\u0c8e','e'),('\u0c8f','ee'),('\u0c90','ai'),('\u0c92','o'),('\u0c93','oo'),('\u0c94','au'),
    ('\u0c95','k'),('\u0c96','kh'),('\u0c97','g'),('\u0c98','gh'),('\u0c99','ng'),('\u0c9a','ch'),('\u0c9b','chh'),('\u0c9c','j'),('\u0c9d','jh'),('\u0c9e','ny'),
    ('\u0c9f','t'),('\u0ca0','th'),('\u0ca1','d'),('\u0ca2','dh'),('\u0ca3','n'),('\u0ca4','t'),('\u0ca5','th'),('\u0ca6','d'),('\u0ca7','dh'),('\u0ca8','n'),
    ('\u0caa','p'),('\u0cab','ph'),('\u0cac','b'),('\u0cad','bh'),('\u0cae','m'),('\u0caf','y'),('\u0cb0','r'),('\u0cb2','l'),('\u0cb5','v'),('\u0cb6','sh'),
    ('\u0cb7','sh'),('\u0cb8','s'),('\u0cb9','h'),('\u0cb3','l'),('\u0cbe','a'),('\u0cbf','i'),('\u0cc0','ee'),('\u0cc1','u'),('\u0cc2','oo'),('\u0cc6','e'),
    ('\u0cc7','ee'),('\u0cc8','ai'),('\u0cca','o'),('\u0ccb','oo'),('\u0ccc','au'),('\u0c82','m'),('\u0ccd',''),
    ('\u0d05','a'),('\u0d06','aa'),('\u0d07','i'),('\u0d08','ee'),('\u0d09','u'),('\u0d0a','oo'),('\u0d0e','e'),('\u0d0f','ee'),('\u0d10','ai'),('\u0d12','o'),('\u0d13','oo'),('\u0d14','au'),
    ('\u0d15','k'),('\u0d16','kh'),('\u0d17','g'),('\u0d18','gh'),('\u0d19','ng'),('\u0d1a','ch'),('\u0d1b','chh'),('\u0d1c','j'),('\u0d1d','jh'),('\u0d1e','ny'),
    ('\u0d1f','t'),('\u0d20','th'),('\u0d21','d'),('\u0d22','dh'),('\u0d23','n'),('\u0d24','t'),('\u0d25','th'),('\u0d26','d'),('\u0d27','dh'),('\u0d28','n'),
    ('\u0d2a','p'),('\u0d2b','ph'),('\u0d2c','b'),('\u0d2d','bh'),('\u0d2e','m'),('\u0d2f','y'),('\u0d30','r'),('\u0d32','l'),('\u0d35','v'),('\u0d36','sh'),
    ('\u0d37','sh'),('\u0d38','s'),('\u0d39','h'),('\u0d33','l'),('\u0d34','zh'),('\u0d31','r'),('\u0d3e','a'),('\u0d3f','i'),('\u0d40','ee'),('\u0d41','u'),
    ('\u0d42','oo'),('\u0d46','e'),('\u0d47','ee'),('\u0d48','ai'),('\u0d4a','o'),('\u0d4b','oo'),('\u0d4c','au'),('\u0d02','m'),('\u0d4d',''),
)
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

def normalize_text(df: pl.DataFrame) -> pl.DataFrame:
    t0 = time.time()
    flush(f"  Normalizing {len(df):,} records...", end=" ")
    has_indic = (
        df["business_name"].fill_null("").str.contains(r"[\u0900-\u0D7F]").any() or
        df["business_address"].fill_null("").str.contains(r"[\u0900-\u0D7F]").any()
    )
    names = df["business_name"].fill_null("").to_list()
    addrs = df["business_address"].fill_null("").to_list()
    if has_indic:
        indic_mask = (
            df["business_name"].fill_null("").str.contains(r"[\u0900-\u0D7F]") |
            df["business_address"].fill_null("").str.contains(r"[\u0900-\u0D7F]")
        )
        for idx in [i for i, v in enumerate(indic_mask.to_list()) if v]:
            names[idx] = transliterate(names[idx])
            addrs[idx] = transliterate(addrs[idx])
    df = df.with_columns([
        pl.Series("_nm", names),
        pl.Series("_ad", addrs),
        pl.col("country").fill_null("").str.strip_chars().alias("country"),
    ])
    df = df.with_columns([
        pl.col("_nm").str.to_lowercase()
        .str.replace_all(r"[éèêë]", "e").str.replace_all(r"[àâäãå]", "a")
        .str.replace_all(r"[îïíì]", "i").str.replace_all(r"[ôöóòõ]", "o")
        .str.replace_all(r"[ùûüúũ]", "u").str.replace_all(r"ç", "c").str.replace_all(r"ñ", "n")
        .str.replace_all(r"\.(com|org|net|in|co|us|gov|io)\b", " ")
        .str.replace_all(r"\b(www|http|https)\b", " ")
        .str.replace_all("&", " and ")
        .str.replace_all(r"[^\w\s]", " ")
        .str.replace_all(r"([a-z])0([a-z])", "$1o$2").str.replace_all(r"([a-z])1([a-z])", "$1l$2")
        .str.replace_all(r"([a-z])3([a-z])", "$1e$2").str.replace_all(r"([a-z])4([a-z])", "$1a$2")
        .str.replace_all(r"([a-z])5([a-z])", "$1s$2")
        .str.replace_all(r"\s+", " ").str.strip_chars().alias("clean_name"),
        pl.col("_ad").str.to_lowercase()
        .str.replace_all(r"[éèêë]", "e").str.replace_all(r"[àâäãå]", "a")
        .str.replace_all(r"[îïíì]", "i").str.replace_all(r"[ôöóòõ]", "o")
        .str.replace_all(r"[ùûüúũ]", "u").str.replace_all(r"ç", "c")
        .str.replace_all(r"[^\w\s]", " ").str.replace_all(r"\b0+(\d+)\b", r"$1")
        .str.replace_all(r"\s+", " ").str.strip_chars().alias("std_address"),
    ])
    df = df.with_columns([
        pl.col("clean_name").str.replace_all(LEGAL_REGEX, " ")
        .str.replace_all(r"\s+", " ").str.strip_chars().alias("stripped_name"),
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
        .str.split(" ").list.sort().list.join(" ").alias("sorted_name"),
        pl.col("stripped_name").str.split(" ").list.get(0).fill_null("").alias("first_word"),
        pl.col("std_address").str.split(" ").list.filter(pl.element().str.len_chars() >= 3)
        .list.sort().list.join(" ").alias("sorted_addr"),
        pl.col("std_address").str.split(" ").list.slice(1, 2).list.join(" ").alias("street_prefix"),
    ]).drop(["_nm", "_ad"])
    flush(f"done in {time.time()-t0:.1f}s")
    return df

def run_hash_channels(q, t):
    channels = []
    def ch(name, prio, q_df, t_df, on_cols):
        pairs = (
            q_df.select(["entity_id"] + on_cols)
            .join(t_df.select(["entity_id"] + on_cols).rename({"entity_id": "target_id"}), on=on_cols, how="inner")
            .select(["entity_id", "target_id"]).unique()
            .with_columns(pl.lit(prio).alias("prio"))
        )
        flush(f"    {name:<35}: {len(pairs):>10,} pairs")
        return pairs
    channels.append(ch("Ch1 Exact Name", 1.00, q.filter(pl.col("clean_name") != ""), t.filter(pl.col("clean_name") != ""), ["country", "clean_name"]))
    channels.append(ch("Ch2 Sorted Name", 0.98, q.filter(pl.col("sorted_name") != ""), t.filter(pl.col("sorted_name") != ""), ["country", "sorted_name"]))
    channels.append(ch("Ch3 Spaceless", 0.95, q.filter(pl.col("spaceless_name").str.len_chars() >= 6), t.filter(pl.col("spaceless_name").str.len_chars() >= 6), ["country", "spaceless_name"]))
    channels.append(ch("Ch4 Exact Addr", 0.90, q.filter(pl.col("std_address").str.len_chars() >= 10), t.filter(pl.col("std_address").str.len_chars() >= 10), ["country", "std_address"]))
    channels.append(ch("Ch5 Sorted Addr", 0.88, q.filter(pl.col("sorted_addr").str.len_chars() >= 15), t.filter(pl.col("sorted_addr").str.len_chars() >= 15), ["country", "sorted_addr"]))
    channels.append(ch("Ch6 HouseNo+FW", 0.82, q.filter((pl.col("house_no").str.len_chars() >= 2) & (pl.col("first_word").str.len_chars() >= 4)), t.filter((pl.col("house_no").str.len_chars() >= 2) & (pl.col("first_word").str.len_chars() >= 4)), ["country", "house_no", "first_word"]))
    channels.append(ch("Ch7 Postal+FW", 0.80, q.filter((pl.col("postal_code").str.len_chars() >= 5) & (pl.col("first_word").str.len_chars() >= 4)), t.filter((pl.col("postal_code").str.len_chars() >= 5) & (pl.col("first_word").str.len_chars() >= 4)), ["country", "postal_code", "first_word"]))
    channels.append(ch("Ch8 HouseNo+StreetPfx", 0.78, q.filter((pl.col("house_no").str.len_chars() >= 2) & (pl.col("street_prefix").str.len_chars() >= 6)), t.filter((pl.col("house_no").str.len_chars() >= 2) & (pl.col("street_prefix").str.len_chars() >= 6)), ["country", "house_no", "street_prefix"]))
    pooled = (
        pl.concat(channels)
        .group_by(["entity_id", "target_id"])
        .agg([pl.col("prio").max().alias("max_prio"), pl.len().alias("n_channels")])
    )
    flush(f"    Total unique pairs: {len(pooled):,}")
    return pooled

def run_tfidf_channel(q_df, t_df, top_k=60, batch_size=1000, min_score=0.10):
    flush(f"    TF-IDF on {len(q_df):,} queries vs {len(t_df):,} targets...")
    t0 = time.time()
    tgt_texts = [f"{r['clean_name']} {r['std_address']}".lower().strip()
                 for r in t_df.select(['entity_id','clean_name','std_address']).iter_rows(named=True)]
    qry_texts = [f"{r['clean_name']} {r['std_address']}".lower().strip()
                 for r in q_df.select(['entity_id','clean_name','std_address']).iter_rows(named=True)]
    q_ids = q_df['entity_id'].to_list()
    t_ids = np.array(t_df['entity_id'].to_list())
    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 4), max_features=150_000,
                          sublinear_tf=True, min_df=3, max_df=0.03, strip_accents='unicode')
    tgt_mat = vec.fit_transform(tgt_texts)
    qry_mat = vec.transform(qry_texts)
    tgt_mat = normalize(tgt_mat, norm='l2', copy=False).tocsc()
    qry_mat = normalize(qry_mat, norm='l2', copy=False)
    flush(f"    Vectorizer fit done {time.time()-t0:.1f}s. Shape: {tgt_mat.shape}")
    cands = []
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
                    cands.append((qid, t_ids[ridx], float(sc)))
        if b % 20000 == 0:
            flush(f"    [{end:,}/{len(q_ids):,}] cands: {len(cands):,}")
    flush(f"    TF-IDF done in {time.time()-t0:.1f}s. Total pairs: {len(cands):,}")
    return cands

def char_ngram_jac(a, b, n):
    if not a or not b or len(a) < n or len(b) < n: return 0.0
    sa = {a[i:i+n] for i in range(len(a)-n+1)}
    sb = {b[i:i+n] for i in range(len(b)-n+1)}
    u = len(sa | sb); return len(sa & sb) / u if u > 0 else 0.0

def tok_jac(a, b):
    sa = set((a or '').split()); sb = set((b or '').split())
    if not sa and not sb: return 1.0
    u = len(sa | sb); return len(sa & sb) / u if u > 0 else 0.0

def tok_overlap(a, b):
    sa = set((a or '').split()); sb = set((b or '').split())
    if not sa or not sb: return 0.0
    return len(sa & sb) / min(len(sa), len(sb))

def build_features(cands_df, q_dict, t_dict):
    records = []
    t0 = time.time()
    rows = cands_df.iter_rows(named=True)
    total = len(cands_df)
    for i, row in enumerate(rows):
        if i % 200000 == 0 and i > 0:
            flush(f"    [{i:,}/{total:,}] {time.time()-t0:.1f}s")
        qid = str(row["entity_id"]); tid = str(row["target_id"])
        q = q_dict.get(qid, {}); t = t_dict.get(tid, {})
        qn=q.get("clean_name","") or ""; tn=t.get("clean_name","") or ""
        qns=q.get("sorted_name","") or ""; tns=t.get("sorted_name","") or ""
        qns2=q.get("stripped_name","") or ""; tns2=t.get("stripped_name","") or ""
        qsp=q.get("spaceless_name","") or ""; tsp=t.get("spaceless_name","") or ""
        qa=q.get("std_address","") or ""; ta=t.get("std_address","") or ""
        qas=q.get("sorted_addr","") or ""; tas=t.get("sorted_addr","") or ""
        qhno=q.get("house_no","") or ""; thno=t.get("house_no","") or ""
        qpin=q.get("postal_code","") or ""; tpin=t.get("postal_code","") or ""
        qfw=q.get("first_word","") or ""; tfw=t.get("first_word","") or ""
        cty=(q.get("country","") or "").lower()
        rec = {
            "entity_id": qid, "target_id": tid,
            "max_prio": float(row["max_prio"]), "n_channels": int(row["n_channels"]),
            "exact_clean_name": int(qn==tn and qn!=""), "exact_sorted_name": int(qns==tns and qns!=""),
            "exact_stripped_name": int(qns2==tns2 and qns2!=""),
            "exact_spaceless": int(len(qsp)>=5 and qsp==tsp),
            "spaceless_subset": int(len(qsp)>=5 and len(tsp)>=5 and (qsp in tsp or tsp in qsp) and qsp!=tsp),
            "exact_clean_addr": int(qa==ta and len(qa)>=10), "exact_sorted_addr": int(qas==tas and len(qas)>=12),
            "hno_match": int(qhno==thno and len(qhno)>=2), "pin_match": int(qpin==tpin and len(qpin)>=5),
            "fw_match": int(qfw==tfw and len(qfw)>=4),
            "name_jw": rfuzz.jaro_winkler(qn, tn)/100.0, "name_tok_sort": rfuzz.token_sort_ratio(qn,tn)/100.0,
            "name_tok_set": rfuzz.token_set_ratio(qn,tn)/100.0, "name_partial": rfuzz.partial_ratio(qn,tn)/100.0,
            "name_char2_jac": char_ngram_jac(qn,tn,2), "name_char3_jac": char_ngram_jac(qn,tn,3),
            "name_char4_jac": char_ngram_jac(qn,tn,4),
            "name_tok_jac": tok_jac(qn,tn), "name_tok_ovlp": tok_overlap(qn,tn),
            "addr_jw": rfuzz.jaro_winkler(qa,ta)/100.0 if qa and ta else 0.0,
            "addr_tok_sort": rfuzz.token_sort_ratio(qa,ta)/100.0 if qa and ta else 0.0,
            "addr_tok_set": rfuzz.token_set_ratio(qa,ta)/100.0 if qa and ta else 0.0,
            "addr_char3_jac": char_ngram_jac(qa,ta,3), "addr_tok_jac": tok_jac(qas,tas),
            "addr_null_tgt": int(not ta),
            "name_len_diff": abs(len(qn)-len(tn)), "name_len_ratio": min(len(qn),len(tn))/max(len(qn),len(tn),1),
            "addr_len_diff": abs(len(qa)-len(ta)), "name_wc_diff": abs(len(qn.split())-len(tn.split())),
            "name_wc_ratio": min(len(qn.split()),len(tn.split()))/max(len(qn.split()),len(tn.split()),1),
            "name_prefix3": int(qn[:3]==tn[:3] if len(qn)>=3 and len(tn)>=3 else 0),
            "name_prefix5": int(qn[:5]==tn[:5] if len(qn)>=5 and len(tn)>=5 else 0),
            "hno_contradiction": int(qhno!=thno and qhno and thno),
            "pin_contradiction": int(qpin!=tpin and qpin and tpin),
            "pin_prefix3": int(qpin[:3]==tpin[:3] if len(qpin)>=3 and len(tpin)>=3 else 0),
            "pin_both_present": int(bool(qpin) and bool(tpin)),
            "name_x_addr": rfuzz.jaro_winkler(qn,tn)/100.0*rfuzz.token_sort_ratio(qa,ta)/100.0 if qa and ta else 0.0,
            "name_x_pin": rfuzz.jaro_winkler(qn,tn)/100.0*int(qpin==tpin and len(qpin)>=5),
            "sorted_x_addr": rfuzz.token_sort_ratio(qns,tns)/100.0*rfuzz.token_set_ratio(qa,ta)/100.0 if qa and ta else 0.0,
            "is_source3": int(tid.startswith("S3-")),
            "india_flag": int("india" in cty), "france_flag": int("france" in cty),
        }
        records.append(rec)
    flush(f"    Features built in {time.time()-t0:.1f}s for {total:,} pairs")
    return pd.DataFrame(records)

# ============================================================
# MAIN
# ============================================================
flush("\n[1] Loading test queries and full target corpus...")
t_load = time.time()
test_q  = pl.read_csv(ROOT / "dataset/raw/test/test_source1.tsv", separator="\t", quote_char=None)
tgt_s2  = pl.read_csv(ROOT / "dataset/raw/test/test_source2.tsv", separator="\t", quote_char=None)
tgt_s3  = pl.read_csv(ROOT / "dataset/raw/test/test_source3.tsv", separator="\t", quote_char=None)
targets = pl.concat([tgt_s2, tgt_s3])
flush(f"  Queries: {len(test_q):,} | Targets: {len(targets):,} | Load: {time.time()-t_load:.1f}s")

flush("\n[2] Normalizing...")
q_norm = normalize_text(test_q)
t_norm = normalize_text(targets)

flush("\n[3] Hash-join channels...")
hash_cands = run_hash_channels(q_norm, t_norm)

flush("\n[4] TF-IDF expansion for sparse-candidate queries...")
from collections import defaultdict
hash_map = defaultdict(set)
for r in hash_cands.select(["entity_id","target_id"]).iter_rows(named=True):
    hash_map[r["entity_id"]].add(r["target_id"])
sparse_qids = [q for q in q_norm["entity_id"].to_list() if len(hash_map.get(q, set())) < 20]
flush(f"  Sparse queries needing TF-IDF: {len(sparse_qids):,}")
tfidf_cands = []
if sparse_qids:
    q_tfidf = q_norm.filter(pl.col("entity_id").is_in(sparse_qids))
    raw_tfidf = run_tfidf_channel(q_tfidf, t_norm, top_k=60, batch_size=1000)
    tfidf_cands = raw_tfidf
tfidf_df = pl.DataFrame({"entity_id": [c[0] for c in tfidf_cands],
                          "target_id": [c[1] for c in tfidf_cands],
                          "max_prio":  [c[2] for c in tfidf_cands],
                          "n_channels": [1]*len(tfidf_cands)}) if tfidf_cands else pl.DataFrame(schema={"entity_id":pl.Utf8,"target_id":pl.Utf8,"max_prio":pl.Float64,"n_channels":pl.Int64})
all_cands = pl.concat([hash_cands, tfidf_df]).group_by(["entity_id","target_id"]).agg([
    pl.col("max_prio").max().alias("max_prio"), pl.col("n_channels").sum().alias("n_channels")
])
flush(f"  Combined candidate pairs: {len(all_cands):,}")

flush("\n[5] Building features...")
q_dict = {str(r["entity_id"]): r for r in q_norm.select([
    "entity_id","clean_name","stripped_name","sorted_name","spaceless_name",
    "std_address","sorted_addr","house_no","postal_code","country","first_word"
]).iter_rows(named=True)}
t_dict = {str(r["entity_id"]): r for r in t_norm.select([
    "entity_id","clean_name","stripped_name","sorted_name","spaceless_name",
    "std_address","sorted_addr","house_no","postal_code","country","first_word"
]).iter_rows(named=True)}
feat_df = build_features(all_cands, q_dict, t_dict)

flush("\n[6] Predicting with ensemble...")
X = feat_df[FEAT_COLS].values.astype(np.float32)
p_lgb = model_lgb.predict_proba(X)[:,1]
p_xgb = model_xgb.predict_proba(X)[:,1]
p_cb  = model_cb.predict_proba(X)[:,1]
meta_X = np.column_stack([p_lgb, p_xgb, p_cb])
probs = meta_model.predict_proba(meta_X)[:,1]
feat_df["prob"] = probs
flush(f"  Predictions done. Threshold: {THETA}")

flush("\n[7] Building submission...")
pred_map = defaultdict(list)
for _, row in feat_df[["entity_id","target_id","prob"]].iterrows():
    if row["prob"] >= THETA:
        pred_map[row["entity_id"]].append(row["target_id"])

# Build final submission
all_query_ids = test_q["entity_id"].to_list()
rows_out = []
for qid in all_query_ids:
    matched = pred_map.get(qid, [])
    rows_out.append({"entity_id": qid, "matched_entity_ids": ",".join(matched)})

sub_df = pd.DataFrame(rows_out)
sub_path = OUT_DIR / "phase2c_submission.csv"
sub_df.to_csv(sub_path, index=False)

n_matched = sum(1 for _, r in sub_df.iterrows() if r["matched_entity_ids"])
avg_matched = sub_df["matched_entity_ids"].apply(lambda x: len(x.split(",")) if x else 0).mean()
flush(f"\nSubmission stats:")
flush(f"  Total queries: {len(sub_df):,}")
flush(f"  Queries with matches: {n_matched:,} ({n_matched/len(sub_df)*100:.1f}%)")
flush(f"  Avg matches per query: {avg_matched:.2f}")
flush(f"  Saved to: {sub_path}")
flush("\nDONE!")
