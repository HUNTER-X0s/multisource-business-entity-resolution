import polars as pl
import re
import time
from collections import defaultdict

# 1. Load validation sample and ground truth
val_df = pl.read_parquet("experiments/val_sample_50k.parquet")
s2 = pl.read_csv("dataset/raw/train/train_source2.tsv", separator="\t", quote_char=None)
s3 = pl.read_csv("dataset/raw/train/train_source3.tsv", separator="\t", quote_char=None)
targets = pl.concat([s2, s3])

val_gt = {}
for row in val_df.iter_rows(named=True):
    qid = str(row["entity_id"])
    raw = str(row.get("matched_entity_ids", "") or "").strip()
    val_gt[qid] = {t.strip() for t in raw.split(",") if t.strip() and t.strip() not in ("nan", "None")}

queries_with_gt = {q: ts for q, ts in val_gt.items() if ts}
print(f"Total validation queries: {len(val_df):,}")
print(f"Queries with true matches: {len(queries_with_gt):,}")

# Transliteration mapping
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

LEGAL_SUFFIX_REGEX = (
    r"(?i)\b(?:"
    r"private limited|pvt ltd|pvt limited|private ltd|limited|ltd|pvt|"
    r"llc|l\.l\.c\.|inc|incorporated|corporation|corp|llp|l\.l\.p\.|"
    r"co|company|gmbh|sarl|sa|sas|plc|center|services|service|partners|group|holdings"
    r")\b"
)

def normalize_df(df: pl.DataFrame):
    t0 = time.time()
    names = df["business_name"].fill_null("").to_list()
    addrs = df["business_address"].fill_null("").to_list()
    for i in range(len(names)):
        if any('\u0900' <= c <= '\u0d7f' for c in names[i]): names[i] = transliterate(names[i])
        if any('\u0900' <= c <= '\u0d7f' for c in addrs[i]): addrs[i] = transliterate(addrs[i])
    
    res = df.with_columns([
        pl.Series("_nm", names),
        pl.Series("_ad", addrs),
        pl.col("country").fill_null("").str.to_uppercase().str.strip_chars().alias("country"),
    ])
    res = res.with_columns([
        pl.col("_nm").str.to_lowercase()
        .str.replace_all(r"\.(com|org|net|in|co|us|gov|io)\b", " ")
        .str.replace_all(r"\b(www|http|https)\b", " ")
        .str.replace_all("&", " and ")
        .str.replace_all(r"[^\w\s]", " ")
        .str.replace_all(r"\s+", " ").str.strip_chars()
        .alias("clean_name"),
        
        pl.col("_ad").str.to_lowercase()
        .str.replace_all(r"[^\w\s]", " ")
        .str.replace_all(r"\b0+(\d+)\b", r"$1")
        .str.replace_all(r"\s+", " ").str.strip_chars()
        .alias("std_address"),
    ])
    res = res.with_columns([
        pl.col("clean_name").str.replace_all(LEGAL_SUFFIX_REGEX, " ").str.replace_all(r"\s+", " ").str.strip_chars().alias("stripped_name"),
        pl.col("clean_name").str.replace_all(r"\s+", "").alias("spaceless_name"),
        pl.col("std_address").str.extract(r"\b(\d{1,6}[a-z]?)\b", 1).fill_null("").alias("house_no"),
        pl.when(pl.col("country") == "INDIA").then(pl.col("std_address").str.extract(r"\b([1-9]\d{5})\b", 1))
          .when(pl.col("country") == "US").then(pl.col("std_address").str.extract(r"\b(\d{5})(?:-\d{4})?\b", 1))
          .when(pl.col("country") == "FRANCE").then(pl.col("std_address").str.extract(r"\b((?:0[1-9]|[1-8]\d|9[0-8])\d{3})\b", 1))
          .otherwise(pl.col("std_address").str.extract(r"\b(\d{5,6})\b", 1))
          .fill_null("").alias("postal_code"),
    ])
    res = res.with_columns([
        pl.when(pl.col("stripped_name") != "").then(pl.col("stripped_name")).otherwise(pl.col("clean_name"))
        .str.split(" ").list.sort().list.join(" ").alias("sorted_name"),
        pl.col("stripped_name").str.split(" ").list.get(0).fill_null("").alias("first_word"),
        pl.col("std_address").str.split(" ").list.filter(pl.element().str.len_chars() >= 3)
        .list.sort().list.join(" ").alias("sorted_addr"),
        pl.col("std_address").str.split(" ").list.slice(1, 2).list.join(" ").alias("street_prefix"),
    ]).drop(["_nm", "_ad"])
    print(f"Normalized {len(df):,} records in {time.time()-t0:.1f}s")
    return res

print("\nNormalizing validation queries...")
q = normalize_df(val_df)
print("\nNormalizing targets...")
t = normalize_df(targets)

def run_ch(name, q_df, t_df, on_cols):
    t_ch = time.time()
    pairs = (
        q_df.select(["entity_id"] + on_cols)
        .join(t_df.select(["entity_id"] + on_cols).rename({"entity_id": "target_id"}), on=on_cols, how="inner")
        .select(["entity_id", "target_id"]).unique()
    )
    print(f"  {name:<35}: {len(pairs):>10,} pairs ({time.time()-t_ch:.2f}s)")
    return pairs

channels = []
channels.append(run_ch("Ch1 Exact Name", q.filter(pl.col("clean_name") != ""), t.filter(pl.col("clean_name") != ""), ["country", "clean_name"]))
channels.append(run_ch("Ch2 Sorted Name", q.filter(pl.col("sorted_name") != ""), t.filter(pl.col("sorted_name") != ""), ["country", "sorted_name"]))
channels.append(run_ch("Ch3 Spaceless Name", q.filter(pl.col("spaceless_name").str.len_chars() >= 6), t.filter(pl.col("spaceless_name").str.len_chars() >= 6), ["country", "spaceless_name"]))
channels.append(run_ch("Ch4 Exact Address", q.filter(pl.col("std_address").str.len_chars() >= 10), t.filter(pl.col("std_address").str.len_chars() >= 10), ["country", "std_address"]))
channels.append(run_ch("Ch5 Sorted Address", q.filter(pl.col("sorted_addr").str.len_chars() >= 15), t.filter(pl.col("sorted_addr").str.len_chars() >= 15), ["country", "sorted_addr"]))
channels.append(run_ch("Ch6 HouseNo + First Word", q.filter((pl.col("house_no").str.len_chars() >= 2) & (pl.col("first_word").str.len_chars() >= 4)), t.filter((pl.col("house_no").str.len_chars() >= 2) & (pl.col("first_word").str.len_chars() >= 4)), ["country", "house_no", "first_word"]))
channels.append(run_ch("Ch7 Postal + First Word", q.filter((pl.col("postal_code").str.len_chars() >= 5) & (pl.col("first_word").str.len_chars() >= 4)), t.filter((pl.col("postal_code").str.len_chars() >= 5) & (pl.col("first_word").str.len_chars() >= 4)), ["country", "postal_code", "first_word"]))
channels.append(run_ch("Ch8 HouseNo + Street Prefix", q.filter((pl.col("house_no").str.len_chars() >= 2) & (pl.col("street_prefix").str.len_chars() >= 5)), t.filter((pl.col("house_no").str.len_chars() >= 2) & (pl.col("street_prefix").str.len_chars() >= 5)), ["country", "house_no", "street_prefix"]))

pooled = pl.concat(channels).unique()
print(f"\nTotal candidate pairs: {len(pooled):,}")

cand_map = defaultdict(set)
for row in pooled.iter_rows(named=True):
    cand_map[str(row["entity_id"])].add(str(row["target_id"]))

# Evaluate blackout queries and Oracle F0.5
tot_true = 0
ret_true = 0
blackouts = 0
f05_list = []

for qid in sorted(val_gt.keys()):
    true_s = val_gt.get(qid, set())
    pred = true_s & cand_map.get(qid, set())
    tot_true += len(true_s)
    ret_true += len(pred)
    if not true_s:
        f05_list.append(1.0)
    elif not pred:
        f05_list.append(0.0)
        blackouts += 1
    else:
        rec = len(pred) / len(true_s)
        f05_list.append(1.25 * rec / (0.25 + rec))

import numpy as np
print("\n" + "="*70)
print(f"ORACLE EVALUATION WITH TRANSLITERATION + DOMAIN STRIPPING:")
print(f"  Macro F0.5:        {np.mean(f05_list):.5f}")
print(f"  Link Recall:       {ret_true / tot_true * 100:.2f}% ({ret_true:,} / {tot_true:,})")
print(f"  Blackout queries:  {blackouts:,} (out of {len(val_gt):,})")
print("="*70)
