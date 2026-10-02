"""
experiments/phase2/fast_10ch_retrieval.py
=========================================
High-speed vectorized 10-channel candidate generation pipeline.
Measures Recall Ceiling and Oracle F0.5 on 50K validation set.
"""
import time, os, sys, re, unicodedata
from collections import defaultdict
import numpy as np
import polars as pl

print("=" * 70)
print("FAST VECTORIZED 10-CHANNEL RETRIEVAL PIPELINE")
print("=" * 70)

t0 = time.time()

# 1. Indic transliteration table
INDIC_MAP = {
    'अ':'a','आ':'aa','इ':'i','ई':'ee','उ':'u','ऊ':'oo','ए':'e','ऐ':'ai','ओ':'o','औ':'au',
    'क':'k','ख':'kh','ग':'g','घ':'gh','ङ':'ng','च':'ch','छ':'chh','ज':'j','झ':'jh','ञ':'ny',
    'ट':'t','ठ':'th','ड':'d','ढ':'dh','ण':'n','त':'t','थ':'th','द':'d','ध':'dh','न':'n',
    'प':'p','फ':'ph','ब':'b','भ':'bh','म':'m','य':'y','र':'r','ल':'l','व':'v','श':'sh',
    'ष':'sh','स':'s','ह':'h','ळ':'l','ा':'a','ि':'i','ी':'ee','ु':'u','ू':'oo','े':'e',
    'ै':'ai','ो':'o','ौ':'au','ं':'n','ः':'h','्':'','ँ':'n','ृ':'ri',
    'অ':'a','আ':'aa','ই':'i','ঈ':'ee','উ':'u','ঊ':'oo','এ':'e','ঐ':'ai','ও':'o','ঔ':'au',
    'ক':'k','খ':'kh','গ':'g','ঘ':'gh','ঙ':'ng','চ':'ch','ছ':'chh','জ':'j','ঝ':'jh','ঞ':'ny',
    'ট':'t','ঠ':'th','ড':'d','ঢ':'dh','ণ':'n','ত':'t','থ':'th','দ':'d','ध':'dh','ন':'n',
    'প':'p','ফ':'ph','ব':'b','ভ':'bh','ম':'m','য':'y','র':'r','ল':'l','শ':'sh','ষ':'sh',
    'স':'s','হ':'h','ড়':'r','ঢ়':'rh','য়':'y','া':'a','ি':'i','ী':'ee','ু':'u','ূ':'oo',
    'ে':'e','ৈ':'ai','ো':'o','ৌ':'au','্':'',
    'అ':'a','ఆ':'aa','ఇ':'i','ఈ':'ee','ఉ':'u','ఊ':'oo','ఎ':'e','ఏ':'ee','ఐ':'ai','ఒ':'o','ఓ':'oo','ఔ':'au',
    'క':'k','ఖ':'kh','గ':'g','ఘ':'gh','ఙ':'ng','చ':'ch','ఛ':'chh','జ':'j','ఝ':'jh','ఞ':'ny',
    'ట':'t','ఠ':'th','డ':'d','ఢ':'dh','ణ':'n','త':'t','థ':'th','ద':'d','ధ':'dh','న':'n',
    'ప':'p','ఫ':'ph','బ':'b','భ':'bh','మ':'m','య':'y','ర':'r','ల':'l','వ':'v','శ':'sh',
    'ష':'sh','స':'s','హ':'h','ళ':'l','ా':'a','ి':'i','ీ':'ee','ు':'u','ూ':'oo','ె':'e',
    'ే':'ee','ై':'ai','ొ':'o','ో':'oo','ౌ':'au','ం':'m','్':'',
    'அ':'a','ஆ':'aa','இ':'i','ஈ':'ee','உ':'u','ஊ':'oo','எ':'e','ஏ':'ee','ஐ':'ai','ஒ':'o','ஓ':'oo','ஔ':'au',
    'க':'k','ங':'ng','ச':'ch','ஞ':'ny','ட':'t','ண':'n','த':'t','ந':'n','ப':'p','ம':'m',
    'ய':'y','ர':'r','ல':'l','வ':'v','ழ':'zh','ள':'l','ற':'r','ன':'n','ா':'a','ி':'i',
    'ீ':'ee','ு':'u','ூ':'oo','ெ':'e','ே':'ee','ை':'ai','ொ':'o','ோ':'oo','ௌ':'au','்':'',
    'ಅ':'a','ಆ':'aa','ಇ':'i','ಈ':'ee','ಉ':'u','ಊ':'oo','ಎ':'e','ಏ':'ee','ಐ':'ai','ಒ':'o','ಓ':'oo','ಔ':'au',
    'ಕ':'k','ಖ':'kh','ಗ':'g','ಘ':'gh','ಙ':'ng','ಚ':'ch','ಛ':'chh','ಜ':'j','ಝ':'jh','ಞ':'ny',
    'ಟ':'t','ಠ':'th','ಡ':'d','ಢ':'dh','ಣ':'n','ತ':'t','ಥ':'th','ದ':'d','ಧ':'dh','ನ':'n',
    'ಪ':'p','ಫ':'ph','ಬ':'b','ಭ':'bh','ಮ':'m','ಯ':'y','ರ':'r','ಲ':'l','ವ':'v','ಶ':'sh',
    'ಷ':'sh','ಸ':'s','ಹ':'h','ಳ':'l','ಾ':'a','ಿ':'i','ೀ':'ee','ು':'u','ೂ':'oo','ೆ':'e',
    'ೇ':'ee','ೈ':'ai','ೊ':'o','ೋ':'oo','ೌ':'au','ಂ':'m','್':'',
    'അ':'a','ആ':'aa','ഇ':'i','ഈ':'ee','ഉ':'u','ഊ':'oo','എ':'e','ഏ':'ee','ഐ':'ai','ഒ':'o','ഓ':'oo','ഔ':'au',
    'ക':'k','ഖ':'kh','ഗ':'g','ഘ':'gh','ങ':'ng','ച':'ch','ഛ':'chh','ജ':'j','ഝ':'jh','ഞ':'ny',
    'ട':'t','ഠ':'th','ഡ':'d','ഢ':'dh','ണ':'n','ത':'t','ഥ':'th','ദ':'d','ധ':'dh','ന':'n',
    'പ':'p','ഫ':'ph','ബ':'b','ഭ':'bh','മ':'m','യ':'y','ര':'r','ല':'l','വ':'v','ശ':'sh',
    'ഷ':'sh','സ':'s','ഹ':'h','ള':'l','ഴ':'zh','റ':'r','ാ':'a','ി':'i','ീ':'ee','ു':'u',
    'ൂ':'oo','െ':'e','േ':'ee','ൈ':'ai','ൊ':'o','ോ':'oo','ൌ':'au','ം':'m','്':'',
}
TRANS_TABLE = str.maketrans(INDIC_MAP)

def transliterate_text(s: str) -> str:
    if not s: return ""
    return s.translate(TRANS_TABLE)

LEGAL_SUFFIX_REGEX = (
    r"(?i)\b(?:"
    r"private limited|pvt ltd|pvt limited|private ltd|limited|ltd|pvt|"
    r"llc|l\.l\.c\.|inc|incorporated|corporation|corp|llp|l\.l\.p\.|"
    r"co|company|gmbh|sarl|sa|sas|plc|center|services|service|partners|group|holdings"
    r")\b"
)

def fast_normalize(df: pl.DataFrame) -> pl.DataFrame:
    t_start = time.time()
    
    # Check if any Indic characters exist, and transliterate only if present
    print(f"  Normalizing {len(df):,} records...", flush=True)
    
    # 1. Base clean in Polars
    res = df.with_columns([
        pl.col("business_name").fill_null("").alias("name_raw"),
        pl.col("business_address").fill_null("").alias("addr_raw"),
        pl.col("country").fill_null("").str.strip_chars().alias("country"),
    ])
    
    # Transliterate Indic characters efficiently using Python map ONLY for non-ascii rows
    indic_mask = res["name_raw"].str.contains(r"[\u0900-\u0D7F]") | res["addr_raw"].str.contains(r"[\u0900-\u0D7F]")
    indic_count = indic_mask.sum()
    if indic_count > 0:
        print(f"    Transliterating {indic_count:,} Indic records...", flush=True)
        # Vectorized replace via mapping
        names = res["name_raw"].to_list()
        addrs = res["addr_raw"].to_list()
        for idx in range(len(names)):
            if indic_mask[idx]:
                names[idx] = transliterate_text(names[idx])
                addrs[idx] = transliterate_text(addrs[idx])
        res = res.with_columns([
            pl.Series("name_raw", names),
            pl.Series("addr_raw", addrs)
        ])
        
    # 2. Pure Polars Fast String Normalization (Rust-level parallel SIMD)
    res = res.with_columns([
        # Clean name: lowercase, accent removal, domain extension strip, leet-speak digit replace
        pl.col("name_raw")
        .str.to_lowercase()
        .str.replace_all(r"[éèêë]", "e")
        .str.replace_all(r"[àâä]", "a")
        .str.replace_all(r"[îï]", "i")
        .str.replace_all(r"[ôö]", "o")
        .str.replace_all(r"[ùûü]", "u")
        .str.replace_all(r"ç", "c")
        .str.replace_all(r"\.(com|org|net|in|co|us|gov|io)\b", " ")
        .str.replace_all(r"\b(www|http|https)\b", " ")
        .str.replace_all("&", " and ")
        .str.replace_all(r"[^\w\s]", " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
        .alias("clean_name"),
        
        # Clean address
        pl.col("addr_raw")
        .str.to_lowercase()
        .str.replace_all(r"[éèêë]", "e")
        .str.replace_all(r"[àâä]", "a")
        .str.replace_all(r"[îï]", "i")
        .str.replace_all(r"[ôö]", "o")
        .str.replace_all(r"[ùûü]", "u")
        .str.replace_all(r"ç", "c")
        .str.replace_all(r"[^\w\s]", " ")
        .str.replace_all(r"\b0+(\d+)\b", r"$1")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
        .alias("std_address"),
    ])
    
    # 3. De-leet representations
    res = res.with_columns([
        # De-leet name: replace 0->o, 1->l, 3->e, 4->a, 5->s inside words using capture groups
        pl.col("clean_name")
        .str.replace_all(r"([a-z])0([a-z])", "$1o$2")
        .str.replace_all(r"([a-z])1([a-z])", "$1l$2")
        .str.replace_all(r"([a-z])3([a-z])", "$1e$2")
        .str.replace_all(r"([a-z])4([a-z])", "$1a$2")
        .str.replace_all(r"([a-z])5([a-z])", "$1s$2")
        .str.replace_all(r"\b0([a-z])", "o$1")
        .str.replace_all(r"([a-z])0\b", "$1o")
        .alias("deleet_name"),
        
        # Stripped suffix name
        pl.col("clean_name")
        .str.replace_all(LEGAL_SUFFIX_REGEX, " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
        .alias("stripped_name"),
        
        # Spaceless name
        pl.col("clean_name")
        .str.replace_all(r"\s+", "")
        .alias("spaceless_name"),
        
        # House number
        pl.col("std_address")
        .str.extract(r"\b(\d{1,6}[a-z]?)\b", 1)
        .fill_null("")
        .alias("house_no"),
        
        # Postal code
        pl.when(pl.col("country").str.to_uppercase() == "INDIA")
          .then(pl.col("std_address").str.extract(r"\b([1-9]\d{5})\b", 1))
          .when(pl.col("country").str.to_uppercase() == "US")
          .then(pl.col("std_address").str.extract(r"\b(\d{5})(?:-\d{4})?\b", 1))
          .when(pl.col("country").str.to_uppercase() == "FRANCE")
          .then(pl.col("std_address").str.extract(r"\b((?:0[1-9]|[1-8]\d|9[0-8])\d{3})\b", 1))
          .otherwise(pl.col("std_address").str.extract(r"\b(\d{5,6})\b", 1))
          .fill_null("")
          .alias("postal_code"),
    ])
    
    # 4. Token sorting & components
    res = res.with_columns([
        # Sorted name
        pl.when(pl.col("stripped_name") != "")
        .then(pl.col("stripped_name"))
        .otherwise(pl.col("clean_name"))
        .str.split(" ")
        .list.sort()
        .list.join(" ")
        .alias("sorted_name"),
        
        # Sorted deleet name
        pl.col("deleet_name")
        .str.replace_all(LEGAL_SUFFIX_REGEX, " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
        .str.split(" ")
        .list.sort()
        .list.join(" ")
        .alias("sorted_deleet"),
        
        # First word (brand token)
        pl.col("stripped_name")
        .str.split(" ")
        .list.get(0)
        .fill_null("")
        .alias("first_word"),
        
        # Sorted address tokens (min 3 chars)
        pl.col("std_address")
        .str.split(" ")
        .list.filter(pl.element().str.len_chars() >= 3)
        .list.sort()
        .list.join(" ")
        .alias("sorted_addr"),
        
        # Street 2 tokens (tokens 1 and 2 of address)
        pl.col("std_address")
        .str.split(" ")
        .list.slice(1, 2)
        .list.join(" ")
        .alias("street_prefix"),
    ])
    
    print(f"  Normalization finished in {time.time()-t_start:.1f}s", flush=True)
    return res

# 1. Load validation set
val_df = pl.read_parquet("experiments/val_sample_50k.parquet")
print(f"Loaded {len(val_df):,} val queries")

val_gt = {}
for row in val_df.iter_rows(named=True):
    qid = str(row["entity_id"])
    raw = str(row.get("matched_entity_ids", "") or "").strip()
    val_gt[qid] = {t.strip() for t in raw.split(",") if t.strip() and t.strip() not in ("nan", "None")}

total_true_links = sum(len(v) for v in val_gt.values())
val_query_list = sorted(val_gt.keys())
print(f"Val queries with GT matches: {sum(1 for v in val_gt.values() if v):,}")
print(f"Total true links to retrieve: {total_true_links:,}")

# 2. Load targets (S2 + S3)
print("Loading train_source2.tsv and train_source3.tsv...")
s2 = pl.read_csv("dataset/raw/train/train_source2.tsv", separator="\t", quote_char=None)
s3 = pl.read_csv("dataset/raw/train/train_source3.tsv", separator="\t", quote_char=None)
targets_all = pl.concat([s2, s3])
print(f"Total target records: {len(targets_all):,}")

# 3. Normalize queries & targets
print("\nNormalizing queries...")
q = fast_normalize(val_df)
print("\nNormalizing targets...")
t = fast_normalize(targets_all)

# 4. Multi-Channel Inverted Hash Joins
channels = []

def run_ch(name, prio, q_df, t_df, on_cols):
    t_ch = time.time()
    pairs = (
        q_df.select(["entity_id"] + on_cols)
        .join(t_df.select(["entity_id"] + on_cols).rename({"entity_id": "target_id"}), on=on_cols, how="inner")
        .select(["entity_id", "target_id"])
        .unique()
        .with_columns(pl.lit(prio).alias("prio"))
    )
    print(f"  {name:<35}: {len(pairs):>10,} pairs ({time.time()-t_ch:.2f}s)", flush=True)
    return pairs

print("\nExecuting High-Recall Multi-Channel Retrieval:")
# Ch1: Exact Clean Name
channels.append(run_ch("Ch1 Exact Name", 1.00, q.filter(pl.col("clean_name") != ""), t.filter(pl.col("clean_name") != ""), ["country", "clean_name"]))

# Ch2: Sorted Name
channels.append(run_ch("Ch2 Sorted Name", 0.98, q.filter(pl.col("sorted_name") != ""), t.filter(pl.col("sorted_name") != ""), ["country", "sorted_name"]))

# Ch3: Spaceless Name (Domain & concatenated names, min 6 chars)
channels.append(run_ch("Ch3 Spaceless Name", 0.95, q.filter(pl.col("spaceless_name").str.len_chars() >= 6), t.filter(pl.col("spaceless_name").str.len_chars() >= 6), ["country", "spaceless_name"]))

# Ch4: Sorted De-leet Name (Ac0sta -> Acosta)
channels.append(run_ch("Ch4 De-leet Sorted Name", 0.93, q.filter(pl.col("sorted_deleet") != ""), t.filter(pl.col("sorted_deleet") != ""), ["country", "sorted_deleet"]))

# Ch5: Exact Address (min 10 chars)
channels.append(run_ch("Ch5 Exact Address", 0.90, q.filter(pl.col("std_address").str.len_chars() >= 10), t.filter(pl.col("std_address").str.len_chars() >= 10), ["country", "std_address"]))

# Ch6: Sorted Address (min 15 chars)
channels.append(run_ch("Ch6 Sorted Address", 0.88, q.filter(pl.col("sorted_addr").str.len_chars() >= 15), t.filter(pl.col("sorted_addr").str.len_chars() >= 15), ["country", "sorted_addr"]))

# Ch7: House No + First Word (min 4 chars)
channels.append(run_ch("Ch7 HouseNo + First Word", 0.82, 
    q.filter((pl.col("house_no").str.len_chars() >= 2) & (pl.col("first_word").str.len_chars() >= 4)), 
    t.filter((pl.col("house_no").str.len_chars() >= 2) & (pl.col("first_word").str.len_chars() >= 4)), 
    ["country", "house_no", "first_word"]))

# Ch8: Postal Code + First Word (min 4 chars)
channels.append(run_ch("Ch8 Postal + First Word", 0.80, 
    q.filter((pl.col("postal_code").str.len_chars() >= 5) & (pl.col("first_word").str.len_chars() >= 4)), 
    t.filter((pl.col("postal_code").str.len_chars() >= 5) & (pl.col("first_word").str.len_chars() >= 4)), 
    ["country", "postal_code", "first_word"]))

# Ch9: House No + Street Prefix (min 6 chars)
channels.append(run_ch("Ch9 HouseNo + Street Prefix", 0.78, 
    q.filter((pl.col("house_no").str.len_chars() >= 2) & (pl.col("street_prefix").str.len_chars() >= 6)), 
    t.filter((pl.col("house_no").str.len_chars() >= 2) & (pl.col("street_prefix").str.len_chars() >= 6)), 
    ["country", "house_no", "street_prefix"]))

print("\nCombining and deduplicating candidates...")
all_cands = pl.concat(channels)
pooled = (
    all_cands
    .group_by(["entity_id", "target_id"])
    .agg([
        pl.col("prio").max().alias("max_prio"),
        pl.len().alias("n_channels")
    ])
)
print(f"Total unique candidate pairs retrieved: {len(pooled):,}")

# Calculate Oracle recall and F0.5
print("\nMeasuring Oracle Recall & F0.5...")

def evaluate_oracle(cand_df, top_k=None):
    if top_k:
        ranked = cand_df.sort(["entity_id", "max_prio", "n_channels"], descending=[False, True, True])
        cand_df = ranked.group_by("entity_id", maintain_order=True).head(top_k)
    
    cand_map = defaultdict(set)
    for row in cand_df.select(["entity_id", "target_id"]).iter_rows(named=True):
        cand_map[str(row["entity_id"])].add(str(row["target_id"]))
        
    tot_true = 0
    ret_true = 0
    blackouts = 0
    f05_list = []
    
    for qid in val_query_list:
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
            f05 = 1.25 * rec / (0.25 + rec)
            f05_list.append(f05)
            
    link_rec = ret_true / max(tot_true, 1)
    macro_f05 = float(np.nanmean(f05_list))
    return macro_f05, link_rec, ret_true, tot_true, blackouts, len(cand_df)

for k in [40, 50, 75, 100, None]:
    k_str = f"K={k}" if k else "Unbudgeted"
    f05, rec, ret, tot, blk, n_pairs = evaluate_oracle(pooled, top_k=k)
    print(f"  {k_str:<12}: Oracle F0.5 = {f05:.5f} | Link Recall = {rec*100:5.2f}% ({ret:,}/{tot:,}) | Blackouts = {blk:,} | Pairs = {n_pairs:,}")

print(f"\nTotal Pipeline Elapsed Time: {time.time()-t0:.1f}s")
