"""
experiments/phase2/benchmark_10ch_retrieval.py
==============================================
Benchmarks the new 10-Channel Candidate Generation Pipeline on
the full 50K validation set against ALL 10.07M training targets.
"""
import time, os, sys, re, unicodedata
from collections import defaultdict
import numpy as np
import polars as pl

# Transliteration Table for Indic Scripts: Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada, Malayalam
INDIC_MAP = {
    # Devanagari
    'अ':'a','आ':'aa','इ':'i','ई':'ee','उ':'u','ऊ':'oo','ए':'e','ऐ':'ai','ओ':'o','औ':'au',
    'क':'k','ख':'kh','ग':'g','घ':'gh','ङ':'ng','च':'ch','छ':'chh','ज':'j','झ':'jh','ञ':'ny',
    'ट':'t','ठ':'th','ड':'d','ढ':'dh','ण':'n','त':'t','थ':'th','द':'d','ध':'dh','न':'n',
    'प':'p','फ':'ph','ब':'b','भ':'bh','म':'m','य':'y','र':'r','ल':'l','व':'v','श':'sh',
    'ष':'sh','स':'s','ह':'h','ळ':'l','ा':'a','ि':'i','ी':'ee','ु':'u','ू':'oo','े':'e',
    'ै':'ai','ो':'o','ौ':'au','ं':'n','ः':'h','्':'','ँ':'n','ृ':'ri',
    # Bengali
    'অ':'a','আ':'aa','ই':'i','ঈ':'ee','উ':'u','ঊ':'oo','এ':'e','ঐ':'ai','ও':'o','ঔ':'au',
    'ক':'k','খ':'kh','গ':'g','ঘ':'gh','ঙ':'ng','চ':'ch','ছ':'chh','জ':'j','ঝ':'jh','ঞ':'ny',
    'ট':'t','ঠ':'th','ড':'d','ঢ':'dh','ণ':'n','ত':'t','থ':'th','দ':'d','ধ':'dh','ন':'n',
    'প':'p','ফ':'ph','ব':'b','ভ':'bh','ম':'m','য':'y','র':'r','ল':'l','শ':'sh','ষ':'sh',
    'স':'s','হ':'h','ড়':'r','ঢ়':'rh','য়':'y','া':'a','ি':'i','ী':'ee','ু':'u','ূ':'oo',
    'ে':'e','ৈ':'ai','ো':'o','ৌ':'au','্':'',
    # Telugu
    'అ':'a','ఆ':'aa','ఇ':'i','ఈ':'ee','ఉ':'u','ఊ':'oo','ఎ':'e','ఏ':'ee','ఐ':'ai','ఒ':'o','ఓ':'oo','ఔ':'au',
    'క':'k','ఖ':'kh','గ':'g','ఘ':'gh','ఙ':'ng','చ':'ch','ఛ':'chh','జ':'j','ఝ':'jh','ఞ':'ny',
    'ట':'t','ఠ':'th','డ':'d','ఢ':'dh','ణ':'n','త':'t','థ':'th','ద':'d','ధ':'dh','న':'n',
    'ప':'p','ఫ':'ph','బ':'b','భ':'bh','మ':'m','య':'y','ర':'r','ల':'l','వ':'v','శ':'sh',
    'ష':'sh','స':'s','హ':'h','ళ':'l','ా':'a','ి':'i','ీ':'ee','ు':'u','ూ':'oo','ె':'e',
    'ే':'ee','ై':'ai','ొ':'o','ో':'oo','ౌ':'au','ం':'m','్':'',
    # Tamil
    'அ':'a','ஆ':'aa','இ':'i','ஈ':'ee','உ':'u','ஊ':'oo','எ':'e','ஏ':'ee','ஐ':'ai','ஒ':'o','ஓ':'oo','ஔ':'au',
    'க':'k','ங':'ng','ச':'ch','ஞ':'ny','ட':'t','ண':'n','த':'t','ந':'n','ப':'p','ம':'m',
    'ய':'y','ர':'r','ல':'l','வ':'v','ழ':'zh','ள':'l','ற':'r','ன':'n','ா':'a','ி':'i',
    'ீ':'ee','ு':'u','ூ':'oo','ெ':'e','ே':'ee','ை':'ai','ொ':'o','ோ':'oo','ௌ':'au','்':'',
    # Kannada
    'ಅ':'a','ಆ':'aa','ಇ':'i','ಈ':'ee','ಉ':'u','ಊ':'oo','ಎ':'e','ಏ':'ee','ಐ':'ai','ಒ':'o','ಓ':'oo','ಔ':'au',
    'ಕ':'k','ಖ':'kh','ಗ':'g','ಘ':'gh','ಙ':'ng','ಚ':'ch','ಛ':'chh','ಜ':'j','ಝ':'jh','ಞ':'ny',
    'ಟ':'t','ಠ':'th','ಡ':'d','ಢ':'dh','ಣ':'n','ತ':'t','ಥ':'th','ದ':'d','ಧ':'dh','ನ':'n',
    'ಪ':'p','ಫ':'ph','ಬ':'b','ಭ':'bh','ಮ':'m','ಯ':'y','ರ':'r','ಲ':'l','ವ':'v','ಶ':'sh',
    'ಷ':'sh','ಸ':'s','ಹ':'h','ಳ':'l','ಾ':'a','ಿ':'i','ೀ':'ee','ು':'u','ೂ':'oo','ೆ':'e',
    'ೇ':'ee','ೈ':'ai','ೊ':'o','ೋ':'oo','ೌ':'au','ಂ':'m','್':'',
    # Malayalam
    'അ':'a','ആ':'aa','ഇ':'i','ഈ':'ee','ഉ':'u','ഊ':'oo','എ':'e','ഏ':'ee','ഐ':'ai','ഒ':'o','ഓ':'oo','ഔ':'au',
    'ക':'k','ഖ':'kh','ഗ':'g','ഘ':'gh','ങ':'ng','ച':'ch','ഛ':'chh','ജ':'j','ഝ':'jh','ഞ':'ny',
    'ട':'t','ഠ':'th','ഡ':'d','ഢ':'dh','ണ':'n','ത':'t','ഥ':'th','ദ':'d','ധ':'dh','ന':'n',
    'പ':'p','ഫ':'ph','ബ':'b','ഭ':'bh','മ':'m','യ':'y','ര':'r','ല':'l','വ':'v','ശ':'sh',
    'ഷ':'sh','സ':'s','ഹ':'h','ള':'l','ഴ':'zh','റ':'r','ാ':'a','ി':'i','ീ':'ee','ു':'u',
    'ൂ':'oo','െ':'e','േ':'ee','ൈ':'ai','ൊ':'o','ോ':'oo','ൌ':'au','ം':'m','്':'',
}

TRANS_TABLE = str.maketrans(INDIC_MAP)

def transliterate(s: str) -> str:
    if not s: return ""
    return s.translate(TRANS_TABLE)

def strip_accents(s: str) -> str:
    if not s: return ""
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

LEGAL_SUFFIX_REGEX = (
    r"(?i)\b(?:"
    r"private limited|pvt ltd|pvt limited|private ltd|limited|ltd|pvt|"
    r"llc|l\.l\.c\.|inc|incorporated|corporation|corp|llp|l\.l\.p\.|"
    r"co|company|gmbh|sarl|sa|sas|plc|center|services|service|partners|group|holdings"
    r")\b"
)

STREET_ABBRS = {
    r"\bst\b": "street", r"\bave\b": "avenue", r"\brd\b": "road",
    r"\bdr\b": "drive", r"\bblvd\b": "boulevard", r"\bln\b": "lane",
    r"\bdelaware\b": "de", r"\bindiana\b": "in", r"\bnew york\b": "ny",
    r"\butah\b": "ut", r"\bvirginia\b": "va", r"\bmaine\b": "me",
    r"\bcalifornia\b": "ca", r"\btexas\b": "tx", r"\bflorida\b": "fl",
    r"\bohio\b": "oh", r"\bdelhi\b": "dl", r"\bwest bengal\b": "wb",
    r"\bkarnataka\b": "ka", r"\bmaharashtra\b": "mh", r"\bgujarat\b": "gj",
    r"\btamil nadu\b": "tn",
}

def normalize_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    print(f"  Normalizing {len(df):,} records...", flush=True)
    t0 = time.time()
    
    # 1. Transliterate and strip accents
    names = [strip_accents(transliterate(str(n or ''))).lower() for n in df['business_name'].to_list()]
    addrs = [strip_accents(transliterate(str(a or ''))).lower() for a in df['business_address'].to_list()]
    
    # 2. De-leet function
    def deleet_str(s):
        s = re.sub(r'(?<=[a-zA-Z])0(?=[a-zA-Z])|(?<=[a-zA-Z])0|0(?=[a-zA-Z])', 'o', s)
        s = re.sub(r'(?<=[a-zA-Z])1(?=[a-zA-Z])|(?<=[a-zA-Z])1|1(?=[a-zA-Z])', 'l', s)
        s = re.sub(r'(?<=[a-zA-Z])3(?=[a-zA-Z])|(?<=[a-zA-Z])3|3(?=[a-zA-Z])', 'e', s)
        s = re.sub(r'(?<=[a-zA-Z])4(?=[a-zA-Z])|(?<=[a-zA-Z])4|4(?=[a-zA-Z])', 'a', s)
        s = re.sub(r'(?<=[a-zA-Z])5(?=[a-zA-Z])|(?<=[a-zA-Z])5|5(?=[a-zA-Z])', 's', s)
        return s

    names = [deleet_str(n) for n in names]
    
    # Put back into Polars
    res = df.with_columns([
        pl.Series("raw_clean_name", names),
        pl.Series("raw_clean_addr", addrs),
        pl.col("country").fill_null("").str.strip_chars().alias("country"),
    ])
    
    # Clean name: remove domains, punctuation
    res = res.with_columns([
        pl.col("raw_clean_name")
        .str.replace_all(r"\.(com|org|net|in|co|us|gov|io)\b", " ")
        .str.replace_all(r"\b(www|http|https)\b", " ")
        .str.replace_all("&", " and ")
        .str.replace_all(r"[^\w\s]", " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
        .alias("clean_name"),
        
        pl.col("raw_clean_addr")
        .str.replace_all(r"[^\w\s]", " ")
        .str.replace_all(r"\b0+(\d+)\b", r"$1")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
        .alias("std_address"),
    ])
    
    # Derived representations
    res = res.with_columns([
        pl.col("clean_name")
        .str.replace_all(LEGAL_SUFFIX_REGEX, " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
        .alias("stripped_name"),
        
        # Spaceless name (for concatenated & domain names)
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
    
    res = res.with_columns([
        # Sorted name
        pl.when(pl.col("stripped_name") != "")
        .then(pl.col("stripped_name"))
        .otherwise(pl.col("clean_name"))
        .str.split(" ")
        .list.sort()
        .list.join(" ")
        .alias("sorted_name"),
        
        # First word of name (min 3 chars)
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
    ])
    
    print(f"  Normalization done in {time.time()-t0:.1f}s", flush=True)
    return res

print("=" * 70)
print("BENCHMARK 10-CHANNEL RETRIEVAL ON FULL 50K VALIDATION SET")
print("=" * 70)

t_all = time.time()

# 1. Load validation set
val_df = pl.read_parquet("experiments/val_sample_50k.parquet")
print(f"Loaded val queries: {len(val_df):,}")

# Ground truth map
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

# Normalize queries & targets
print("\nNormalizing queries...")
q = normalize_dataframe(val_df)
print("\nNormalizing targets...")
t = normalize_dataframe(targets_all)

# Channels
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

print("\nExecuting 10 retrieval channels:")
# Ch1: Exact clean name
channels.append(run_ch("Ch1 Exact Name", 1.00, q.filter(pl.col("clean_name") != ""), t.filter(pl.col("clean_name") != ""), ["country", "clean_name"]))

# Ch2: Sorted name
channels.append(run_ch("Ch2 Sorted Name", 0.98, q.filter(pl.col("sorted_name") != ""), t.filter(pl.col("sorted_name") != ""), ["country", "sorted_name"]))

# Ch3: Spaceless Name (min 6 chars)
channels.append(run_ch("Ch3 Spaceless Name", 0.95, q.filter(pl.col("spaceless_name").str.len_chars() >= 6), t.filter(pl.col("spaceless_name").str.len_chars() >= 6), ["country", "spaceless_name"]))

# Ch4: Exact address (min 10 chars)
channels.append(run_ch("Ch4 Exact Address", 0.92, q.filter(pl.col("std_address").str.len_chars() >= 10), t.filter(pl.col("std_address").str.len_chars() >= 10), ["country", "std_address"]))

# Ch5: Sorted address (min 15 chars)
channels.append(run_ch("Ch5 Sorted Address", 0.90, q.filter(pl.col("sorted_addr").str.len_chars() >= 15), t.filter(pl.col("sorted_addr").str.len_chars() >= 15), ["country", "sorted_addr"]))

# Ch6: Postal Code + First Word (min 4 chars)
channels.append(run_ch("Ch6 Postal + First Word", 0.85, 
    q.filter((pl.col("postal_code").str.len_chars() >= 5) & (pl.col("first_word").str.len_chars() >= 4)), 
    t.filter((pl.col("postal_code").str.len_chars() >= 5) & (pl.col("first_word").str.len_chars() >= 4)), 
    ["country", "postal_code", "first_word"]))

# Ch7: House No + First Word (min 4 chars)
channels.append(run_ch("Ch7 HouseNo + First Word", 0.80, 
    q.filter((pl.col("house_no").str.len_chars() >= 2) & (pl.col("first_word").str.len_chars() >= 4)), 
    t.filter((pl.col("house_no").str.len_chars() >= 2) & (pl.col("first_word").str.len_chars() >= 4)), 
    ["country", "house_no", "first_word"]))

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

print(f"\nTotal Elapsed Time: {time.time()-t_all:.1f}s")
