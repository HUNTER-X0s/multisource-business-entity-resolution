"""
experiments/analyze_unbudgeted_missed_links.py
==============================================
Analyzes why 22.3% of true links are missed by the 6 primary channels.
Samples missed (S1, target) pairs and inspects string differences.
"""

import os
import sys
import polars as pl

sys.path.insert(0, os.path.abspath("."))
from src.vectorized_pipeline import normalize_dataframe

val_df = pl.read_parquet("experiments/val_sample_50k.parquet")
targets_clean = pl.read_parquet("experiments/cache/targets_train_normalized.parquet")
val_clean = normalize_dataframe(val_df)

# Build map of S1 ground truth
gt_pairs = []
for row in val_df.iter_rows(named=True):
    s1_id = row["entity_id"]
    m_str = row.get("matched_entity_ids") or ""
    for m in m_str.split(","):
        m = m.strip()
        if m:
            gt_pairs.append((s1_id, m))

gt_df = pl.DataFrame(gt_pairs, schema=["entity_id", "target_id"], orient="row")
print(f"Total ground truth links in validation set: {len(gt_df):,}")

# Read the channels from the benchmark logic
STREET_ABBRS = {
    r"\bst\b": "street", r"\bave\b": "avenue", r"\brd\b": "road", r"\bdr\b": "drive",
    r"\bblvd\b": "boulevard", r"\bln\b": "lane", r"\bdelaware\b": "de", r"\bindiana\b": "in",
    r"\bnew york\b": "ny", r"\butah\b": "ut", r"\bvirginia\b": "va", r"\bmaine\b": "me",
    r"\bcalifornia\b": "ca", r"\btexas\b": "tx", r"\bflorida\b": "fl", r"\bohio\b": "oh",
    r"\bdelhi\b": "dl", r"\bwest bengal\b": "wb", r"\bkarnataka\b": "ka", r"\bmaharashtra\b": "mh",
    r"\bgujarat\b": "gj", r"\btamil nadu\b": "tn"
}

def standardize_address_expr(col_name: str) -> pl.Expr:
    expr = pl.col(col_name)
    for pat, rep in STREET_ABBRS.items():
        expr = expr.str.replace_all(pat, rep)
    expr = expr.str.replace_all(r"\b0+(\d+)\b", r"$1")
    return expr.str.replace_all(r"\s+", " ").str.strip_chars()

val_clean = val_clean.with_columns([standardize_address_expr("clean_address").alias("std_address")])
targets_clean = targets_clean.with_columns([standardize_address_expr("clean_address").alias("std_address")])

# Channel 1: Exact Clean Name
p1 = val_clean.select(["entity_id", "country", "clean_name"]).filter(pl.col("clean_name") != "").join(
    targets_clean.select(["entity_id", "country", "clean_name"]).filter(pl.col("clean_name") != "").rename({"entity_id": "target_id"}),
    on=["country", "clean_name"], how="inner"
).select(["entity_id", "target_id"]).unique()

# Channel 2: Token-Sorted Name
p2 = val_clean.select(["entity_id", "country", "sorted_name"]).filter(pl.col("sorted_name") != "").join(
    targets_clean.select(["entity_id", "country", "sorted_name"]).filter(pl.col("sorted_name") != "").rename({"entity_id": "target_id"}),
    on=["country", "sorted_name"], how="inner"
).select(["entity_id", "target_id"]).unique()

# Channel 3: Standardized Exact Address
p3 = val_clean.select(["entity_id", "country", "std_address"]).filter(pl.col("std_address").str.len_chars() >= 10).join(
    targets_clean.select(["entity_id", "country", "std_address"]).filter(pl.col("std_address").str.len_chars() >= 10).rename({"entity_id": "target_id"}),
    on=["country", "std_address"], how="inner"
).select(["entity_id", "target_id"]).unique()

# Channel 4: Street Prefix Match
val_addr_p3 = val_clean.with_columns([pl.col("std_address").str.split(" ").list.slice(0, 3).list.join(" ").alias("addr_p3")])
tgt_addr_p3 = targets_clean.with_columns([pl.col("std_address").str.split(" ").list.slice(0, 3).list.join(" ").alias("addr_p3")])
p4 = val_addr_p3.select(["entity_id", "country", "addr_p3"]).filter(
    (pl.col("addr_p3").str.contains(r"\d")) & (pl.col("addr_p3").str.len_chars() >= 8)
).join(
    tgt_addr_p3.select(["entity_id", "country", "addr_p3"]).filter(
        (pl.col("addr_p3").str.contains(r"\d")) & (pl.col("addr_p3").str.len_chars() >= 8)
    ).rename({"entity_id": "target_id"}),
    on=["country", "addr_p3"], how="inner"
).select(["entity_id", "target_id"]).unique()

# Channel 5: 2-Word Name Bigram
val_bigrams = val_clean.with_columns([
    pl.col("stripped_name").str.split(" ").list.filter(pl.element().str.len_chars() >= 3).alias("words")
]).with_columns([
    pl.when(pl.col("words").list.len() >= 2)
      .then(pl.col("words").list.slice(0, 2).list.sort().list.join(" "))
      .otherwise(pl.lit(""))
      .alias("name_bigram")
])
tgt_bigrams = targets_clean.with_columns([
    pl.col("stripped_name").str.split(" ").list.filter(pl.element().str.len_chars() >= 3).alias("words")
]).with_columns([
    pl.when(pl.col("words").list.len() >= 2)
      .then(pl.col("words").list.slice(0, 2).list.sort().list.join(" "))
      .otherwise(pl.lit(""))
      .alias("name_bigram")
])
bigram_counts = tgt_bigrams.filter(pl.col("name_bigram") != "").group_by(["country", "name_bigram"]).len()
rare_bigrams = bigram_counts.filter(pl.col("len") <= 500).select(["country", "name_bigram"])
val_rare_bi = val_bigrams.select(["entity_id", "country", "name_bigram"]).join(rare_bigrams, on=["country", "name_bigram"], how="inner")
tgt_rare_bi = tgt_bigrams.select(["entity_id", "country", "name_bigram"]).join(rare_bigrams, on=["country", "name_bigram"], how="inner")
p5 = val_rare_bi.join(
    tgt_rare_bi.rename({"entity_id": "target_id"}),
    on=["country", "name_bigram"], how="inner"
).select(["entity_id", "target_id"]).unique()

# Channel 6: Distinctive Brand Token
GENERIC_WORDS = {
    "center", "services", "service", "enterprises", "enterprise", "solutions",
    "group", "holdings", "holding", "associates", "consultancy", "consulting",
    "management", "industries", "industry", "international", "global", "national",
    "united", "american", "india", "delhi", "mumbai", "texas", "california",
    "medical", "health", "care", "clinic", "hospital", "pharma", "pharmacy",
    "dental", "dentistry", "realty", "properties", "estate", "construction",
    "builders", "technologies", "technology", "tech", "systems", "corp", "inc"
}
val_brand = val_clean.with_columns([
    pl.col("stripped_name").str.split(" ").list.get(0).alias("raw_brand")
]).with_columns([
    pl.when((pl.col("raw_brand").str.len_chars() >= 4) & (~pl.col("raw_brand").is_in(list(GENERIC_WORDS))))
      .then(pl.col("raw_brand"))
      .otherwise(pl.lit(""))
      .alias("brand_token")
])
tgt_brand = targets_clean.with_columns([
    pl.col("stripped_name").str.split(" ").list.get(0).alias("raw_brand")
]).with_columns([
    pl.when((pl.col("raw_brand").str.len_chars() >= 4) & (~pl.col("raw_brand").is_in(list(GENERIC_WORDS))))
      .then(pl.col("raw_brand"))
      .otherwise(pl.lit(""))
      .alias("brand_token")
])
brand_counts = tgt_brand.filter(pl.col("brand_token") != "").group_by(["country", "brand_token"]).len()
rare_brands = brand_counts.filter(pl.col("len") <= 300).select(["country", "brand_token"])
val_rare_br = val_brand.select(["entity_id", "country", "brand_token"]).join(rare_brands, on=["country", "brand_token"], how="inner")
tgt_rare_br = tgt_brand.select(["entity_id", "country", "brand_token"]).join(rare_brands, on=["country", "brand_token"], how="inner")
p6 = val_rare_br.join(
    tgt_rare_br.rename({"entity_id": "target_id"}),
    on=["country", "brand_token"], how="inner"
).select(["entity_id", "target_id"]).unique()

all_retrieved = pl.concat([p1, p2, p3, p4, p5, p6]).unique()
print(f"Total unique pairs retrieved: {len(all_retrieved):,}")

# Find missed pairs
missed = gt_df.join(all_retrieved, on=["entity_id", "target_id"], how="anti")
print(f"Total true links missed: {len(missed):,} ({len(missed)/len(gt_df)*100:.2f}%)")

# Sample 30 missed links and join raw text
sample_missed = missed.head(30)
sample_annotated = sample_missed.join(
    val_df.select(["entity_id", "business_name", "business_address", "country"]),
    on="entity_id", how="inner"
).join(
    targets_clean.select(["entity_id", "business_name", "business_address"]).rename({
        "entity_id": "target_id",
        "business_name": "tgt_name",
        "business_address": "tgt_address"
    }),
    on="target_id", how="inner"
)

with open("experiments/missed_links_investigation.txt", "w", encoding="utf-8") as f:
    f.write(f"SAMPLE OF 30 MISSED TRUE LINKS OUT OF {len(missed):,} TOTAL MISSED\n")
    f.write("=" * 80 + "\n\n")
    for r in sample_annotated.iter_rows(named=True):
        f.write(f"Country: {r['country']}\n")
        f.write(f"  S1:  Name: '{r['business_name']}'\n")
        f.write(f"       Addr: '{r['business_address']}'\n")
        f.write(f"  Tgt: Name: '{r['tgt_name']}'\n")
        f.write(f"       Addr: '{r['tgt_address']}'\n\n")

print("Saved 30 missed links to experiments/missed_links_investigation.txt")
