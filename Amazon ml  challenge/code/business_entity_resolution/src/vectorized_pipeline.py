"""
src/vectorized_pipeline.py
==========================
Ultra-high-throughput, zero-memory-leakage Polars vectorized representation
and candidate retrieval engine.

Leverages Polars multithreaded SIMD Rust string expressions:
  - clean_name: lowercased, whitespace collapsed, punctuation removed
  - stripped_name: legal suffixes removed
  - sorted_name: order-independent token-sorted representation
  - postal_code: country-aware PIN/ZIP regex extraction
  - clean_address: normalized address tokens

Performs multi-pass candidate generation via parallel Polars hash-joins.
"""

import os
import time
import polars as pl

# Compiled regex for legal entity suffixes
LEGAL_SUFFIX_REGEX = (
    r"(?i)\b(?:"
    r"private limited|pvt ltd|pvt limited|private ltd|limited|ltd|pvt|"
    r"llc|l\.l\.c\.|inc|incorporated|corporation|corp|llp|l\.l\.p\.|"
    r"co|company|gmbh|sarl|sa|sas|plc"
    r")\b"
)

# Postal code regexes
INDIA_PIN_REGEX = r"\b([1-9]\d{5})\b"
US_ZIP_REGEX = r"\b(\d{5})(?:-\d{4})?\b"
FRANCE_POSTAL_REGEX = r"\b((?:0[1-9]|[1-8]\d|9[0-8])\d{3})\b"


def normalize_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    """
    Applies high-speed vectorized string normalizations to a Polars DataFrame.
    Expects columns: entity_id, country, business_name, business_address.
    """
    # 1. Clean names and addresses
    df_clean = df.with_columns([
        pl.col("business_name").fill_null("")
          .str.to_lowercase()
          .str.replace_all("&", " and ")
          .str.replace_all(r"[^\w\s]", " ")
          .str.replace_all(r"\s+", " ")
          .str.strip_chars()
          .alias("clean_name"),
        pl.col("business_address").fill_null("")
          .str.to_lowercase()
          .str.replace_all(r"[^\w\s]", " ")
          .str.replace_all(r"\s+", " ")
          .str.strip_chars()
          .alias("clean_address"),
        pl.col("country").fill_null("").str.strip_chars().alias("country")
    ])
    
    # 2. Legal suffix stripping
    df_stripped = df_clean.with_columns([
        pl.col("clean_name")
          .str.replace_all(LEGAL_SUFFIX_REGEX, " ")
          .str.replace_all(r"\s+", " ")
          .str.strip_chars()
          .alias("stripped_name")
    ])
    
    # 3. Order-independent token sorting & postal extraction
    df_final = df_stripped.with_columns([
        pl.when(pl.col("stripped_name") != "")
          .then(pl.col("stripped_name"))
          .otherwise(pl.col("clean_name"))
          .str.split(" ")
          .list.sort()
          .list.join(" ")
          .alias("sorted_name"),
        # Country-aware postal code
        pl.when(pl.col("country").str.to_uppercase() == "INDIA")
          .then(pl.col("clean_address").str.extract(INDIA_PIN_REGEX, 1))
          .when(pl.col("country").str.to_uppercase() == "US")
          .then(pl.col("clean_address").str.extract(US_ZIP_REGEX, 1))
          .when(pl.col("country").str.to_uppercase() == "FRANCE")
          .then(pl.col("clean_address").str.extract(FRANCE_POSTAL_REGEX, 1))
          .otherwise(pl.col("clean_address").str.extract(r"\b(\d{5,6})\b", 1))
          .fill_null("")
          .alias("postal_code")
    ])
    
    return df_final


def run_polars_candidate_generation(
    queries_df: pl.DataFrame,
    targets_df: pl.DataFrame,
    top_k_per_query: int = 30,
    include_postal: bool = True
) -> pl.DataFrame:
    """
    Executes multi-pass candidate generation via parallel Polars hash-joins.
    Pass 1: Exact Clean Name match (within country)
    Pass 2: Suffix-Stripped Token-Sorted Name match (within country)
    Pass 3 (Optional): Postal code + First Name Token match
    Returns a DataFrame of candidate pairs [entity_id, target_id].
    """
    t0 = time.time()
    
    # Pass 1: Exact Clean Name Join
    p1 = queries_df.select(["entity_id", "country", "clean_name"]).filter(pl.col("clean_name") != "").join(
        targets_df.select(["entity_id", "country", "clean_name"]).filter(pl.col("clean_name") != "").rename({"entity_id": "target_id"}),
        on=["country", "clean_name"],
        how="inner"
    ).select(["entity_id", "target_id"]).with_columns(pl.lit("exact_clean").alias("provenance"))
    
    # Pass 2: Suffix-Stripped Token-Sorted Join
    p2 = queries_df.select(["entity_id", "country", "sorted_name"]).filter(pl.col("sorted_name") != "").join(
        targets_df.select(["entity_id", "country", "sorted_name"]).filter(pl.col("sorted_name") != "").rename({"entity_id": "target_id"}),
        on=["country", "sorted_name"],
        how="inner"
    ).select(["entity_id", "target_id"]).with_columns(pl.lit("token_sorted").alias("provenance"))
    
    passes = [p1, p2]
    
    # Pass 3: Postal Code + 1st Word Join (Disambiguates branches & hard negatives)
    if include_postal:
        # Extract 1st significant word of name (len >= 3)
        q_postal = queries_df.select(["entity_id", "country", "postal_code", "clean_name"]).filter(
            (pl.col("postal_code") != "") & (pl.col("clean_name") != "")
        ).with_columns([
            pl.col("clean_name").str.split(" ").list.get(0).alias("first_word")
        ]).filter(pl.col("first_word").str.len_chars() >= 3)
        
        t_postal = targets_df.select(["entity_id", "country", "postal_code", "clean_name"]).filter(
            (pl.col("postal_code") != "") & (pl.col("clean_name") != "")
        ).with_columns([
            pl.col("clean_name").str.split(" ").list.get(0).alias("first_word")
        ]).filter(pl.col("first_word").str.len_chars() >= 3).rename({"entity_id": "target_id"})
        
        p3 = q_postal.join(
            t_postal,
            on=["country", "postal_code", "first_word"],
            how="inner"
        ).select(["entity_id", "target_id"]).with_columns(pl.lit("postal_first_word").alias("provenance"))
        
        passes.append(p3)
        
    # Concatenate all passes
    all_pairs = pl.concat(passes).unique(subset=["entity_id", "target_id"])
    
    # Apply Top-K budget per query entity to prevent candidate explosion
    all_pairs = all_pairs.group_by("entity_id").head(top_k_per_query)
    
    return all_pairs
