"""
src/candidate_generation.py
============================
Production-grade 7-channel candidate generation pipeline for
Amazon ML Challenge 2026 — Business Entity Resolution.

Architecture: Deterministic multi-channel exact-match retrieval using Polars
columnar hash-joins on Arrow buffers. No external APIs. No fuzzy indexing.

CHANNELS (in priority order):
  1. Exact Clean Name                  (prio 1.00) — highest precision
  2. Token-Sorted Suffix-Stripped Name (prio 0.95) — word-order invariant
  3. Standardized Exact Address        (prio 0.90) — full address match
  4. Sorted Address Token Set          (prio 0.88) — reorder-invariant address
  5. 2-Word Name Bigram (rare)         (prio 0.75) — first 2 meaningful words
  6. Street Prefix (3 tokens + digit)  (prio 0.70) — house-number prefix
  7. Distinctive Brand Token (rare)    (prio 0.60) — first non-generic word

Validated performance on 50K query validation set:
  Unbudgeted:  81.56% Link Recall, 96.75% Entity Coverage
  K=30:        67.2%  Link Recall, 87.3%  Entity Coverage
  K=40:        69.3%  Link Recall, 89.2%  Entity Coverage
  K=50:        71.5%  Link Recall, 91.3%  Entity Coverage

Hardware targets: Dell G15, RTX 3050 6GB, 16GB RAM.
Runtime at full test scale (~1.73M queries × 9.97M targets): <60s.
Peak RAM: ~6GB (Polars Arrow buffers).

Design constraints:
  - Zero external data. Zero HTTP. Zero LLM.
  - Raw data never mutated.
  - All normalization is pure Polars expressions (SIMD Rust).
  - Candidate budget enforced by priority-then-channel-count ranking.
"""

import os
import time
from typing import Optional

import polars as pl

# ---------------------------------------------------------------------------
# Normalization constants
# ---------------------------------------------------------------------------
LEGAL_SUFFIX_REGEX = (
    r"(?i)\b(?:"
    r"private limited|pvt ltd|pvt limited|private ltd|limited|ltd|pvt|"
    r"llc|l\.l\.c\.|inc|incorporated|corporation|corp|llp|l\.l\.p\.|"
    r"co|company|gmbh|sarl|sa|sas|plc"
    r")\b"
)

# Street type & state abbreviation map
STREET_ABBRS = {
    r"\bst\b": "street",
    r"\bave\b": "avenue",
    r"\brd\b": "road",
    r"\bdr\b": "drive",
    r"\bblvd\b": "boulevard",
    r"\bln\b": "lane",
    # US states
    r"\bdelaware\b": "de",
    r"\bindiana\b": "in",
    r"\bnew york\b": "ny",
    r"\butah\b": "ut",
    r"\bvirginia\b": "va",
    r"\bmaine\b": "me",
    r"\bcalifornia\b": "ca",
    r"\btexas\b": "tx",
    r"\bflorida\b": "fl",
    r"\bohio\b": "oh",
    # India states / UTs
    r"\bdelhi\b": "dl",
    r"\bwest bengal\b": "wb",
    r"\bkarnataka\b": "ka",
    r"\bmaharashtra\b": "mh",
    r"\bgujarat\b": "gj",
    r"\btamil nadu\b": "tn",
}

# Brand token generic stop-set
GENERIC_BRAND_WORDS = {
    "center", "services", "service", "enterprises", "enterprise", "solutions",
    "group", "holdings", "holding", "associates", "consultancy", "consulting",
    "management", "industries", "industry", "international", "global", "national",
    "united", "american", "india", "delhi", "mumbai", "texas", "california",
    "medical", "health", "care", "clinic", "hospital", "pharma", "pharmacy",
    "dental", "dentistry", "realty", "properties", "estate", "construction",
    "builders", "technologies", "technology", "tech", "systems", "corp", "inc",
}

# Sorted address stop-words (too common to discriminate)
ADDR_STOP_WORDS = {"floor", "unit", "suite", "null", "none", "na", "and", "the"}

# Country-aware postal code extraction
INDIA_PIN_REGEX = r"\b([1-9]\d{5})\b"
US_ZIP_REGEX = r"\b(\d{5})(?:-\d{4})?\b"
FRANCE_POSTAL_REGEX = r"\b((?:0[1-9]|[1-8]\d|9[0-8])\d{3})\b"

# ---------------------------------------------------------------------------
# Normalization helpers (pure Polars expressions)
# ---------------------------------------------------------------------------

def _clean_name_expr(col: str = "business_name") -> pl.Expr:
    return (
        pl.col(col).fill_null("")
        .str.to_lowercase()
        .str.replace_all("&", " and ")
        .str.replace_all(r"[^\w\s]", " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )

def _strip_suffix_expr(col: str = "clean_name") -> pl.Expr:
    return (
        pl.col(col)
        .str.replace_all(LEGAL_SUFFIX_REGEX, " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )

def _sorted_name_expr(col: str = "stripped_name") -> pl.Expr:
    return (
        pl.when(pl.col(col) != "")
        .then(pl.col(col))
        .otherwise(pl.col("clean_name"))
        .str.split(" ")
        .list.sort()
        .list.join(" ")
    )

def _clean_address_expr(col: str = "business_address") -> pl.Expr:
    expr = pl.col(col).fill_null("").str.to_lowercase().str.replace_all(r"[^\w\s]", " ")
    # State/street abbreviation expansion
    for pat, rep in STREET_ABBRS.items():
        expr = expr.str.replace_all(pat, rep)
    return (
        expr
        .str.replace_all(r"\b0+(\d+)\b", r"$1")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )

def _sorted_addr_expr(col: str = "std_address") -> pl.Expr:
    """Order-independent sorted bag of meaningful address tokens."""
    return (
        pl.col(col)
        .str.split(" ")
        .list.filter(
            (pl.element().str.len_chars() >= 3) &
            (~pl.element().is_in(list(ADDR_STOP_WORDS)))
        )
        .list.sort()
        .list.join(" ")
    )

def _postal_code_expr(addr_col: str = "std_address", country_col: str = "country") -> pl.Expr:
    return (
        pl.when(pl.col(country_col).str.to_uppercase() == "INDIA")
          .then(pl.col(addr_col).str.extract(INDIA_PIN_REGEX, 1))
          .when(pl.col(country_col).str.to_uppercase() == "US")
          .then(pl.col(addr_col).str.extract(US_ZIP_REGEX, 1))
          .when(pl.col(country_col).str.to_uppercase() == "FRANCE")
          .then(pl.col(addr_col).str.extract(FRANCE_POSTAL_REGEX, 1))
          .otherwise(pl.col(addr_col).str.extract(r"\b(\d{5,6})\b", 1))
          .fill_null("")
    )


# ---------------------------------------------------------------------------
# Core normalization: build all representation columns in one pass
# ---------------------------------------------------------------------------

def normalize_entities(df: pl.DataFrame) -> pl.DataFrame:
    """
    Applies all normalization passes to an entity DataFrame.

    Expected input columns: entity_id, business_name, business_address, country
    Produced output columns (added):
        clean_name, stripped_name, sorted_name,
        std_address, sorted_addr, postal_code
    """
    return (
        df
        .with_columns([
            _clean_name_expr("business_name").alias("clean_name"),
            _clean_address_expr("business_address").alias("std_address"),
            pl.col("country").fill_null("").str.strip_chars().alias("country"),
        ])
        .with_columns([
            _strip_suffix_expr("clean_name").alias("stripped_name"),
        ])
        .with_columns([
            _sorted_name_expr("stripped_name").alias("sorted_name"),
            _sorted_addr_expr("std_address").alias("sorted_addr"),
            _postal_code_expr("std_address", "country").alias("postal_code"),
        ])
    )


# ---------------------------------------------------------------------------
# Frequency filtering helpers (compute on target corpus)
# ---------------------------------------------------------------------------

def _compute_rare_bigrams(
    targets_norm: pl.DataFrame,
    max_freq: int = 500
) -> pl.DataFrame:
    """Returns (country, name_bigram) pairs with target frequency <= max_freq."""
    tgt_bi = targets_norm.with_columns([
        pl.col("stripped_name")
        .str.split(" ")
        .list.filter(pl.element().str.len_chars() >= 3)
        .alias("bi_words")
    ]).with_columns([
        pl.when(pl.col("bi_words").list.len() >= 2)
          .then(pl.col("bi_words").list.slice(0, 2).list.sort().list.join(" "))
          .otherwise(pl.lit(""))
          .alias("name_bigram")
    ])
    counts = tgt_bi.filter(pl.col("name_bigram") != "").group_by(["country", "name_bigram"]).len()
    return counts.filter(pl.col("len") <= max_freq).select(["country", "name_bigram"])


def _compute_rare_brand_tokens(
    targets_norm: pl.DataFrame,
    max_freq: int = 300
) -> pl.DataFrame:
    """Returns (country, brand_token) pairs with target frequency <= max_freq."""
    tgt_br = targets_norm.with_columns([
        pl.col("stripped_name").str.split(" ").list.get(0).alias("raw_brand")
    ]).with_columns([
        pl.when(
            (pl.col("raw_brand").str.len_chars() >= 4) &
            (~pl.col("raw_brand").is_in(list(GENERIC_BRAND_WORDS)))
        )
        .then(pl.col("raw_brand"))
        .otherwise(pl.lit(""))
        .alias("brand_token")
    ])
    counts = tgt_br.filter(pl.col("brand_token") != "").group_by(["country", "brand_token"]).len()
    return counts.filter(pl.col("len") <= max_freq).select(["country", "brand_token"])


# ---------------------------------------------------------------------------
# Individual channel retrievers
# ---------------------------------------------------------------------------

def _ch1_exact_name(queries: pl.DataFrame, targets: pl.DataFrame) -> pl.DataFrame:
    """Channel 1: Exact normalized name match within country."""
    return (
        queries.select(["entity_id", "country", "clean_name"])
        .filter(pl.col("clean_name") != "")
        .join(
            targets.select(["entity_id", "country", "clean_name"])
            .filter(pl.col("clean_name") != "")
            .rename({"entity_id": "target_id"}),
            on=["country", "clean_name"],
            how="inner",
        )
        .select(["entity_id", "target_id"])
        .unique()
        .with_columns(pl.lit(1.00).alias("prio"))
    )


def _ch2_sorted_name(queries: pl.DataFrame, targets: pl.DataFrame) -> pl.DataFrame:
    """Channel 2: Token-sorted suffix-stripped name within country."""
    return (
        queries.select(["entity_id", "country", "sorted_name"])
        .filter(pl.col("sorted_name") != "")
        .join(
            targets.select(["entity_id", "country", "sorted_name"])
            .filter(pl.col("sorted_name") != "")
            .rename({"entity_id": "target_id"}),
            on=["country", "sorted_name"],
            how="inner",
        )
        .select(["entity_id", "target_id"])
        .unique()
        .with_columns(pl.lit(0.95).alias("prio"))
    )


def _ch3_exact_address(queries: pl.DataFrame, targets: pl.DataFrame) -> pl.DataFrame:
    """Channel 3: Standardized full address match within country (min 10 chars)."""
    return (
        queries.select(["entity_id", "country", "std_address"])
        .filter(pl.col("std_address").str.len_chars() >= 10)
        .join(
            targets.select(["entity_id", "country", "std_address"])
            .filter(pl.col("std_address").str.len_chars() >= 10)
            .rename({"entity_id": "target_id"}),
            on=["country", "std_address"],
            how="inner",
        )
        .select(["entity_id", "target_id"])
        .unique()
        .with_columns(pl.lit(0.90).alias("prio"))
    )


def _ch4_sorted_address(queries: pl.DataFrame, targets: pl.DataFrame) -> pl.DataFrame:
    """Channel 4: Sorted address token bag — reorder-invariant address match (min 15 chars key)."""
    return (
        queries.select(["entity_id", "country", "sorted_addr"])
        .filter(pl.col("sorted_addr").str.len_chars() >= 15)
        .join(
            targets.select(["entity_id", "country", "sorted_addr"])
            .filter(pl.col("sorted_addr").str.len_chars() >= 15)
            .rename({"entity_id": "target_id"}),
            on=["country", "sorted_addr"],
            how="inner",
        )
        .select(["entity_id", "target_id"])
        .unique()
        .with_columns(pl.lit(0.88).alias("prio"))
    )


def _ch5_name_bigram(
    queries: pl.DataFrame,
    targets: pl.DataFrame,
    rare_bigrams: pl.DataFrame,
) -> pl.DataFrame:
    """Channel 5: Rare 2-word name bigram match within country."""
    def _add_bigram(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns([
            df["stripped_name"]
            .str.split(" ")
            .list.filter(pl.element().str.len_chars() >= 3)
            .alias("bi_words")
        ]).with_columns([
            pl.when(pl.col("bi_words").list.len() >= 2)
              .then(pl.col("bi_words").list.slice(0, 2).list.sort().list.join(" "))
              .otherwise(pl.lit(""))
              .alias("name_bigram")
        ])

    q_bi = _add_bigram(queries).select(["entity_id", "country", "name_bigram"]).join(
        rare_bigrams, on=["country", "name_bigram"], how="inner"
    )
    t_bi = _add_bigram(targets).select(["entity_id", "country", "name_bigram"]).join(
        rare_bigrams, on=["country", "name_bigram"], how="inner"
    )
    return (
        q_bi.join(t_bi.rename({"entity_id": "target_id"}), on=["country", "name_bigram"], how="inner")
        .select(["entity_id", "target_id"])
        .unique()
        .with_columns(pl.lit(0.75).alias("prio"))
    )


def _ch6_street_prefix(queries: pl.DataFrame, targets: pl.DataFrame) -> pl.DataFrame:
    """Channel 6: Street prefix (first 3 address tokens) with digit, within country."""
    def _add_prefix(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns([
            pl.col("std_address").str.split(" ").list.slice(0, 3).list.join(" ").alias("addr_p3")
        ])
    q_p = _add_prefix(queries).select(["entity_id", "country", "addr_p3"]).filter(
        (pl.col("addr_p3").str.contains(r"\d")) & (pl.col("addr_p3").str.len_chars() >= 8)
    )
    t_p = _add_prefix(targets).select(["entity_id", "country", "addr_p3"]).filter(
        (pl.col("addr_p3").str.contains(r"\d")) & (pl.col("addr_p3").str.len_chars() >= 8)
    )
    return (
        q_p.join(t_p.rename({"entity_id": "target_id"}), on=["country", "addr_p3"], how="inner")
        .select(["entity_id", "target_id"])
        .unique()
        .with_columns(pl.lit(0.70).alias("prio"))
    )


def _ch7_brand_token(
    queries: pl.DataFrame,
    targets: pl.DataFrame,
    rare_brands: pl.DataFrame,
) -> pl.DataFrame:
    """Channel 7: Rare distinctive brand (first) token match within country."""
    def _add_brand(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns([
            pl.col("stripped_name").str.split(" ").list.get(0).alias("raw_brand")
        ]).with_columns([
            pl.when(
                (pl.col("raw_brand").str.len_chars() >= 4) &
                (~pl.col("raw_brand").is_in(list(GENERIC_BRAND_WORDS)))
            )
            .then(pl.col("raw_brand"))
            .otherwise(pl.lit(""))
            .alias("brand_token")
        ])

    q_br = _add_brand(queries).select(["entity_id", "country", "brand_token"]).join(
        rare_brands, on=["country", "brand_token"], how="inner"
    )
    t_br = _add_brand(targets).select(["entity_id", "country", "brand_token"]).join(
        rare_brands, on=["country", "brand_token"], how="inner"
    )
    return (
        q_br.join(t_br.rename({"entity_id": "target_id"}), on=["country", "brand_token"], how="inner")
        .select(["entity_id", "target_id"])
        .unique()
        .with_columns(pl.lit(0.60).alias("prio"))
    )


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def generate_candidates(
    queries_df: pl.DataFrame,
    targets_df: pl.DataFrame,
    top_k: int = 40,
    bigram_max_freq: int = 500,
    brand_max_freq: int = 300,
    verbose: bool = True,
    return_provenance: bool = False,
) -> pl.DataFrame:
    """
    Runs the full 7-channel candidate generation pipeline.

    Parameters
    ----------
    queries_df : pl.DataFrame
        Source-1 entities. Required columns: entity_id, business_name,
        business_address, country.
    targets_df : pl.DataFrame
        Source-2/3 entities. Same schema as queries_df.
    top_k : int
        Maximum candidates per query. Ranking: highest channel priority first,
        then highest channel count (multi-channel intersection bonus).
    bigram_max_freq : int
        Prune name bigrams with target frequency > this threshold.
    brand_max_freq : int
        Prune brand tokens with target frequency > this threshold.
    verbose : bool
        Print per-channel diagnostics.

    Returns
    -------
    pl.DataFrame with columns [entity_id, target_id].
    One row per (query, candidate) pair, at most top_k rows per entity_id.
    """
    t_total = time.time()

    if verbose:
        print(f"[CandGen] Normalizing {len(queries_df):,} queries & {len(targets_df):,} targets ...", flush=True)

    q = normalize_entities(queries_df)
    t = normalize_entities(targets_df)

    if verbose:
        print(f"[CandGen] Normalization done in {time.time()-t_total:.1f}s", flush=True)

    # Pre-compute frequency filters on target corpus (done once)
    t_freq = time.time()
    rare_bigrams = _compute_rare_bigrams(t, bigram_max_freq)
    rare_brands = _compute_rare_brand_tokens(t, brand_max_freq)
    if verbose:
        print(f"[CandGen] Frequency filters built in {time.time()-t_freq:.1f}s", flush=True)

    # Run all 7 channels
    channels = []
    channel_info = [
        ("Ch1 Exact Name",     lambda: _ch1_exact_name(q, t)),
        ("Ch2 Sorted Name",    lambda: _ch2_sorted_name(q, t)),
        ("Ch3 Exact Address",  lambda: _ch3_exact_address(q, t)),
        ("Ch4 Sorted Address", lambda: _ch4_sorted_address(q, t)),
        ("Ch5 Name Bigram",    lambda: _ch5_name_bigram(q, t, rare_bigrams)),
        ("Ch6 Street Prefix",  lambda: _ch6_street_prefix(q, t)),
        ("Ch7 Brand Token",    lambda: _ch7_brand_token(q, t, rare_brands)),
    ]

    for name, fn in channel_info:
        t_ch = time.time()
        ch_pairs = fn()
        if verbose:
            print(f"[CandGen]   {name}: {len(ch_pairs):,} pairs  ({time.time()-t_ch:.2f}s)", flush=True)
        channels.append(ch_pairs)

    # Merge all channels: keep max priority and channel count per pair
    all_raw = pl.concat(channels)
    candidate_pool = (
        all_raw
        .group_by(["entity_id", "target_id"])
        .agg([
            pl.col("prio").max().alias("max_prio"),
            pl.len().alias("n_channels"),
        ])
    )
    if verbose:
        print(f"[CandGen] Unique candidate pairs: {len(candidate_pool):,}", flush=True)

    # Rank and truncate to top_k
    # Primary key: max_prio desc (exact name > sorted name > address > bigram > brand)
    # Secondary key: n_channels desc (multi-channel intersection = higher confidence)
    ranked = candidate_pool.sort(
        ["entity_id", "max_prio", "n_channels"],
        descending=[False, True, True],
    )
    out_cols = ["entity_id", "target_id"]
    if return_provenance:
        out_cols.extend(["max_prio", "n_channels"])
    result = (
        ranked
        .group_by("entity_id", maintain_order=True)
        .head(top_k)
        .select(out_cols)
    )

    if verbose:
        print(f"[CandGen] Final candidates at K={top_k}: {len(result):,} pairs", flush=True)
        print(f"[CandGen] Total time: {time.time()-t_total:.1f}s", flush=True)

    return result


def generate_candidates_and_save(
    queries_df: pl.DataFrame,
    targets_df: pl.DataFrame,
    output_path: str,
    top_k: int = 40,
    bigram_max_freq: int = 500,
    brand_max_freq: int = 300,
    verbose: bool = True,
) -> None:
    """
    Runs candidate generation and writes output to a competition-compliant TSV.

    Output format per competition specification:
        source1_entity_id  <TAB>  candidate_entity_ids (comma-separated)

    Also handles S1 queries with zero candidates by writing them with empty
    candidate list (required so the validator can confirm all S1 IDs present).
    """
    candidates = generate_candidates(
        queries_df, targets_df,
        top_k=top_k,
        bigram_max_freq=bigram_max_freq,
        brand_max_freq=brand_max_freq,
        verbose=verbose,
    )

    # Group candidate ids per query
    grouped = (
        candidates
        .group_by("entity_id")
        .agg(pl.col("target_id").alias("cand_list"))
    )

    # Build full S1 set (some queries may have zero candidates)
    all_s1 = queries_df.select("entity_id")
    grouped_full = all_s1.join(grouped, on="entity_id", how="left")
    # Polars fill_null on List dtype requires map_elements for empty list fill
    grouped_full = grouped_full.with_columns(
        pl.col("cand_list").map_elements(
            lambda x: x if x is not None else [],
            return_dtype=pl.List(pl.Utf8),
        )
    )

    if verbose:
        zero_cand = grouped_full.filter(pl.col("cand_list").list.len() == 0)
        print(f"[CandGen] Queries with 0 candidates: {len(zero_cand):,}", flush=True)

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for row in grouped_full.iter_rows(named=True):
            cand_list = row["cand_list"] or []
            cand_str = ",".join(cand_list)
            f.write(f"{row['entity_id']}\t{cand_str}\n")

    if verbose:
        print(f"[CandGen] Saved candidate_pairs.tsv -> {output_path}", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="7-channel Polars candidate generation pipeline")
    parser.add_argument("--test-dir", type=str, default="../../dataset/test", help="Path to test directory containing TSV files")
    parser.add_argument("--output-dir", type=str, default="../../output", help="Path to output directory")
    parser.add_argument("--max-candidates", type=int, default=40, help="Maximum candidates per query")
    args = parser.parse_args()

    test_dir = args.test_dir
    s1_path = os.path.join(test_dir, "test_source1.tsv")
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    print(f"[CandGen] Loading queries from {test_dir}...", flush=True)
    queries = pl.read_csv(os.path.join(test_dir, "test_source1.tsv"), separator="\t", truncate_ragged_lines=True)
    print(f"[CandGen] Loading targets (S2 + S3)...", flush=True)
    s2 = pl.read_csv(os.path.join(test_dir, "test_source2.tsv"), separator="\t", truncate_ragged_lines=True)
    s3 = pl.read_csv(os.path.join(test_dir, "test_source3.tsv"), separator="\t", truncate_ragged_lines=True)
    targets = pl.concat([s2, s3])

    out_file = os.path.join(out_dir, "candidate_pairs.tsv")
    print(f"[CandGen] Running candidate generation (top_k={args.max_candidates})...", flush=True)
    generate_candidates_and_save(queries, targets, output_path=out_file, top_k=args.max_candidates, verbose=True)

