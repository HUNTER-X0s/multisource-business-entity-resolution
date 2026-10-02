"""
experiments/phase2/feature_engineering.py
=========================================
High-Performance Pairwise Feature Extraction Engine for Phase 2.
Computes multi-view lexical, character, address, numeric, postal,
provenance, and negative contradiction features over candidate pairs.
"""

import re
import time
from typing import List, Tuple, Dict, Any, Optional
import polars as pl
import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein


# Precompiled regex for numeric tokens and house numbers
NUMERIC_RE = re.compile(r"\b\d+\b")


def extract_pairwise_features(
    cand_df: pl.DataFrame,
    queries_df: pl.DataFrame,
    targets_df: pl.DataFrame,
    verbose: bool = True
) -> Tuple[pl.DataFrame, List[str]]:
    """
    Computes a comprehensive pairwise feature set for each (entity_id, target_id) pair.

    Parameters:
        cand_df: DataFrame with ['entity_id', 'target_id', 'max_prio', 'n_channels']
        queries_df: DataFrame with query attributes
        targets_df: DataFrame with target attributes

    Returns:
        feature_df: Polars DataFrame with all original columns + extracted features
        feature_names: List of column names corresponding to numeric features
    """
    t0 = time.time()
    n_pairs = len(cand_df)
    if verbose:
        print(f"[FeatureEng] Extracting features for {n_pairs:,} candidate pairs...", flush=True)

    addr_col_q = "clean_address" if "clean_address" in queries_df.columns else "std_address"
    addr_col_t = "clean_address" if "clean_address" in targets_df.columns else "std_address"

    # 1. Join Query Attributes
    q_cols = queries_df.select([
        pl.col("entity_id"),
        pl.col("clean_name").alias("q_clean_name"),
        pl.col("stripped_name").alias("q_stripped_name"),
        pl.col("sorted_name").alias("q_sorted_name"),
        pl.col(addr_col_q).alias("q_clean_addr"),
        pl.col("postal_code").alias("q_postal"),
    ])

    # 2. Join Target Attributes
    t_cols = targets_df.select([
        pl.col("entity_id").alias("target_id"),
        pl.col("clean_name").alias("t_clean_name"),
        pl.col("stripped_name").alias("t_stripped_name"),
        pl.col("sorted_name").alias("t_sorted_name"),
        pl.col(addr_col_t).alias("t_clean_addr"),
        pl.col("postal_code").alias("t_postal"),
    ])

    merged = cand_df.join(q_cols, on="entity_id", how="inner").join(t_cols, on="target_id", how="inner")

    # Extract lists for SIMD / vectorized processing
    q_names = merged["q_clean_name"].fill_null("").to_list()
    t_names = merged["t_clean_name"].fill_null("").to_list()

    q_strip = merged["q_stripped_name"].fill_null("").to_list()
    t_strip = merged["t_stripped_name"].fill_null("").to_list()

    q_sort = merged["q_sorted_name"].fill_null("").to_list()
    t_sort = merged["t_sorted_name"].fill_null("").to_list()

    q_addrs = merged["q_clean_addr"].fill_null("").to_list()
    t_addrs = merged["t_clean_addr"].fill_null("").to_list()

    q_posts = merged["q_postal"].fill_null("").to_list()
    t_posts = merged["t_postal"].fill_null("").to_list()

    targets = merged["target_id"].to_list()

    # Feature 1: Exact Name Matches
    exact_clean = [1.0 if q == t and q != "" else 0.0 for q, t in zip(q_names, t_names)]
    exact_stripped = [1.0 if q == t and q != "" else 0.0 for q, t in zip(q_strip, t_strip)]
    exact_sorted = [1.0 if q == t and q != "" else 0.0 for q, t in zip(q_sort, t_sort)]

    # Feature 2: Name String Similarities (RapidFuzz SIMD)
    name_jw = [
        1.0 if q == t else float(JaroWinkler.similarity(q, t))
        for q, t in zip(q_names, t_names)
    ]
    name_token_sort = [
        1.0 if q == t else (fuzz.token_sort_ratio(q, t) / 100.0)
        for q, t in zip(q_names, t_names)
    ]
    name_token_set = [
        1.0 if q == t else (fuzz.token_set_ratio(q, t) / 100.0)
        for q, t in zip(q_names, t_names)
    ]
    name_len_diff = [
        abs(len(q) - len(t))
        for q, t in zip(q_names, t_names)
    ]
    name_len_ratio = [
        min(len(q), len(t)) / max(len(q), len(t)) if max(len(q), len(t)) > 0 else 0.0
        for q, t in zip(q_names, t_names)
    ]

    # Feature 3: Address Similarities & Contradiction Evidence
    addr_is_null_t = [1.0 if t == "" else 0.0 for t in t_addrs]
    exact_addr = [1.0 if q == t and q != "" else 0.0 for q, t in zip(q_addrs, t_addrs)]
    
    addr_jw = []
    addr_jaccard = []
    addr_overlap = []
    
    for q, t in zip(q_addrs, t_addrs):
        if q == "" or t == "":
            addr_jw.append(0.0)
            addr_jaccard.append(0.0)
            addr_overlap.append(0.0)
        elif q == t:
            addr_jw.append(1.0)
            addr_jaccard.append(1.0)
            addr_overlap.append(1.0)
        else:
            addr_jw.append(float(JaroWinkler.similarity(q, t)))
            sq = set(q.split())
            st = set(t.split())
            intersect = len(sq & st)
            union = len(sq | st)
            min_len = min(len(sq), len(st))
            addr_jaccard.append(intersect / union if union > 0 else 0.0)
            addr_overlap.append(intersect / min_len if min_len > 0 else 0.0)

    # Feature 4: Numeric Tokens & House Number Disagreement (Contradiction)
    num_jaccard = []
    house_num_match = []
    contradiction_house_no = []

    for q, t in zip(q_addrs, t_addrs):
        q_nums = NUMERIC_RE.findall(q)
        t_nums = NUMERIC_RE.findall(t)
        
        if not q_nums or not t_nums:
            num_jaccard.append(0.0)
            house_num_match.append(0.0)
            contradiction_house_no.append(0.0)
        else:
            sq_n = set(q_nums)
            st_n = set(t_nums)
            num_jaccard.append(len(sq_n & st_n) / len(sq_n | st_n))
            # First numeric token is usually building / street number
            if q_nums[0] == t_nums[0]:
                house_num_match.append(1.0)
                contradiction_house_no.append(0.0)
            else:
                house_num_match.append(0.0)
                contradiction_house_no.append(1.0)

    # Feature 5: Postal Code Verification & Contradiction
    postal_match = []
    postal_both_present = []
    contradiction_postal = []

    for q, t in zip(q_posts, t_posts):
        if q != "" and t != "":
            postal_both_present.append(1.0)
            if q == t:
                postal_match.append(1.0)
                contradiction_postal.append(0.0)
            else:
                postal_match.append(0.0)
                contradiction_postal.append(1.0)
        else:
            postal_both_present.append(0.0)
            postal_match.append(0.0)
            contradiction_postal.append(0.0)

    # Feature 6: Source & Channel Provenance
    is_source3 = [1.0 if t.startswith("S3-") else 0.0 for t in targets]

    # Feature 7: Cross-Feature Interactions
    name_x_addr = [jw * aj for jw, aj in zip(name_jw, addr_jaccard)]
    name_x_postal = [jw * pm for jw, pm in zip(name_jw, postal_match)]

    # Assemble into DataFrame
    feature_dict = {
        "exact_clean_name": pl.Series(exact_clean, dtype=pl.Float32),
        "exact_stripped_name": pl.Series(exact_stripped, dtype=pl.Float32),
        "exact_sorted_name": pl.Series(exact_sorted, dtype=pl.Float32),
        "name_jaro_winkler": pl.Series(name_jw, dtype=pl.Float32),
        "name_token_sort": pl.Series(name_token_sort, dtype=pl.Float32),
        "name_token_set": pl.Series(name_token_set, dtype=pl.Float32),
        "name_len_diff": pl.Series(name_len_diff, dtype=pl.Float32),
        "name_len_ratio": pl.Series(name_len_ratio, dtype=pl.Float32),
        "exact_clean_address": pl.Series(exact_addr, dtype=pl.Float32),
        "addr_is_null_target": pl.Series(addr_is_null_t, dtype=pl.Float32),
        "addr_jaro_winkler": pl.Series(addr_jw, dtype=pl.Float32),
        "addr_token_jaccard": pl.Series(addr_jaccard, dtype=pl.Float32),
        "addr_token_overlap": pl.Series(addr_overlap, dtype=pl.Float32),
        "numeric_token_jaccard": pl.Series(num_jaccard, dtype=pl.Float32),
        "house_number_match": pl.Series(house_num_match, dtype=pl.Float32),
        "contradiction_house_no": pl.Series(contradiction_house_no, dtype=pl.Float32),
        "postal_code_match": pl.Series(postal_match, dtype=pl.Float32),
        "postal_both_present": pl.Series(postal_both_present, dtype=pl.Float32),
        "contradiction_postal": pl.Series(contradiction_postal, dtype=pl.Float32),
        "is_source3": pl.Series(is_source3, dtype=pl.Float32),
        "channel_max_priority": merged["max_prio"].cast(pl.Float32),
        "channel_count": merged["n_channels"].cast(pl.Float32),
        "name_x_addr": pl.Series(name_x_addr, dtype=pl.Float32),
        "name_x_postal": pl.Series(name_x_postal, dtype=pl.Float32),
    }

    feature_names = list(feature_dict.keys())
    
    # Keep key identifier columns and append features
    result_df = merged.select([
        "entity_id", "target_id", "max_prio", "n_channels"
    ] + ([col for col in ["label"] if col in merged.columns])).with_columns([
        v.alias(k) for k, v in feature_dict.items()
    ])

    if verbose:
        print(f"[FeatureEng] Generated {len(feature_names)} features in {time.time()-t0:.2f}s", flush=True)

    return result_df, feature_names
