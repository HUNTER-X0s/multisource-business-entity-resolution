"""
Feature Engineering v2 — EXP-06
Extends v1 (24 features) with additional signals targeting:
  - India country slice gap (F0.5 0.691 vs US 0.837)
  - match_bin_1 weakness (queries with exactly 1 true match)
  - Low recall on singleton detection

New features (16 additional):
  name_char2_jaccard        : character bigram Jaccard
  name_char3_jaccard        : character trigram Jaccard
  name_prefix3_match        : first 3 chars match exactly
  name_prefix5_match        : first 5 chars match exactly
  name_abbrev_ratio         : ratio of single-char tokens (abbreviations)
  name_word_count_diff      : abs difference in word count
  name_word_count_ratio     : ratio of word counts
  addr_char3_jaccard        : address char trigram Jaccard
  addr_word_count_diff      : abs difference in address word count
  addr_numeric_density      : fraction of numeric chars in address
  city_token_overlap        : token overlap restricted to city/state tokens
  pincode_prefix_match      : first 3 chars of postal code match (subzone)
  candidate_rank            : position of this candidate in ranked list
  n_candidates              : total candidate pool size for this query
  name_x_addr_x_postal      : three-way interaction product
  india_flag                : binary flag if entity is from India
"""

import re
import numpy as np
import pandas as pd
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# CHARACTER N-GRAM UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _char_ngrams(s: str, n: int) -> set:
    """Return set of character n-grams for string s."""
    if not s or len(s) < n:
        return set()
    return {s[i:i+n] for i in range(len(s) - n + 1)}


def char_ngram_jaccard(a: Optional[str], b: Optional[str], n: int) -> float:
    if not a or not b:
        return 0.0
    sa = _char_ngrams(str(a).lower(), n)
    sb = _char_ngrams(str(b).lower(), n)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# TOKEN UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _tokens(s: Optional[str]) -> list:
    if not s:
        return []
    return re.split(r'\s+', str(s).strip().lower())


def word_count_diff(a: Optional[str], b: Optional[str]) -> float:
    return abs(len(_tokens(a)) - len(_tokens(b)))


def word_count_ratio(a: Optional[str], b: Optional[str]) -> float:
    ca, cb = len(_tokens(a)), len(_tokens(b))
    if ca == 0 and cb == 0:
        return 1.0
    mx = max(ca, cb)
    return min(ca, cb) / mx if mx > 0 else 0.0


def abbrev_ratio(s: Optional[str]) -> float:
    """Fraction of tokens that are single characters (abbreviations)."""
    toks = _tokens(s)
    if not toks:
        return 0.0
    return sum(1 for t in toks if len(t) == 1) / len(toks)


def prefix_match(a: Optional[str], b: Optional[str], k: int) -> int:
    if not a or not b:
        return 0
    return int(str(a).lower()[:k] == str(b).lower()[:k])


def addr_char3_jaccard(a: Optional[str], b: Optional[str]) -> float:
    return char_ngram_jaccard(a, b, 3)


def addr_numeric_density(s: Optional[str]) -> float:
    if not s:
        return 0.0
    digits = sum(c.isdigit() for c in str(s))
    return digits / len(str(s)) if len(str(s)) > 0 else 0.0


def city_token_overlap(a: Optional[str], b: Optional[str]) -> float:
    """Jaccard of tokens likely to be city/state names (non-numeric, len > 3)."""
    def city_toks(s):
        return {t for t in _tokens(s) if len(t) > 3 and not t.isdigit()}
    sa = city_toks(a)
    sb = city_toks(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union > 0 else 0.0


def pincode_prefix_match(a: Optional[str], b: Optional[str], k: int = 3) -> int:
    if not a or not b:
        return 0
    sa = re.sub(r'\s+', '', str(a))
    sb = re.sub(r'\s+', '', str(b))
    return int(sa[:k] == sb[:k]) if len(sa) >= k and len(sb) >= k else 0


# ─────────────────────────────────────────────────────────────────────────────
# MAIN FEATURE EXTRACTOR (v2)
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_COLS_V1 = [
    'exact_clean_name', 'exact_stripped_name', 'exact_sorted_name',
    'name_jaro_winkler', 'name_token_sort', 'name_token_set',
    'name_len_diff', 'name_len_ratio',
    'exact_clean_address', 'addr_is_null_target', 'addr_jaro_winkler',
    'addr_token_jaccard', 'addr_token_overlap', 'numeric_token_jaccard',
    'house_number_match', 'contradiction_house_no',
    'postal_code_match', 'postal_both_present', 'contradiction_postal',
    'is_source3', 'channel_max_priority', 'channel_count',
    'name_x_addr', 'name_x_postal',
]

NEW_FEATURE_COLS = [
    'name_char2_jaccard', 'name_char3_jaccard',
    'name_prefix3_match', 'name_prefix5_match',
    'name_abbrev_ratio_q', 'name_abbrev_ratio_t',
    'name_word_count_diff', 'name_word_count_ratio',
    'addr_char3_jaccard', 'addr_word_count_diff',
    'addr_numeric_density_t',
    'city_token_overlap',
    'pincode_prefix_match',
    'candidate_rank', 'n_candidates',
    'name_x_addr_x_postal',
    'india_flag',
]

FEATURE_COLS_V2 = FEATURE_COLS_V1 + NEW_FEATURE_COLS


def compute_new_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame already containing v1 features, compute the additional
    v2 features and append them. Modifies df in-place, returns it.

    Expected columns from candidate generation:
      query_name, target_name         (raw / clean names)
      query_address, target_address   (address strings, may be null)
      query_postal, target_postal     (postal codes, may be null)
      query_country, target_country   (ISO country codes)
      channel_max_priority            (from v1 features)
      name_x_addr, name_x_postal     (from v1 features)
      postal_code_match               (from v1 features)
    """

    # Resolve name columns (try multiple names for compatibility)
    def col(candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    qname_col = col(['query_name', 'name_query', 'q_name'])
    tname_col = col(['target_name', 'name_target', 't_name'])
    qaddr_col = col(['query_address', 'addr_query', 'q_address', 'std_address_query'])
    taddr_col = col(['target_address', 'addr_target', 't_address', 'std_address_target'])
    qpost_col = col(['query_postal', 'postal_query', 'q_postal'])
    tpost_col = col(['target_postal', 'postal_target', 't_postal'])
    ctry_col  = col(['query_country', 'country_query', 'q_country', 'country'])

    print(f"  [v2] Name cols: ({qname_col}, {tname_col})")
    print(f"  [v2] Addr cols: ({qaddr_col}, {taddr_col})")
    print(f"  [v2] Postal cols: ({qpost_col}, {tpost_col})")
    print(f"  [v2] Country col: {ctry_col}")

    qnames = df[qname_col].tolist() if qname_col else [''] * len(df)
    tnames = df[tname_col].tolist() if tname_col else [''] * len(df)
    qaddrs = df[qaddr_col].tolist() if qaddr_col else [None] * len(df)
    taddrs = df[taddr_col].tolist() if taddr_col else [None] * len(df)
    qposts = df[qpost_col].tolist() if qpost_col else [None] * len(df)
    tposts = df[tpost_col].tolist() if tpost_col else [None] * len(df)

    n = len(df)

    # --- Character n-gram features ---
    nc2 = np.zeros(n, dtype=np.float32)
    nc3 = np.zeros(n, dtype=np.float32)
    np3 = np.zeros(n, dtype=np.int8)
    np5 = np.zeros(n, dtype=np.int8)
    nar_q = np.zeros(n, dtype=np.float32)
    nar_t = np.zeros(n, dtype=np.float32)
    nwcd = np.zeros(n, dtype=np.float32)
    nwcr = np.zeros(n, dtype=np.float32)

    ac3 = np.zeros(n, dtype=np.float32)
    awcd = np.zeros(n, dtype=np.float32)
    and_t = np.zeros(n, dtype=np.float32)
    cto = np.zeros(n, dtype=np.float32)
    ppm = np.zeros(n, dtype=np.int8)

    for i in range(n):
        qn, tn = qnames[i], tnames[i]
        qa, ta = qaddrs[i], taddrs[i]
        qp, tp = qposts[i], tposts[i]

        nc2[i] = char_ngram_jaccard(qn, tn, 2)
        nc3[i] = char_ngram_jaccard(qn, tn, 3)
        np3[i] = prefix_match(qn, tn, 3)
        np5[i] = prefix_match(qn, tn, 5)
        nar_q[i] = abbrev_ratio(qn)
        nar_t[i] = abbrev_ratio(tn)
        nwcd[i] = word_count_diff(qn, tn)
        nwcr[i] = word_count_ratio(qn, tn)

        ac3[i] = addr_char3_jaccard(qa, ta)
        awcd[i] = word_count_diff(qa, ta)
        and_t[i] = addr_numeric_density(ta)
        cto[i] = city_token_overlap(qa, ta)
        ppm[i] = pincode_prefix_match(qp, tp, k=3)

    df['name_char2_jaccard']    = nc2
    df['name_char3_jaccard']    = nc3
    df['name_prefix3_match']    = np3.astype(np.float32)
    df['name_prefix5_match']    = np5.astype(np.float32)
    df['name_abbrev_ratio_q']   = nar_q
    df['name_abbrev_ratio_t']   = nar_t
    df['name_word_count_diff']  = nwcd
    df['name_word_count_ratio'] = nwcr
    df['addr_char3_jaccard']    = ac3
    df['addr_word_count_diff']  = awcd
    df['addr_numeric_density_t'] = and_t
    df['city_token_overlap']    = cto
    df['pincode_prefix_match']  = ppm.astype(np.float32)

    # --- Candidate rank within query ---
    # Rank by name_x_addr descending; lower rank = better candidate
    if 'name_x_addr' in df.columns and ('query_id' in df.columns or 'entity_id' in df.columns):
        qid_col = 'query_id' if 'query_id' in df.columns else 'entity_id'
        df['candidate_rank'] = (
            df.groupby(qid_col)['name_x_addr']
            .rank(ascending=False, method='first')
            .astype(np.float32)
        )
        df['n_candidates'] = (
            df.groupby(qid_col)[qid_col]
            .transform('count')
            .astype(np.float32)
        )
    else:
        df['candidate_rank'] = 0.0
        df['n_candidates']   = 0.0

    # --- Three-way interaction ---
    df['name_x_addr_x_postal'] = (
        df['name_x_addr'].fillna(0) *
        df['postal_code_match'].fillna(0)
    ).astype(np.float32)

    # --- India flag ---
    if ctry_col and ctry_col in df.columns:
        df['india_flag'] = (df[ctry_col].str.upper() == 'IN').astype(np.float32)
    else:
        df['india_flag'] = 0.0

    return df


if __name__ == '__main__':
    print("Feature Engineering v2 module loaded.")
    print(f"V1 features ({len(FEATURE_COLS_V1)}): {FEATURE_COLS_V1}")
    print(f"New features ({len(NEW_FEATURE_COLS)}): {NEW_FEATURE_COLS}")
    print(f"Total v2 features: {len(FEATURE_COLS_V2)}")
