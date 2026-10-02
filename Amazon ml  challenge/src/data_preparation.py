"""
src/data_preparation.py
======================
High-performance text representation and normalization engine for Business Entity Resolution.
Constructs multi-representation views of business names and addresses without destroying raw data:
  - raw (original unchanged)
  - clean (lowercased, whitespace collapsed, normalized punctuation)
  - stripped_suffix (legal entity suffixes standardized / removed)
  - token_sorted (alphabetically sorted token bag for order-invariance)
  - accent_folded (Unicode NFKD ascii-folded representation for French accents)
  - postal_code (extracted PIN / ZIP code if present)
  - numeric_tokens (extracted street/suite/building numbers from address)
"""

import re
import unicodedata
from typing import List, Optional, Tuple, Set

# Precompiled regex patterns
WHITESPACE_RE = re.compile(r"\s+")
PUNCT_RE = re.compile(r"[^\w\s]")
DIGITS_RE = re.compile(r"\d+")

# Legal suffixes to standardize or strip
LEGAL_SUFFIXES = [
    "private limited", "pvt ltd", "pvt limited", "private ltd",
    "limited", "ltd", "pvt", "llc", "l.l.c.", "inc", "incorporated",
    "corporation", "corp", "llp", "l.l.p.", "co", "company",
    "gmbh", "sarl", "sa", "sas", "plc"
]

# Compile legal suffix regex (at end of string or as isolated token)
# Sort by length descending so multi-word suffixes match first
LEGAL_SUFFIXES_SORTED = sorted(LEGAL_SUFFIXES, key=len, reverse=True)
SUFFIX_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(s) for s in LEGAL_SUFFIXES_SORTED) + r")\b",
    re.IGNORECASE
)

# Postal code patterns
INDIA_PIN_RE = re.compile(r"\b([1-9]\d{5})\b")
US_ZIP_RE = re.compile(r"\b(\d{5})(?:-\d{4})?\b")
FRANCE_POSTAL_RE = re.compile(r"\b(?:0[1-9]|[1-8]\d|9[0-8])\d{3}\b")

# Stop words for business core tokens
STOP_WORDS = {
    "and", "the", "of", "in", "at", "for", "on", "a", "an", "&",
    "services", "service", "enterprises", "enterprise", "solutions",
    "international", "group", "holdings", "holding", "associates"
}


def clean_text(text: Optional[str]) -> str:
    """Basic lowercasing, ampersand normalization, punctuation removal, whitespace collapse."""
    if not text:
        return ""
    s = text.lower()
    s = s.replace("&", " and ")
    s = PUNCT_RE.sub(" ", s)
    s = WHITESPACE_RE.sub(" ", s).strip()
    return s


def strip_legal_suffixes(clean_text_str: str) -> str:
    """Strips legal entity suffixes from an already cleaned text string."""
    if not clean_text_str:
        return ""
    s = SUFFIX_PATTERN.sub(" ", clean_text_str)
    return WHITESPACE_RE.sub(" ", s).strip()


def sort_tokens(text_str: str) -> str:
    """Returns order-independent alphabetical sort of whitespace-delimited tokens."""
    if not text_str:
        return ""
    tokens = [t for t in text_str.split() if t]
    tokens.sort()
    return " ".join(tokens)


def fold_accents(text: Optional[str]) -> str:
    """Unicode NFKD decomposition to fold Latin accented characters to ASCII equivalents."""
    if not text:
        return ""
    nfkd = unicodedata.normalize("NFKD", text)
    ascii_bytes = nfkd.encode("ascii", "ignore")
    return ascii_bytes.decode("utf-8")


def extract_postal_code(address_str: Optional[str], country: Optional[str] = None) -> str:
    """Extracts standard 5-6 digit postal codes based on country context if available."""
    if not address_str:
        return ""
    
    country_upper = (country or "").upper()
    if country_upper == "INDIA":
        m = INDIA_PIN_RE.search(address_str)
        if m:
            return m.group(1)
    elif country_upper == "US":
        m = US_ZIP_RE.search(address_str)
        if m:
            return m.group(1)
    elif country_upper == "FRANCE":
        m = FRANCE_POSTAL_RE.search(address_str)
        if m:
            return m.group(0)
    else:
        # Generic postal regex: look for 5 or 6 digit numbers
        m = INDIA_PIN_RE.search(address_str) or US_ZIP_RE.search(address_str)
        if m:
            return m.group(1)
    return ""


def extract_numeric_tokens(address_str: Optional[str]) -> str:
    """Extracts all numeric tokens from an address string, sorted."""
    if not address_str:
        return ""
    nums = DIGITS_RE.findall(address_str)
    # Deduplicate while preserving order or sort
    unique_nums = sorted(set(nums), key=lambda x: (len(x), x))
    return " ".join(unique_nums)


def get_core_tokens(clean_name_str: str, min_len: int = 3) -> List[str]:
    """Extracts discriminative core tokens (length >= min_len, not in stop words)."""
    if not clean_name_str:
        return []
    tokens = clean_name_str.split()
    return [t for t in tokens if len(t) >= min_len and t not in STOP_WORDS]


def compute_all_representations(raw_name: str, raw_address: str, country: str) -> dict:
    """
    Computes the complete multi-representation dictionary for a single entity record.
    """
    clean_n = clean_text(raw_name)
    clean_a = clean_text(raw_address)
    
    suffix_stripped_n = strip_legal_suffixes(clean_n)
    token_sorted_n = sort_tokens(suffix_stripped_n if suffix_stripped_n else clean_n)
    
    accent_n = fold_accents(clean_n)
    accent_a = fold_accents(clean_a)
    
    postal = extract_postal_code(raw_address, country)
    num_tokens = extract_numeric_tokens(raw_address)
    core_tokens = get_core_tokens(suffix_stripped_n if suffix_stripped_n else clean_n)
    
    return {
        "raw_name": raw_name or "",
        "raw_address": raw_address or "",
        "country": (country or "").strip(),
        "clean_name": clean_n,
        "clean_address": clean_a,
        "stripped_suffix_name": suffix_stripped_n,
        "token_sorted_name": token_sorted_n,
        "accent_folded_name": accent_n,
        "accent_folded_address": accent_a,
        "postal_code": postal,
        "numeric_tokens": num_tokens,
        "core_tokens": core_tokens
    }
