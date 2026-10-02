"""
tests/test_data_invariants.py
=============================
Property and invariant tests for Phase 1A Data Foundation.
Verifies that all representations preserve raw information, are deterministic,
handle edge cases/missingness gracefully, and enforce schema contracts.
"""

import pytest
import polars as pl
from src.data_preparation import (
    clean_text,
    strip_legal_suffixes,
    sort_tokens,
    fold_accents,
    extract_postal_code,
    extract_numeric_tokens,
    get_core_tokens,
    compute_all_representations
)
from src.schema_validation import (
    validate_dataframe_schema,
    SchemaValidationError
)


def test_raw_information_preservation():
    """Invariant: Raw input attributes are preserved verbatim in representation output."""
    raw_name = "  M/s Tata Consultancy Services Ltd.  "
    raw_addr = "123, Nariman Point, Mumbai - 400021"
    country = "India"
    
    reps = compute_all_representations(raw_name, raw_addr, country)
    assert reps["raw_name"] == raw_name
    assert reps["raw_address"] == raw_addr
    assert reps["country"] == country


def test_determinism_invariant():
    """Invariant: Repeated calls on identical inputs yield bit-for-bit identical representations."""
    name = "Acme Global Trading & Logistics Inc."
    addr = "450 7th Ave, New York, NY 10123"
    
    rep1 = compute_all_representations(name, addr, "US")
    rep2 = compute_all_representations(name, addr, "US")
    assert rep1 == rep2


def test_null_and_empty_resilience():
    """Invariant: Null, None, empty strings, and whitespace-only strings never crash functions."""
    edge_cases = [None, "", "   ", "\t\n", "---", "###"]
    for case in edge_cases:
        assert isinstance(clean_text(case), str)
        assert isinstance(strip_legal_suffixes(clean_text(case)), str)
        assert isinstance(sort_tokens(case or ""), str)
        assert isinstance(fold_accents(case), str)
        assert isinstance(extract_postal_code(case), str)
        assert isinstance(extract_numeric_tokens(case), str)
        assert isinstance(get_core_tokens(case or ""), list)


def test_token_sorting_order_invariance():
    """Invariant: Permutations of identical tokens produce identical token-sorted keys."""
    variant1 = "National Logistics Transport Services"
    variant2 = "Services Transport National Logistics"
    variant3 = "Logistics Services National Transport"
    
    sorted1 = sort_tokens(clean_text(variant1))
    sorted2 = sort_tokens(clean_text(variant2))
    sorted3 = sort_tokens(clean_text(variant3))
    
    assert sorted1 == sorted2 == sorted3
    assert sorted1 == "logistics national services transport"


def test_accent_folding_french_records():
    """Invariant: French accents decompose into ASCII equivalents without data loss."""
    raw_fr = "Société d'Équipement Général et Électronique"
    folded = fold_accents(clean_text(raw_fr))
    assert "societe" in folded
    assert "equipement" in folded
    assert "general" in folded
    assert "electronique" in folded
    assert all(ord(c) < 128 for c in folded)


def test_legal_suffix_stripping_precision():
    """Invariant: Legal suffix stripping only removes business entity legal designations."""
    assert strip_legal_suffixes("reliance industries limited") == "reliance industries"
    assert strip_legal_suffixes("tata consultancy services pvt ltd") == "tata consultancy services"
    assert strip_legal_suffixes("apple inc") == "apple"
    assert strip_legal_suffixes("carrefour sarl") == "carrefour"
    assert strip_legal_suffixes("bnp paribas sa") == "bnp paribas"


def test_postal_code_extraction_multinational():
    """Invariant: Postal codes extract accurately by country format."""
    # India 6-digit PIN
    assert extract_postal_code("Plot 42, Sector 18, Gurgaon, Haryana 122015", "India") == "122015"
    # US 5-digit ZIP
    assert extract_postal_code("100 Main St, Austin, TX 78701-1234", "US") == "78701"
    # France 5-digit code
    assert extract_postal_code("15 Rue de Rivoli, 75001 Paris", "France") == "75001"


def test_schema_validator_valid():
    """Contract test: Valid DataFrame passes schema validation cleanly."""
    df = pl.DataFrame({
        "entity_id": ["S1-001", "S1-002"],
        "business_name": ["Alpha Corp", "Beta LLC"],
        "business_address": ["123 Main St", "456 Oak Rd"],
        "country": ["US", "US"]
    })
    res = validate_dataframe_schema(df, source_prefix="S1-")
    assert res["status"] == "PASS"
    assert res["rows"] == 2


def test_schema_validator_catches_violations():
    """Contract test: Schema validator catches missing columns, duplicate IDs, and invalid prefixes."""
    # Missing column
    df_bad_cols = pl.DataFrame({
        "entity_id": ["S1-001"],
        "business_name": ["Alpha Corp"]
    })
    with pytest.raises(SchemaValidationError, match="Missing required columns"):
        validate_dataframe_schema(df_bad_cols)
        
    # Duplicate IDs
    df_dup_ids = pl.DataFrame({
        "entity_id": ["S1-001", "S1-001"],
        "business_name": ["Alpha Corp", "Alpha Corp"],
        "business_address": ["123 Main St", "123 Main St"],
        "country": ["US", "US"]
    })
    with pytest.raises(SchemaValidationError, match="Entity IDs are not unique"):
        validate_dataframe_schema(df_dup_ids)
        
    # Invalid prefix
    df_bad_prefix = pl.DataFrame({
        "entity_id": ["S2-001"],
        "business_name": ["Alpha Corp"],
        "business_address": ["123 Main St"],
        "country": ["US"]
    })
    with pytest.raises(SchemaValidationError, match="invalid prefixes"):
        validate_dataframe_schema(df_bad_prefix, source_prefix="S1-")
