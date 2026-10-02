"""
tests/test_data_preparation.py
==============================
Unit tests for data preparation and representation engineering functions.
"""

import sys
import os

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath("."))

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


def test_clean_text():
    assert clean_text("  Tata   Consultancy & Services  ") == "tata consultancy and services"
    assert clean_text("McDonald's Corp.") == "mcdonald s corp"
    assert clean_text(None) == ""
    assert clean_text("") == ""
    print("test_clean_text: PASSED")


def test_strip_legal_suffixes():
    assert strip_legal_suffixes("sharma enterprises pvt ltd") == "sharma enterprises"
    assert strip_legal_suffixes("acme global llc") == "acme global"
    assert strip_legal_suffixes("alpha beta private limited") == "alpha beta"
    assert strip_legal_suffixes("general electric company") == "general electric"
    print("test_strip_legal_suffixes: PASSED")


def test_sort_tokens():
    assert sort_tokens("bank of america") == "america bank of"
    assert sort_tokens("sharma enterprises") == "enterprises sharma"
    assert sort_tokens("") == ""
    print("test_sort_tokens: PASSED")


def test_fold_accents():
    assert fold_accents("société générale") == "societe generale"
    assert fold_accents("café de la paix") == "cafe de la paix"
    assert fold_accents("boulangerie saint-honoré") == "boulangerie saint-honore"
    print("test_fold_accents: PASSED")


def test_extract_postal_code():
    # India PIN
    assert extract_postal_code("123 MG Road, Bangalore 560001, Karnataka", "India") == "560001"
    # US ZIP
    assert extract_postal_code("742 Evergreen Terrace, Springfield, OR 97477-1234", "US") == "97477"
    # France Postal Code
    assert extract_postal_code("15 Rue de la Paix, 75002 Paris", "France") == "75002"
    # None / empty
    assert extract_postal_code("No postal code here", "US") == ""
    print("test_extract_postal_code: PASSED")


def test_extract_numeric_tokens():
    res = extract_numeric_tokens("Building 4, Flat 102, 5th Cross Road, Pin 560001")
    tokens = set(res.split())
    assert "4" in tokens
    assert "102" in tokens
    assert "5" in tokens
    assert "560001" in tokens
    print("test_extract_numeric_tokens: PASSED")


def test_compute_all_representations():
    reps = compute_all_representations(
        raw_name="Société Générale Private Limited",
        raw_address="15 Boulevard Haussmann, 75009 Paris",
        country="France"
    )
    assert reps["country"] == "France"
    assert "societe generale" in reps["accent_folded_name"]
    assert reps["postal_code"] == "75009"
    assert "15" in reps["numeric_tokens"]
    assert "75009" in reps["numeric_tokens"]
    print("test_compute_all_representations: PASSED")


if __name__ == "__main__":
    test_clean_text()
    test_strip_legal_suffixes()
    test_sort_tokens()
    test_fold_accents()
    test_extract_postal_code()
    test_extract_numeric_tokens()
    test_compute_all_representations()
    print("\nALL UNIT TESTS PASSED SUCCESSFULLY!")
