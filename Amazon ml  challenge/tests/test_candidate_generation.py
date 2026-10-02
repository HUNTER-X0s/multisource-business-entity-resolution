"""
tests/test_candidate_generation.py
===================================
Unit and integration tests for src/candidate_generation.py.
Run with: python -m pytest tests/test_candidate_generation.py -v
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath("."))
import polars as pl
import pytest

from src.candidate_generation import (
    normalize_entities,
    generate_candidates,
    generate_candidates_and_save,
    _compute_rare_bigrams,
    _compute_rare_brand_tokens,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _make_df(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema={
        "entity_id": pl.Utf8,
        "business_name": pl.Utf8,
        "business_address": pl.Utf8,
        "country": pl.Utf8,
    })


@pytest.fixture
def simple_entities():
    return _make_df([
        {"entity_id": "s1_001", "business_name": "Apex Solutions Pvt Ltd",
         "business_address": "12 Main Street, Springfield, IL",
         "country": "US"},
        {"entity_id": "s1_002", "business_name": "Green Valley Inc",
         "business_address": "45 Oak Avenue, Shelby, NC",
         "country": "US"},
        {"entity_id": "s1_003", "business_name": "Tech Corp",
         "business_address": "",
         "country": "US"},
    ])


@pytest.fixture
def target_entities():
    return _make_df([
        # Exact name match to s1_001
        {"entity_id": "t2_001", "business_name": "Apex Solutions Pvt Ltd",
         "business_address": "12 Main Street, Springfield, IL",
         "country": "US"},
        # Sorted name match to s1_001 (word order changed)
        {"entity_id": "t2_002", "business_name": "Solutions Apex",
         "business_address": "12 Main St, Springfield, Illinois",
         "country": "US"},
        # Address match to s1_002
        {"entity_id": "t2_003", "business_name": "GreenValley Enterprises",
         "business_address": "45 Oak Avenue, Shelby, NC",
         "country": "US"},
        # Different country — should not match
        {"entity_id": "t2_004", "business_name": "Apex Solutions Pvt Ltd",
         "business_address": "12 Main Street, Springfield, IL",
         "country": "INDIA"},
        # Should not match anything
        {"entity_id": "t2_005", "business_name": "Unrelated Business XYZ",
         "business_address": "999 Nowhere Road",
         "country": "US"},
    ])


# ---------------------------------------------------------------------------
# normalize_entities tests
# ---------------------------------------------------------------------------
class TestNormalizeEntities:
    def test_output_columns(self, simple_entities):
        result = normalize_entities(simple_entities)
        expected_cols = {"entity_id", "business_name", "business_address", "country",
                         "clean_name", "stripped_name", "sorted_name",
                         "std_address", "sorted_addr", "postal_code"}
        assert expected_cols.issubset(set(result.columns))

    def test_legal_suffix_stripped(self, simple_entities):
        result = normalize_entities(simple_entities)
        row = result.filter(pl.col("entity_id") == "s1_001").row(0, named=True)
        # "Pvt Ltd" should be stripped
        assert "pvt" not in row["stripped_name"]
        assert "ltd" not in row["stripped_name"]

    def test_sorted_name_order_invariant(self):
        df = _make_df([
            {"entity_id": "a", "business_name": "Bravo Alpha Corp",
             "business_address": "1 Test St", "country": "US"},
            {"entity_id": "b", "business_name": "Alpha Bravo",
             "business_address": "1 Test St", "country": "US"},
        ])
        result = normalize_entities(df)
        rows = {r["entity_id"]: r["sorted_name"] for r in result.iter_rows(named=True)}
        # After suffix stripping "corp" removed; both should sort to "alpha bravo"
        assert rows["a"] == rows["b"]

    def test_address_abbreviation_expansion(self):
        df = _make_df([
            {"entity_id": "a", "business_name": "Test",
             "business_address": "100 Oak St, Springfield, Indiana", "country": "US"},
        ])
        result = normalize_entities(df)
        addr = result["std_address"].to_list()[0]
        assert "street" in addr
        assert "in" in addr  # Indiana -> in
        assert "indiana" not in addr

    def test_empty_address_handled(self, simple_entities):
        result = normalize_entities(simple_entities)
        row = result.filter(pl.col("entity_id") == "s1_003").row(0, named=True)
        assert row["std_address"] == ""
        assert row["sorted_addr"] == ""
        assert row["postal_code"] == ""

    def test_null_name_handled(self):
        df = _make_df([
            {"entity_id": "x", "business_name": None,
             "business_address": "5 Test Rd", "country": "US"},
        ])
        result = normalize_entities(df)
        assert result["clean_name"].to_list()[0] == ""

    def test_india_postal_extraction(self):
        df = _make_df([
            {"entity_id": "i", "business_name": "Test",
             "business_address": "Plot 5, Connaught Place, New Delhi 110001", "country": "INDIA"},
        ])
        result = normalize_entities(df)
        assert result["postal_code"].to_list()[0] == "110001"

    def test_us_zip_extraction(self):
        df = _make_df([
            {"entity_id": "u", "business_name": "Test",
             "business_address": "123 Main St, Anytown, IL 62701-1234", "country": "US"},
        ])
        result = normalize_entities(df)
        assert result["postal_code"].to_list()[0] == "62701"


# ---------------------------------------------------------------------------
# Frequency filter tests
# ---------------------------------------------------------------------------
class TestFrequencyFilters:
    def test_rare_bigrams_computed(self, target_entities):
        t_norm = normalize_entities(target_entities)
        rare = _compute_rare_bigrams(t_norm, max_freq=500)
        assert "country" in rare.columns
        assert "name_bigram" in rare.columns
        assert len(rare) > 0

    def test_rare_brands_computed(self, target_entities):
        t_norm = normalize_entities(target_entities)
        rare = _compute_rare_brand_tokens(t_norm, max_freq=300)
        assert "country" in rare.columns
        assert "brand_token" in rare.columns


# ---------------------------------------------------------------------------
# generate_candidates tests
# ---------------------------------------------------------------------------
class TestGenerateCandidates:
    def test_returns_dataframe(self, simple_entities, target_entities):
        result = generate_candidates(simple_entities, target_entities, top_k=10, verbose=False)
        assert isinstance(result, pl.DataFrame)
        assert "entity_id" in result.columns
        assert "target_id" in result.columns

    def test_exact_name_match_found(self, simple_entities, target_entities):
        result = generate_candidates(simple_entities, target_entities, top_k=10, verbose=False)
        # s1_001 and t2_001 have identical names
        pair = result.filter(
            (pl.col("entity_id") == "s1_001") & (pl.col("target_id") == "t2_001")
        )
        assert len(pair) == 1

    def test_cross_country_no_match(self, simple_entities, target_entities):
        result = generate_candidates(simple_entities, target_entities, top_k=10, verbose=False)
        # t2_004 is INDIA, s1_001 is US — should NOT match
        cross = result.filter(
            (pl.col("entity_id") == "s1_001") & (pl.col("target_id") == "t2_004")
        )
        assert len(cross) == 0

    def test_address_match_found(self, simple_entities, target_entities):
        result = generate_candidates(simple_entities, target_entities, top_k=10, verbose=False)
        # s1_002 and t2_003 share same address "45 Oak Avenue, Shelby, NC"
        pair = result.filter(
            (pl.col("entity_id") == "s1_002") & (pl.col("target_id") == "t2_003")
        )
        assert len(pair) == 1

    def test_top_k_respected(self, simple_entities, target_entities):
        result = generate_candidates(simple_entities, target_entities, top_k=2, verbose=False)
        for s1_id in simple_entities["entity_id"].to_list():
            count = result.filter(pl.col("entity_id") == s1_id)
            assert len(count) <= 2

    def test_no_self_matches(self, simple_entities, target_entities):
        # If queries and targets overlap in entity_id, no self-match should appear
        result = generate_candidates(simple_entities, target_entities, top_k=10, verbose=False)
        self_matches = result.filter(pl.col("entity_id") == pl.col("target_id"))
        assert len(self_matches) == 0

    def test_empty_query(self, target_entities):
        empty_q = _make_df([])
        result = generate_candidates(empty_q, target_entities, top_k=10, verbose=False)
        assert len(result) == 0

    def test_empty_targets(self, simple_entities):
        empty_t = _make_df([])
        result = generate_candidates(simple_entities, empty_t, top_k=10, verbose=False)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# generate_candidates_and_save tests
# ---------------------------------------------------------------------------
class TestGenerateCandidatesAndSave:
    def test_tsv_format(self, simple_entities, target_entities):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "candidate_pairs.tsv")
            generate_candidates_and_save(
                simple_entities, target_entities,
                output_path=out_path,
                top_k=10,
                verbose=False,
            )
            assert os.path.exists(out_path)
            with open(out_path, encoding="utf-8") as f:
                lines = f.readlines()

            # First line is header
            assert lines[0].strip() == "source1_entity_id\tcandidate_entity_ids"

            # Every S1 entity should appear
            ids_written = {l.split("\t")[0] for l in lines[1:] if l.strip()}
            for s1_id in simple_entities["entity_id"].to_list():
                assert s1_id in ids_written

    def test_empty_candidate_line_written(self, target_entities):
        """Queries with 0 candidates must still appear in output with empty candidate column."""
        # A query with no possible matches
        q = _make_df([
            {"entity_id": "s1_no_match", "business_name": "ZZZUNKNOWN",
             "business_address": "0 Nowhere", "country": "FRANCE"},
        ])
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "candidate_pairs.tsv")
            generate_candidates_and_save(q, target_entities, output_path=out_path,
                                         top_k=10, verbose=False)
            with open(out_path, encoding="utf-8") as f:
                lines = f.readlines()
            # Use raw lines (no strip) to preserve the trailing tab on zero-candidate rows
            raw_data_lines = [l.rstrip("\n\r") for l in lines[1:] if l.strip()]
            # s1_no_match should appear, either as "s1_no_match\t" or just "s1_no_match"
            assert any("s1_no_match" in l for l in raw_data_lines)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
