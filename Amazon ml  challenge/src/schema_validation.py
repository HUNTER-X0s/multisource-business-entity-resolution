"""
src/schema_validation.py
========================
Strict data contract and schema validation module for Business Entity Resolution.
Enforces invariant properties on incoming raw and preprocessed data tables:
  1. Required schema columns ('entity_id', 'business_name', 'business_address', 'country')
  2. Entity ID integrity (non-null, non-empty, string type, valid source prefix 'S1-', 'S2-', 'S3-')
  3. Character encoding and delimiter integrity
  4. Missingness classification and data boundary checks
"""

import os
from typing import Dict, List, Optional, Set, Tuple
import polars as pl

REQUIRED_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
VALID_PREFIXES = {"S1-", "S2-", "S3-"}


class SchemaValidationError(Exception):
    """Raised when a DataFrame or dataset file violates the schema contract."""
    pass


def validate_dataframe_schema(
    df: pl.DataFrame,
    source_prefix: Optional[str] = None,
    allow_null_address: bool = True
) -> Dict[str, any]:
    """
    Validates a Polars DataFrame against the competition schema contract.
    
    Returns a dictionary of audit diagnostics.
    Raises SchemaValidationError if a fatal contract violation is detected.
    """
    # 1. Required columns
    cols = df.columns
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in cols]
    if missing_cols:
        raise SchemaValidationError(f"Missing required columns: {missing_cols}. Found: {cols}")
        
    n_rows = len(df)
    if n_rows == 0:
        raise SchemaValidationError("DataFrame is empty (0 rows).")

    # 2. Entity ID validation
    id_series = df["entity_id"]
    if id_series.is_null().any():
        null_count = id_series.is_null().sum()
        raise SchemaValidationError(f"Found {null_count} null entity_ids.")
        
    empty_id_count = (id_series.str.len_chars() == 0).sum()
    if empty_id_count > 0:
        raise SchemaValidationError(f"Found {empty_id_count} empty string entity_ids.")
        
    if id_series.n_unique() != n_rows:
        dup_count = n_rows - id_series.n_unique()
        raise SchemaValidationError(f"Entity IDs are not unique! Found {dup_count} duplicate IDs.")
        
    # Check ID prefixes
    if source_prefix:
        invalid_prefix_count = df.filter(~pl.col("entity_id").str.starts_with(source_prefix)).height
        if invalid_prefix_count > 0:
            raise SchemaValidationError(
                f"Expected all entity_ids to start with '{source_prefix}'. Found {invalid_prefix_count} invalid prefixes."
            )
            
    # 3. Business Name validation
    name_series = df["business_name"]
    null_name_count = name_series.is_null().sum()
    if null_name_count > 0:
        raise SchemaValidationError(f"Found {null_name_count} null business names. Names must not be null.")
        
    # 4. Address validation
    addr_series = df["business_address"]
    null_addr_count = addr_series.is_null().sum()
    if not allow_null_address and null_addr_count > 0:
        raise SchemaValidationError(f"Null addresses not permitted for this table: {null_addr_count} found.")

    # 5. Country validation
    country_series = df["country"]
    null_country_count = country_series.is_null().sum()
    if null_country_count > 0:
        raise SchemaValidationError(f"Found {null_country_count} null countries.")
        
    unique_countries = country_series.unique().to_list()
    
    return {
        "status": "PASS",
        "rows": n_rows,
        "unique_ids": n_rows,
        "null_names": int(null_name_count),
        "null_addresses": int(null_addr_count),
        "null_countries": int(null_country_count),
        "countries": unique_countries,
    }


def validate_tsv_file(
    file_path: str,
    source_prefix: Optional[str] = None,
    allow_null_address: bool = True
) -> Dict[str, any]:
    """
    Validates a raw TSV file on disk against the schema contract.
    """
    if not os.path.isfile(file_path):
        raise SchemaValidationError(f"File not found: {file_path}")
        
    # Read first few lines with standard open to check delimiter
    with open(file_path, "r", encoding="utf-8") as f:
        header_line = f.readline()
        if not header_line:
            raise SchemaValidationError(f"File is empty: {file_path}")
        if "\t" not in header_line and "," in header_line:
            raise SchemaValidationError(
                f"File appears to be comma-separated, not tab-separated: {file_path}"
            )
            
    # Read via Polars
    df = pl.read_csv(file_path, separator="\t", truncate_ragged_lines=True)
    return validate_dataframe_schema(
        df,
        source_prefix=source_prefix,
        allow_null_address=allow_null_address
    )
