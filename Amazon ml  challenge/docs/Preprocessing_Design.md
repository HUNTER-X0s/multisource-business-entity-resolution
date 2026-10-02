# Phase 1A — Preprocessing & Multi-Representation System Design

**Project:** Amazon ML Challenge 2026 — Business Entity Resolution Challenge  
**Document:** Preprocessing System Architecture & Engineering Design  
**Implementation Modules:** [src/data_preparation.py](file:///z:/Amazon%20ML/src/data_preparation.py), [src/vectorized_pipeline.py](file:///z:/Amazon%20ML/src/vectorized_pipeline.py), [src/schema_validation.py](file:///z:/Amazon%20ML/src/schema_validation.py)  
**Date:** September 2026  

---

## 1. Core Architectural Principles

1. **Non-Destructive Parallel Representations:**
   Raw inputs (`raw_name`, `raw_address`, `country`, `entity_id`) are preserved verbatim. Downstream stages (feature engineering, cross-encoders, and precision filtering) retain access to raw punctuation, casing, and legal suffixes to break ambiguities created by coarse representations.

2. **Orthogonal Specialized Representations:**
   Different retrieval channels have distinct failure modes. The representation layer exposes distinct views tailored to specific retrieval invariants:
   - Order-invariance (token bag sorting)
   - Trans-corporate invariance (legal suffix stripping)
   - Diacritic-invariance (accent folding)
   - Locality & street invariance (postal code & numeric token extraction)

3. **Open-Set Generalization:**
   The architecture makes zero closed-world assumptions regarding country codes, vocabulary sizes, or postal code formats. French entities appearing in the test set are treated first-class via unified Latin decomposition and multi-national postal code parsing.

4. **Zero-Hallucination Missingness Policy:**
   Missing addresses (3.3% of S2/S3) are never synthetically imputed. They are represented as explicit empty strings with boolean missingness indicators, allowing retrieval channels to branch safely.

---

## 2. Multi-Representation Hierarchy

```
RAW RECORD: {entity_id, business_name, business_address, country}
  │
  ├── 1. RAW VIEW (Unchanged)
  │      └── Preserved for downstream precision scoring & deep learning
  │
  ├── 2. CLEAN VIEW (clean_text)
  │      ├── Lowercased, whitespace collapsed
  │      ├── Ampersand normalized: '&' -> ' and '
  │      └── Punctuation removed (regex: [^\w\s])
  │
  ├── 3. CORPORATE VIEW (strip_legal_suffixes)
  │      ├── Applied to clean text
  │      ├── Regex-boundary stripping of 25+ legal corporate forms
  │      └── Full coverage of US, Indian, and French legal forms:
  │          ['private limited', 'pvt ltd', 'inc', 'corp', 'llc', 'sarl', 'sas', 'sa', 'eurl', 'sci', ...]
  │
  ├── 4. ORDER-INVARIANT VIEW (sort_tokens)
  │      ├── Alphabetically sorted whitespace tokens
  │      └── Resolves word permutations ("Logistics Global Services" == "Global Logistics Services")
  │
  ├── 5. ACCENT-FOLDED VIEW (fold_accents)
  │      ├── Unicode NFKD decomposition
  │      └── ASCII byte encoding (resolves French accents: 'é', 'è', 'ê', 'à', 'ç' -> 'e', 'a', 'c')
  │
  └── 6. LOCALITY & NUMERIC VIEWS
         ├── postal_code: 6-digit India PIN, 5-digit US ZIP, 5-digit French postal code
         ├── numeric_tokens: Sorted unique street/suite/building numbers
         └── core_tokens: Discriminative name tokens excluding common enterprise stopwords
```

---

## 3. Production Representation Specifications

| Representation Name | Input Source | Transformation Function | Primary Downstream Consumer | Collisions Introduced |
|---|---|---|---|---|
| `raw_name` | Raw TSV | Identity (`str`) | Phase 2 pairwise Levenshtein, Jaro-Winkler, Exact Match | Baseline (0%) |
| `raw_address` | Raw TSV | Identity (`str`) | Phase 2 address similarity, street-level validation | Baseline (0%) |
| `clean_name` | `raw_name` | `clean_text()` | Clean exact match channel | -0.79% unique |
| `clean_address` | `raw_address` | `clean_text()` | Address token overlap channel | -1.2% unique |
| `stripped_name` | `clean_name` | `strip_legal_suffixes()` | Core brand & multi-token retrieval channels | -8.37% unique |
| `token_sorted_name` | `stripped_name` | `sort_tokens()` | Sorted token-join retrieval channel | -0.10% unique |
| `accent_folded_name` | `clean_name` | `fold_accents()` | French entity matching & accent invariance | 0.0% on ASCII |
| `postal_code` | `raw_address` | `extract_postal_code()` | Postal code block booster (capped < 500) | Bounded |
| `numeric_tokens` | `raw_address` | `extract_numeric_tokens()` | Street number intersection filtering | Bounded |

---

## 4. Evaluated & Rejected Representations

1. **Phonetic Hash Representations (Soundex, Metaphone, NYSIIS):**
   - **Reason for Rejection:** Soundex reduced 90,483 unique names to 5,529 keys (16.4x compression ratio), causing single keys to contain up to 768 entities in a 100k sample. On a 10M record target pool, this generates millions of false candidates per query. Metaphone and NYSIIS exhibited severe corruption on multilingual Indian and French names.
   - **Status:** **REJECTED**.

2. **Linguistic Stemming / Lemmatization (Porter, Snowball):**
   - **Reason for Rejection:** Business names are proper nouns and corporate brand marks. Stemming conflates distinct corporate entities (e.g. "Target" vs "Targeting", "General" vs "Generally") and truncates coined brand names.
   - **Status:** **REJECTED**.

3. **Destructive Global Stopword Stripping:**
   - **Reason for Rejection:** Common words like "The", "In", "All", "Group" frequently form the entire distinct identity of entities (e.g. "The The", "All In"). Global deletion leaves strings empty.
   - **Status:** **REJECTED**. Stopword filtering is restricted exclusively to secondary candidate ranking signals.

4. **Conventional Tabular ML Encoders (Imputation, One-Hot, Scalar Scaling):**
   - **Reason for Rejection:** Synthetic imputation hallucinates fake addresses, severely polluting entity resolution. One-hot and scalar scaling destroy nominal text identities.
   - **Status:** **REJECTED**.

---

## 5. Scalability & Engineering Implementation

1. **Polars Columnar Acceleration:**
   - Text operations are vectorized using Polars expressions and optimized Rust-level string operations.
   - Target pool normalization (9,969,589 rows) completes in **6.0 seconds**.
   - Peak RAM usage during full 10M normalization remains under **3.5 GB**.

2. **Deterministic Processing:**
   - All transformations are purely functional and stateless.
   - Bit-for-bit reproducibility verified by [test_data_invariants.py](file:///z:/Amazon%20ML/tests/test_data_invariants.py).

3. **Validation & Data Contract Hardening:**
   - [src/schema_validation.py](file:///z:/Amazon%20ML/src/schema_validation.py) acts as a strict gatekeeper at the ingestion boundary, verifying TSV delimiter purity, UTF-8 integrity, mandatory column presence, and `S1-`/`S2-`/`S3-` ID format compliance.
