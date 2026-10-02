# Phase 1A — Preprocessing Error Analysis & Taxonomy

**Project:** Amazon ML Challenge 2026 — Business Entity Resolution Challenge  
**Document:** Error Analysis, Failure Modes & Remediation Taxonomy  
**Date:** September 2026  

---

## 1. Executive Summary

This document formalizes the structured taxonomy of preprocessing problems, corruption patterns, and edge cases discovered during the forensic audit of the Amazon ML Challenge 2026 dataset (24.23M rows). 

For each error class, this document provides concrete examples from the raw dataset, quantifies their incidence, explains their failure mechanisms, and details the architectural remediation implemented.

---

## 2. Structured Error Taxonomy

### Error Class 1: Branch & Franchise Collapse (Same Name, Distinct Locations)
- **Description:** Different real-world entities share an identical corporate or commercial trade name (e.g. retail chains, banks, fuel stations, franchised restaurants).
- **Dataset Incidence:**
  - In `train_source1`, there are **2,206,821 entities**, and **1,539,229 distinct raw business names**.
  - **177,793 distinct names occur >1 time**, encompassing **845,385 physical entities (38.31% of the entire S1 population)**!
  - Excess duplicate occurrences: **667,592**.
- **Concrete Example:**
  - Entity `S1-000104`: `"State Bank of India"` at `"Plot 12, MG Road, Bangalore"`
  - Entity `S1-000892`: `"State Bank of India"` at `"Sector 17, Chandigarh"`
- **Failure Mode if Mismanaged:** Naive deduplication or name-only candidate blocking collapses hundreds of distinct physical locations into a single entity, destroying precision and triggering the severe $F_{0.5}$ no-match penalty.
- **Architectural Remediation:**
  1. Business names are **never deduplicated** or clustered in isolation.
  2. Address-derived signals (street number extraction via `extract_numeric_tokens`, locality tokens, and postal codes) are important for mitigating same-name branch ambiguity.

---

### Error Class 2: The French Diacritic & Legal Form Disconnect
- **Description:** Source 1 entities in the test set contain French entities (~15% of test data, 259,452 entities) with Latin accents and French corporate forms. External sources S2/S3 often strip accents or abbreviate corporate forms.
- **Dataset Incidence:**
  - 14.98% of test Source 1 entities are located in France.
  - Over 63% of French test entities contain French legal designations (`sarl`: 73,486, `sas`: 52,278, `eurl`: 16,980, `sa`: 12,766).
  - 2.3% of French names in S1 feature non-ASCII accented characters (`é`, `è`, `ê`, `à`, `ç`, `ô`, `î`, `ù`).
- **Concrete Example:**
  - S1: `"Société d'Exploitation Commerciale SARL"`
  - S2: `"societe d exploitation commerciale"`
- **Failure Mode if Mismanaged:** Exact match or case-only normalization fails due to diacritic mismatch (`é` != `e`) and suffix retention (`sarl` != `""`).
- **Architectural Remediation:**
  1. `fold_accents()` uses Unicode NFKD decomposition followed by ASCII encoding to guarantee accent-invariance.
  2. `strip_legal_suffixes()` includes all primary French legal forms (`sarl`, `sas`, `sa`, `eurl`, `sci`, `snc`, `ste`).

---

### Error Class 3: S2/S3 Native Script Discrepancy (Devanagari vs Latin)
- **Description:** Source 1 is 100.0% Latin script. However, Source 2 and Source 3 contain native Devanagari script for Indian entities.
- **Dataset Incidence:**
  - S1 names: 0.0% Devanagari.
  - S2 names: 5.38% Devanagari in train, 6.22% in test (~300,000 entities).
  - S3 names: 2.91% Devanagari in train, 3.60% in test (~180,000 entities).
- **Concrete Example:**
  - S1: `"Shree Ganesh Enterprises"`
  - S2: `"श्री गणेश एंटरप्राइजेज"`
- **Failure Mode if Mismanaged:** Pure text matching achieves 0.0% string similarity between English and Devanagari characters.
- **Architectural Remediation:**
  1. Address numeric tokens (PIN codes, street numbers) are preserved and extracted in ASCII digits, providing a potential fallback signal.
  2. Multi-representation engine isolates Devanagari records without crashing Latin normalizers; automated transliteration remains a deferred hypothesis.

---

### Error Class 4: Missing Address Records in Noisy Sources
- **Description:** A significant fraction of Source 2 and Source 3 records contain null or empty addresses, while Source 1 addresses are always present.
- **Dataset Incidence:**
  - `train_source2`: 168,967 null addresses (3.36%).
  - `train_source3`: 175,916 null addresses (3.33%).
  - `test_source2`: 129,408 null addresses (2.65%).
  - Ground Truth: Exactly 337,018 exploded true links (4.41% of all 7.64M true links) connect to a target entity with a null address in S2/S3 (affecting 312,600 S1 entities).
- **Concrete Example:**
  - S1: `"Alpha Tech Solutions Inc."`, `"100 Silicon Way, San Jose, CA 95112"`
  - S2: `"Alpha Tech Solutions"`, `NULL`
- **Failure Mode if Mismanaged:** Synthetic imputation (filling with "Unknown Address") creates massive artificial address collisions. Hard address blocking drops all true links.
- **Architectural Remediation:**
  1. Zero-imputation policy: null addresses remain empty strings with explicit `is_null` flags.
  2. Multi-channel retrieval union ensures that pure name channels (exact clean name, sorted token join, brand token join) retrieve candidates when address channels return zero.

---

### Error Class 5: Catastrophic Phonetic Collisions (Soundex & Metaphone)
- **Description:** Phonetic hashing algorithms map phonetically similar words to short alphanumeric codes.
- **Dataset Incidence:**
  - In a 100,000 S1 name audit, Soundex reduced 90,483 unique names to only 5,529 keys (16.37x compression).
  - The top Soundex key (`I536`) alone contained **768 entities** in just a 100k sample!
- **Concrete Example:**
  - Distinct entities: `"Indian Oil"`, `"International Paper"`, `"Intel Corp"` all map to similar phonetic codes.
- **Failure Mode if Mismanaged:** Explodes candidate pair generation to millions of irrelevant false positives, exhausting RAM and degrading downstream precision.
- **Architectural Remediation:**
  - Complete rejection of phonetic hashing. Candidate retrieval relies on token n-grams and frequency-capped character n-grams instead.

---

### Error Class 6: Extreme Name Length & Truncation Anomalies
- **Description:** Business names range from extreme abbreviations (2-3 characters) to deeply descriptive commercial filings (over 100 characters).
- **Dataset Incidence:**
  - Short names ($\le 3$ chars): 561 in train S1, 2,946 in train S2, 17,006 in train S3, 285 in test S1, 12,559 in test S2, 27,142 in test S3.
  - Long names: Maximum length 105 characters in S1, 123 characters in S3.
- **Concrete Example:**
  - Short: `"IBM"`, `"HP"`, `"3M"`, `"BMW"`, `"KFC"`
  - Long: `"All India Council For Technical Education Western Regional Office Engineering And Technology Directorate"`
- **Failure Mode if Mismanaged:** Length filters or aggressive token stripping discard valid short acronyms or over-fragment long names.
- **Architectural Remediation:**
  - No outlier exclusion. Short names are protected by requiring exact string or prefix agreement rather than character bigrams.
