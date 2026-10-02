# Phase 1A — Preprocessing Decision Record (ADR)

**Project:** Amazon ML Challenge 2026 — Business Entity Resolution Challenge  
**Document:** Architectural Decision Record (Phase 1A)  
**Status:** Approved & Implemented  
**Date:** September 2026  

---

## 1. Context & Operating Mandate

Phase 1A mandated an autonomous, empirical investigation of the Amazon ML Challenge 2026 dataset (24.23M total rows across 6 tables) to determine what preprocessing, representation engineering, quality controls, and data contracts produce the safest, most information-preserving data foundation for business entity resolution.

This record formalizes the decisions made, the empirical evidence backing them, what was changed, what was rejected, and what remains deferred.

---

## 2. What the Dataset Demonstrated

1. **Pristine Reference Population:** Source 1 exhibits 0.000% missingness across all four required columns (`entity_id`, `business_name`, `business_address`, `country`) and 100% ID uniqueness across both training (2,206,821 rows) and test (1,732,544 rows).
2. **Noise and Missingness in Secondary Sources:** Sources 2 and 3 contain 2.6% - 3.4% null addresses (~610,000 entities across all tables) and 16k - 25k exact duplicate records. In ground truth, 337,018 true links (4.412% of all 7,638,365 true links across 2,206,821 S1 entities) connect to target entities with null addresses.
3. **Open-Set Country Distribution (The France Shift):**
   - Training contains only India (40.0%) and US (60.0%).
   - Test introduces France as a major new population: 259,452 entities in Test S1 (14.98%), 703,378 in Test S2 (14.39%), 731,615 in Test S3 (14.40%), totaling 1,694,445 French records across all test tables (14.48% of all 11.70M test records).
   - 63.16% of French S1 business names contain at least one of the top-5 French corporate suffixes (`sarl`, `sas`, `eurl`, `sa`, `sci`).
4. **Multiscript Discrepancy:** Source 1 is 100% Latin script. Sources 2 and 3 contain 13.37% (Test S2) and 7.53% (Test S3) Devanagari script in Indian names (10.39% combined of Indian target records, representing 490,171 records).
5. **Prevalence of Physical Branches:** 177,793 distinct names occur >1 time in S1, encompassing 845,385 physical entities (38.31% of S1). Address-derived signals are important for mitigating same-name branch ambiguity.

---

## 3. What Preprocessing Was Already Correct (Preserved)

1. **Polars Vectorized Execution Engine ([src/vectorized_pipeline.py](file:///z:/Amazon%20ML/src/vectorized_pipeline.py)):**
   - Batch columnar normalization processes 9.97M target entities in 6.0 seconds. Preserved without modification.
2. **Multi-Representation Data Core ([src/data_preparation.py](file:///z:/Amazon%20ML/src/data_preparation.py)):**
   - Stateless, functional implementations of `clean_text`, `strip_legal_suffixes`, `sort_tokens`, `fold_accents`, `extract_postal_code`, and `extract_numeric_tokens` confirmed robust, fast, and safe.
3. **Official Metric Evaluator ([src/evaluate.py](file:///z:/Amazon%20ML/src/evaluate.py)):**
   - Correctly models competition Macro $F_{0.5}$ with singleton cliff handling ($F_{0.5} = 0$ on false positives for singletons). Preserved.

---

## 4. What Was Missing & What Was Remediated

1. **Schema & Data Contract Validation Engine Added ([src/schema_validation.py](file:///z:/Amazon%20ML/src/schema_validation.py)):**
   - *Problem:* Lack of an automated schema barrier allowed malformed TSVs or corrupted rows to silently enter downstream pipelines.
   - *Remediation:* Implemented strict schema validation asserting required column presence, entity ID uniqueness, valid source prefixes (`S1-`, `S2-`, `S3-`), UTF-8 encoding, and non-empty IDs.
2. **Property & Invariant Test Suite Added ([tests/test_data_invariants.py](file:///z:/Amazon%20ML/tests/test_data_invariants.py)):**
   - *Problem:* Need automated regression verification of fundamental ER invariants.
   - *Remediation:* Created 9 dedicated invariant tests verifying raw data preservation, determinism, null/empty resilience, order-invariance, accent decomposition, and schema error handling. All 36 project tests now pass in 1.15s.
3. **Dead Code Removed from `src/`:**
   - *Problem:* Unused heuristic prototype files from before Phase 1 (`src/blocking.py` and `src/preprocess.py`) cluttered the workspace.
   - *Remediation:* Removed both files after verifying 0 imports and confirming full replacement by Phase 1 modules.

---

## 5. What Was Deliberately NOT Changed

1. **Zero Imputation of Missing Addresses:**
   - Imputing synthetic addresses would create catastrophic false collisions. Null addresses are preserved as empty strings with explicit `is_null` indicators.
2. **Zero Deduplication of Source Records:**
   - Exact duplicate records within S2/S3 represent legitimate separate observations or physical branch filings under distinct IDs. Deleting them violates the evaluation contract.
3. **Zero Outlier Exclusion:**
   - Short business names ($\le 3$ characters like `"IBM"`, `"HP"`, `"3M"`) are legitimate corporate entities. They are fully retained.

---

## 6. What Was Rejected

| Technique | Reason for Rejection | Evidence |
|---|---|---|
| **Soundex Phonetic Hashing** | Catastrophic collision explosion; collapses 90k names to 5.5k keys; max cluster 768 entities | `research_scripts/09_phonetic_and_linguistic_results.json` |
| **Metaphone & NYSIIS** | Cross-language distortion; English phonetic rules destroy French & Indian names | `research_scripts/09_phonetic_and_linguistic_results.json` |
| **Linguistic Stemming / Lemmatization** | Destroys proper nouns and corporate brand marks ("Target" -> "target", "General" -> "gener") | Linguistic audit |
| **Global Stopword Removal** | Strips entities like "The The" or "All In" to empty strings | Invariant testing |
| **Conventional Tabular ML Encoders** | One-hot, label encoding, and min-max scaling destroy nominal text identities | Tabular ML audit |

---

## 7. What Remains Experimental vs Deferred

- **`NOT IMPLEMENTED / DEFERRED HYPOTHESIS` — Devanagari Transliteration:**
  - Evaluated conceptually on Indian S2/S3 native script records. Unimplemented in code; candidate generation achieves 77.7% link recall using address numeric and PIN tokens. Transliteration remains a deferred hypothesis for downstream stages.
- **`DEFERRED` — Neural Dense Embeddings:**
  - Embedding models (Sentence-Transformers / Bi-encoders) are deferred to Phase 2 / Phase 3 candidate scoring and feature engineering.
- **`DEFERRED` — Machine Learning Classifiers:**
  - LightGBM, XGBoost, and CatBoost pair classification are deferred to Phase 2/3.

---

## 8. Downstream Retrieval & Candidate Generation Readiness

The multi-representation data foundation directly feeds the 7-channel candidate generation pipeline in [src/candidate_generation.py](file:///z:/Amazon%20ML/src/candidate_generation.py):
1. **Exact Clean Name Join** (prio 1.00)
2. **Token-Sorted Suffix-Stripped Name Join** (prio 0.95)
3. **Standardized Exact Address Join** (prio 0.90)
4. **Sorted Address Token Set Join** (prio 0.88)
5. **2-Word Name Bigram Join (rare)** (prio 0.75)
6. **Street Prefix (3 tokens + digit)** (prio 0.70)
7. **Distinctive Brand Token Join (rare)** (prio 0.60)

This architecture achieved **81.56% unbudgeted link recall (96.75% entity coverage)**, and successfully generated the full **52,025,219-pair test candidate set** (`output/candidate_pairs.tsv`) with 100% format validation against Amazon's official submission validator.

