# Phase 1A — Preprocessing Data Quality & Forensic Report

**Project:** Amazon ML Challenge 2026 — Business Entity Resolution Challenge  
**Document:** Data Quality Report  
**Authoritative Forensic Source:** `research_scripts/08_phase1a_forensics_results.json` & `research_scripts/09_phonetic_and_linguistic_results.json`  
**Date:** September 2026  

---

## 1. Executive Summary & Table Inventory

The Amazon ML Challenge 2026 dataset comprises **24,229,173 total records** split between training and test sets. Source 1 represents the reference entity database, while Sources 2 and 3 represent noisy, incomplete external crawl and registry sources.

### Complete Table Scale & Schema Forensics

| Table | File | Total Rows | Unique IDs | Null Names | Null Addresses | Null Addr % | Null Country | Countries Present |
|---|---|---|---|---|---|---|---|---|
| **Train S1** | `train_source1.tsv` | **2,206,821** | 2,206,821 (100%) | 0 | 0 | 0.00% | 0 | India, US |
| **Train S2** | `train_source2.tsv` | **5,034,616** | 5,034,616 (100%) | 0 | 168,967 | 3.36% | 0 | India, US |
| **Train S3** | `train_source3.tsv` | **5,285,603** | 5,285,603 (100%) | 0 | 175,916 | 3.33% | 0 | India, US |
| **Test S1** | `test_source1.tsv` | **1,732,544** | 1,732,544 (100%) | 0 | 0 | 0.00% | 0 | **France**, India, US |
| **Test S2** | `test_source2.tsv` | **4,887,273** | 4,887,273 (100%) | 0 | 129,408 | 2.65% | 0 | **France**, India, US |
| **Test S3** | `test_source3.tsv` | **5,082,316** | 5,082,316 (100%) | 0 | 136,098 | 2.68% | 0 | **France**, India, US |
| **Total** | | **24,229,173** | **24,229,173** | **0** | **610,389** | **2.52%** | **0** | |

---

## 2. Key Data-Quality Findings Material to Preprocessing

### Finding 1: The France Open-Set Shift in Test Data
- **Observation:** In the training set, records belong exclusively to `India` (40.0%) and `US` (60.0%). France is completely absent from training data (0.0%).
- **Test Set Distribution (Exact Breakdown by Source Table):**
  - **Test S1:** Exactly **259,452** French entities out of 1,732,544 rows (**14.975% / 14.98%**).
  - **Test S2:** Exactly **703,378** French entities out of 4,887,273 rows (**14.392% / 14.39%**).
  - **Test S3:** Exactly **731,615** French entities out of 5,082,316 rows (**14.395% / 14.40%**).
  - **Total France Across All Test Tables:** $259,452 + 703,378 + 731,615 = \mathbf{1,694,445}$ records out of $11,702,133$ total test records (**14.480% / 14.48%**).
- **Architecture Impact:**
  1. Country handling must remain **strictly open-set**.
  2. Legal suffix tables must explicitly support French corporate designations: `sarl` (73,486 occurrences in S1), `sas` (52,278), `eurl` (16,980), `sa` (12,766), `sci` (8,360). The deduplicated union of these top-5 suffixes covers **163,859 unique French businesses (63.16% of all 259,452 French S1 records)**. Double-counting is negligible (11 entities).
  3. Latin accent decomposition (NFKD folding) is mandatory to prevent false mismatches between accented and unaccented French strings (`Société` vs `Societe`).

### Finding 2: Missingness Characteristics & Ground-Truth Behavior
- **Zero Missing Names:** Across all 24.23M rows, there is not a single null or empty business name. Business name is the universal anchor.
- **Address Missingness:**
  - Pristine in Source 1: 0 missing addresses in both train and test S1.
  - S2 missing addresses: 168,967 (3.36% in train), 129,408 (2.65% in test).
  - S3 missing addresses: 175,916 (3.33% in train), 136,098 (2.68% in test).
- **Ground Truth Link Analysis (Authoritative Reconciliation):**
  - `train_ground_truth.tsv` contains **2,206,821 rows**, which correspond 1-to-1 with the **2,206,821 S1 entities** in `train_source1.tsv`.
  - When exploded across all comma-separated target IDs, there are **7,638,365 total true links** (averaging 3.666 links per non-singleton S1 entity).
  - No-match S1 entities (0 matches): **123,247** (5.585%).
  - Single-match S1 entities (exactly 1 match): **119,157** (5.399%).
  - Multi-match S1 entities (>1 match): **1,964,417** (89.016%).
  - Exactly **337,018 true links (4.412% of all 7.64M true links)** connect to a target entity in S2/S3 that has a NULL address.
  - Exactly **312,600 S1 entities** (14.17%) have at least one matched target with a null address.
  - **Engineering Conclusion:** Conventional imputation (hallucinating synthetic addresses or filling with "unknown") would artificially corrupt address matching. Instead, retain an explicit empty string representation and route null-address records through pure name-based retrieval channels.


### Finding 3: Exact Record Duplicates and Branch Structure
- **Duplicate Records within Sources:**
  - `train_source2` contains **25,873** exact record duplicates `(business_name, business_address, country)`.
  - `train_source3` contains **18,860** exact record duplicates.
  - `test_source2` contains **22,641** exact duplicates.
  - `test_source3` contains **16,293** exact duplicates.
- **Why Deduplication is Harmful:** In Entity Resolution, repeated records represent distinct real-world source observations or distinct branches/licensing filings under different `entity_id` values. Deleting duplicate records would violate the 1-to-1 entity ID contract and cause unrecoverable false negatives.

### Finding 4: Same-Name Entity Distribution & Physical Branches
- **Total S1 Entities:** **2,206,821**
- **Distinct Raw Business Names:** **1,539,229**
- **Names Occurring Exactly Once:** **1,361,436** (88.45% of distinct names, covering 61.69% of S1 entities)
- **Names Occurring > 1 Time:** **177,793** distinct names (11.55% of name vocabulary)
- **Entities Belonging to Same-Name Groups (>1):** **845,385 entities** (**38.31%** of the entire S1 population belong to multi-entity name clusters!)
- **Excess Duplicate Occurrences:** **667,592** ($2,206,821 - 1,539,229$)
- **Largest Same-Name Group:** `"Primary Care Group"` with **253 distinct physical locations**, followed by `"Ear Nose & Throat Group"` (251) and `"Pediatric Group"` (222).
- **Architecture Impact:** Relying solely on business name matching will cause catastrophic branch collapse. Address-derived signals (street numbers, locality tokens, postal codes) are important for mitigating same-name branch ambiguity.


---

## 3. Normalization Collision Science (500,000 S1 Sample)

Empirical evaluation of 500,000 S1 business names across sequential normalization stages:

| Normalization Stage | Unique Representations | Unique Reduction | Max Collision Cluster | Forensics & Trade-off Assessment |
|---|---|---|---|---|
| **1. Raw Text** | 402,579 | Baseline | 61 | Preserves verbatim casing, punctuation, and legal terms |
| **2. Clean Text** | 399,385 | -0.79% | 61 | Lowercasing, ampersand standardisation (`&` -> `and`), whitespace collapse. Negligible ambiguity introduced |
| **3. Legal Suffix Stripped** | 365,952 | -8.37% | 118 | Strips `pvt ltd`, `inc`, `llc`, etc. Unifies 8.37% of variants, but increases max cluster to 118 |
| **4. Token Sorted** | 365,604 | -0.10% | 118 | Orders whitespace tokens alphabetically. High order-invariance benefit with near-zero additional collision penalty |
| **5. Accent Folded** | 399,385 | 0.00% (US/IN sample) | 61 | Converts NFKD Latin diacritics to ASCII. Zero penalty on US/India, essential for France |

---

## 4. Phonetic & Linguistic Transformation Analysis

Empirical evaluation of 100,000 business names comparing Soundex, Metaphone, and NYSIIS (`research_scripts/09_phonetic_and_linguistic_results.json`):

1. **Soundex (`HARMFUL`):**
   - Unique keys: **5,529** (compressed from 90,483 unique names; **16.37x compression ratio**).
   - Max cluster size: **768 entities** in a single bucket.
   - Result: Completely destroys identity discrimination; creates massive candidate explosion.
2. **Metaphone & NYSIIS (`REJECTED`):**
   - High compression and cross-language phonetic corruption. English phonetic rules misparse French and Indian business names.
3. **Linguistic Stemming (`HARMFUL`):**
   - Truncates corporate identities (e.g. "Target" vs "Targeting", "General" vs "Generally").
   - Result: Corporate names are proper nouns; standard NLP stemming is destructive.

---

## 5. Script & Character Distributions

| Source Table | Sample Size | Non-ASCII Name % | Devanagari Name % | Non-ASCII Addr % | Devanagari Addr % | Primary Script Characteristics |
|---|---|---|---|---|---|---|
| **Train S1** | 100,000 | 0.000% | 0.000% | 0.017% | 0.000% | 100% Latin / English |
| **Train S2** | 100,000 | 15.266% | 5.380% | 9.337% | 5.456% | Latin + 5.4% Devanagari |
| **Train S3** | 100,000 | 11.343% | 2.907% | 8.938% | 5.149% | Latin + 3-5% Devanagari |
| **Test S1** | 100,000 | 2.318% | 0.000% | 4.266% | 0.000% | Latin (English + French accents) |
| **Test S2** | 100,000 | 18.817% | 6.222% | 14.661% | 6.434% | Latin + French + 6.2% Devanagari |
| **Test S3** | 100,000 | 14.511% | 3.596% | 14.218% | 6.252% | Latin + French + 3-6% Devanagari |

### Key Insight:
- Source 1 is **always Latin script**.
- Because S1 is purely Latin, Devanagari matches in S2/S3 cannot match S1 via exact name equality. Address numeric tokens and PIN codes serve as a potential fallback signal, while transliteration is deferred to downstream feature engineering.
