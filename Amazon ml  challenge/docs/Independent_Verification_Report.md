# Phase 1A — Independent Verification, Reconciliation & Red-Team Audit

**Project:** Amazon ML Challenge 2026 — Business Entity Resolution Challenge  
**Role:** Senior Independent Technical Red-Team Reviewer  
**Date:** September 2026  
**Status:** Verification Complete & Fully Reconciled  
**Verification Script:** [research_scripts/10_reconcile_and_verify.py](file:///z:/Amazon%20ML/research_scripts/10_reconcile_and_verify.py)  
**Authoritative Evidence Artifact:** [research_scripts/10_verification_results.json](file:///z:/Amazon%20ML/research_scripts/10_verification_results.json)  

---

## 1. Red-Team Review Mandate & Approach

This audit was conducted as an adversarial review of all empirical claims, documentation, and code produced during Phase 1A. No claim was accepted on authority. Every number, denominator, percentage, code path, and historical statement was independently re-executed and verified against the actual challenge files, executable Python scripts, Polars Arrow buffers, and generated TSVs.

---

## 2. Systematic Verification & Reconciliation Matrix

| # | Investigated Claim / Topic | Claimed Value in Phase 1A Draft | Independently Verified Value | Verification Method & Artifact | Status | Required Documentation Correction |
|---|---|---|---|---|---|---|
| **1** | **France Test Records & Denominator** | "1,694,445 France records ... 14.98% of Test S1" | • **Test S1 France:** **259,452** / 1,732,544 = **14.975% (14.98%)**<br>• **Test S2 France:** **703,378** / 4,887,273 = **14.392% (14.39%)**<br>• **Test S3 France:** **731,615** / 5,082,316 = **14.395% (14.40%)**<br>• **Total France Across All Test Tables:** $\mathbf{1,694,445}$ / 11,702,133 = $\mathbf{14.480\% (14.48\%)}$ | Full Polars table scan in `10_reconcile_and_verify.py` on `test_source1/2/3.tsv` | `CORRECTED` | Disentangle the total France count across all test tables ($1,694,445$, denominator $11,702,133$) from the Test S1 count ($259,452$, denominator $1,732,544$). Never conflate S1 with the 3-source total. |
| **2** | **Ground-Truth Link Count & Terminology** | "Out of 2,206,821 ground-truth training links..." | • **Ground-Truth Rows (S1 Entities):** **2,206,821**<br>• **Total Exploded True Links:** $\mathbf{7,638,365}$<br>• **Singletons (0 matches):** 123,247 (5.58%)<br>• **Single-Match S1:** 119,157 (5.40%)<br>• **Multi-Match S1 (>1 match):** 1,964,417 (89.02%)<br>• **Avg Links / Non-Singleton S1:** 3.666 links | Exploded comma-separated `matched_entity_ids` in `train_ground_truth.tsv` | `CORRECTED` | Correct terminology: 2,206,821 is the count of **S1 entities** (rows in ground truth). The true link count is **7,638,365** exploded pairs. |
| **3** | **Null Target Address True Links** | "5,375 true links (0.24%) have a NULL target address" | • **Exploded True Links with NULL Target Address:** $\mathbf{337,018}$ out of 7,638,365 true links = $\mathbf{4.412\% (4.41\%)}$<br>• **S1 Entities with $\ge 1$ NULL Target:** 312,600 entities (14.17%) | Exploded ID lookup against `train_source2/3` null-address index in `10_reconcile_and_verify.py` | `CORRECTED` | The 5,375 figure was a code bug in `08_phase1a_forensics.py` which looked up unexploded comma-separated strings. The true link count with null target address is **337,018** (4.41%). |
| **4** | **French Corporate Suffix Coverage** | "covers 63.4% of French business names (73.4k sarl, 52.2k sas...)" | • **Denominator (France S1):** 259,452<br>• `sarl`: 73,486 (28.32%)<br>• `sas`: 52,278 (20.15%)<br>• `eurl`: 16,980 (6.54%)<br>• `sa`: 12,766 (4.92%)<br>• `sci`: 8,360 (3.22%)<br>• **Sum of Top 5:** 163,870 (63.16%)<br>• **Deduplicated Union (Top 5):** $\mathbf{163,859}$ (**63.16%**)<br>• **Double Counting:** Exactly 11 entities | Regex word-boundary search across all 259,452 French S1 names in `10_reconcile_and_verify.py` | `VERIFIED` | Verified at 63.16% union coverage. Clarify that double counting is negligible (11 entities) and show exact individual counts and deduplicated union. |
| **5** | **Devanagari Prevalence Reconciliation** | Phase 1A: "3-6% Devanagari in S2/S3"<br>Phase 0: "0.0% in test; 0.49% in train S2" | • **Train S1:** 0 (0.00%)<br>• **Train S2 Names:** **269,424 (5.35% total / 13.35% India)**<br>• **Train S3 Names:** **158,003 (2.99% total / 7.47% India)**<br>• **Test S1:** 0 (0.00%)<br>• **Test S2 Names:** **309,103 (6.33% total / 13.37% India)**<br>• **Test S3 Names:** **181,068 (3.56% total / 7.53% India)**<br>• **Either Name/Addr in Test S2:** **550,530 (11.27% total)** | Full Unicode character block `[\u0900-\u097F]` scan on all 24.23M rows | `CORRECTED & RECONCILED` | Reconcile discrepancy: Phase 0 only inspected `test_s1[:50000]` (which is indeed 0.0% Devanagari) and incorrectly generalized to all test tables. S2/S3 contain ~6% Devanagari in names (13.4% of Indian records). |
| **6** | **Transliteration Implementation Status** | "`EXPERIMENTAL` ... Evaluated as candidate fallback" | Transliteration is **NOT IMPLEMENTED** in `src/` and was **NOT BENCHMARKED** in code. It is an unverified hypothesis. | Direct code inspection of `src/` and `experiments/` | `CORRECTED` | Downgrade status from `EXPERIMENTAL` to `NOT IMPLEMENTED / DEFERRED HYPOTHESIS`. Do not claim it was evaluated in Phase 1. |
| **7** | **Candidate Generation Channel Count** | "6-channel retrieval pipeline" in handoff | Exactly **7 production retrieval channels** in [src/candidate_generation.py](file:///z:/Amazon%20ML/src/candidate_generation.py) and [run_candidate_generation_chunked.py](file:///z:/Amazon%20ML/run_candidate_generation_chunked.py) | Code inspection of `src/candidate_generation.py` lines 10-18, 36-43 | `CORRECTED` | Update documentation to state exactly 7 channels: 1) Exact Clean Name, 2) Token-Sorted Suffix-Stripped Name, 3) Standardized Exact Address, 4) Sorted Address Token Set, 5) 2-Word Bigram, 6) Street Prefix, 7) Rare Brand Token. |
| **8** | **IDF / Core-Token Leakage Audit** | "high-IDF core tokens" | `get_core_tokens` uses a static list of 20 domain stopwords (`STOP_WORDS`). `_compute_rare_bigrams` and `_compute_rare_brand_tokens` fit frequency counts exclusively on `targets_norm` (S2+S3) without S1 queries or ground truth. | Code inspection of `src/data_preparation.py` and `src/candidate_generation.py` | `VERIFIED` | Zero test query or ground truth leakage. Confirmed 100% competition compliant. |
| **9** | **Normalization Target Pool Runtime** | "~6.0 seconds" | **6.16 seconds** wall-clock time on 9,969,589 rows (`test_source2` + `test_source3`) | Benchmarked in `10_reconcile_and_verify.py` using Polars 1.44.2 on Intel Core i7-13700H, 16GB RAM | `VERIFIED` | Claim verified. Peak memory under 3.5GB. |
| **10** | **Exact Duplicate Records in S2 / S3** | S2 has 25k duplicates; S3 has 16k-18k duplicates | • `train_source2`: **25,873** exact duplicate rows<br>• `train_source3`: **18,860** exact duplicate rows<br>• `test_source2`: **22,641** exact duplicate rows<br>• `test_source3`: **16,293** exact duplicate rows | Multi-column Polars `.n_unique()` check on `(business_name, business_address, country)` | `VERIFIED` | Verified. Confirms why deduplication is harmful in ER (deleting rows destroys distinct entity IDs). |
| **11** | **Same-Name Physical Branch Prevalence** | 1.54M unique names for 2.21M S1 entities | Exactly **1,539,229 unique raw names** for **2,206,821 S1 entities** in `train_source1.tsv` (**667,592 non-unique entities**) | Polars `.n_unique()` on `train_source1.tsv` | `VERIFIED` | Verified. Address tokens are mandatory to prevent branch collapse. |
| **12** | **Test Set Output Compliance** | "52,025,219 pairs written to output/candidate_pairs.tsv" | • Exactly **52,025,219 pairs** written<br>• Exactly **1,732,544 rows** (100% matching `test_source1.tsv`)<br>• Queries with $\ge 1$ candidate: **1,728,442 (99.76%)**<br>• Zero format warnings or schema errors | Tested against official [src/validate_submission.py](file:///z:/Amazon%20ML/src/validate_submission.py) validator | `VERIFIED` | Verified. 100% compliant with Amazon's official submission requirements. |

---

## 3. Detailed Root-Cause Analyses of Identified Defects

### Defect 1: The France Count & Denominator Conflation
- **Root Cause:** In the test set, France records exist across all three sources:
  - `test_source1`: 259,452
  - `test_source2`: 703,378
  - `test_source3`: 731,615
  - **Sum:** $259,452 + 703,378 + 731,615 = 1,694,445$.
  The Phase 1A draft mistakenly wrote that there were "1,694,445 France records (14.98% of Test S1)". The percentage $14.975\%$ ($14.98\%$) belongs strictly to Test S1 ($259,452 / 1,732,544$). The sum $1,694,445$ belongs to the entire test set ($1,694,445 / 11,702,133 = 14.48\%$).
- **Remediation:** All Phase 1A documents updated to explicitly report the exact numerator and denominator for each source table.

### Defect 2: Ground Truth Links vs S1 Entities & The 5,375 Bug
- **Root Cause:**
  1. The author of the Phase 1A draft referred to `train_ground_truth.tsv`'s row count ($2,206,821$) as "links". In reality, $2,206,821$ is the number of **S1 entities**. Each row has a comma-separated list of target IDs. When exploded, there are **7,638,365 individual true links** (averaging 3.666 links per non-singleton S1 entity).
  2. In `research_scripts/08_phase1a_forensics.py`, the code checked:
     `true_links_with_null_target_addr = sum(1 for tid in gt_targets if tid in all_null_target_ids)`
     Because `gt_targets` was read directly without splitting by comma, `tid` was the full string `"S2-1234,S3-5678"`, which could never match a single entity ID in `all_null_target_ids`! The only rows that matched were single-match entities where the single ID had a null address (5,375 rows).
  3. When properly exploded, **337,018 true links** (4.41% of all 7.64M links) connect to a target entity with a null address!
- **Remediation:** Fixed forensic calculation and updated all documentation with the true exploded count ($337,018$ true links, $4.41\%$).

### Defect 3: Devanagari Script Discrepancy with Phase 0
- **Root Cause:**
  - In Phase 0, `research_scripts/07_train_test_overlap_and_leakage_audit.py` only evaluated `test_s1[:50000]`. Because Source 1 is the clean reference database, S1 contains 0.0% Devanagari. Phase 0 prematurely stated: "all observed test records contain 0.0% Indic script".
  - When Sources 2 and 3 were audited in Phase 1A, Devanagari was discovered in **309,103 test S2 records (6.33%)** and **181,068 test S3 records (3.56%)**. Among Indian entities in S2, **13.37%** have Devanagari business names!
- **Remediation:** Phase 0's claim was reconciled as an incomplete audit scope (audited S1 only). Phase 1A's full-table measurements are confirmed authoritative.

### Defect 4: Transliteration Status Mischaracterization
- **Root Cause:** The Phase 1A summary labeled Devanagari transliteration as "`EXPERIMENTAL` — Evaluated as candidate fallback". However, an audit of `src/` and `experiments/` revealed that no transliteration library or script was ever implemented or executed.
- **Remediation:** Downgraded status from `EXPERIMENTAL` to `NOT IMPLEMENTED / DEFERRED HYPOTHESIS`.

### Defect 5: 6 vs 7 Candidate Generation Channels
- **Root Cause:** The Phase 1A handoff listed 6 retrieval channels, but [src/candidate_generation.py](file:///z:/Amazon%20ML/src/candidate_generation.py) explicitly implements and executes **7 channels** (adding Street Prefix as Channel 6).
- **Remediation:** Updated all documentation to uniformly reflect the exact 7-channel production architecture.

---

## 4. Synthesis: Verified Facts vs Corrected Facts

### A. Verified Facts
1. **Pristine Reference Population:** Source 1 is 100% complete across both train (2,206,821) and test (1,732,544), with 0 missing names, 0 missing addresses, and 0 duplicate IDs.
2. **French Suffix Coverage:** 63.16% of French S1 entities contain at least one of the top-5 French legal designations (`sarl`, `sas`, `eurl`, `sa`, `sci`). Double-counting is negligible (11 entities).
3. **Execution Runtime:** Target pool normalization (9.97M rows) runs in **6.16 seconds** with under 3.5GB peak RAM on Polars 1.44.2.
4. **Leakage Safety:** Zero test query or ground truth labels are leaked into representation or frequency filters.
5. **Output Compliance:** `output/candidate_pairs.tsv` contains 52,025,219 pairs for 1,732,544 test S1 queries, passing the official Amazon challenge validator with 0 errors.

### B. Corrected Facts
1. **France Counts & Percentages:**
   - Test S1: **259,452** (14.98% of 1,732,544).
   - Test S2: **703,378** (14.39% of 4,887,273).
   - Test S3: **731,615** (14.40% of 5,082,316).
   - Combined Test: **1,694,445** (14.48% of 11,702,133).
2. **Ground Truth Counts:**
   - 2,206,821 is the count of **S1 entities** (ground truth rows).
   - **7,638,365** is the count of exploded true links.
   - **337,018** true links (4.41%) connect to a target entity with a null address (correcting the defective 5,375 figure).
3. **Channel Count:** Exactly **7 retrieval channels**, not 6.
4. **Transliteration Status:** Classified as **NOT IMPLEMENTED / DEFERRED HYPOTHESIS**, not experimental.

### C. Contradictions Discovered & Reconciled
- **Phase 0 vs Phase 1A Script Audit:** Phase 0 claimed 0.0% Indic script in the test set. This was contradicted by full-table audits which showed 6.33% Devanagari in Test S2 and 3.56% in Test S3 (13.37% of Indian records). Reconciled: Phase 0 only inspected `test_s1[:50000]`.

---

## 5. Closure Assessment & Next Phase Handoff

### Can Phase 1A Legitimately Be Considered Closed?
**YES.**
With all numbers independently audited, denominators corrected, code defects in exploratory scripts resolved, and documentation reconciled against executable code, Phase 1A has established a solid, verified data foundation.

### Exact Handoff State for Phase 2:
1. **Data Invariants:** Enforced by [src/schema_validation.py](file:///z:/Amazon%20ML/src/schema_validation.py) and verified by 36 unit/invariant tests passing in 1.06s.
2. **Candidate Pool:** Production candidate pairs file `output/candidate_pairs.tsv` (52,025,219 candidate pairs across 1.73M queries) ready for feature extraction.
3. **Representation Hierarchy:** Verbatim raw text, clean text, suffix-stripped text, token-sorted text, accent-folded text, and numeric/postal tokens available for pairwise feature computation.
4. **Hardware State:** GPU (RTX 3050 6GB) remains completely unallocated and ready for Phase 2/3 embedding generation and gradient boosted trees.
