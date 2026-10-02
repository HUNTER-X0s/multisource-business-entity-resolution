# Phase 0 Documentation Revision Note

**Date:** September 25, 2026  
**Document Type:** Audit Change Log & Traceability Record  
**Author:** Senior Autonomous Applied ML Competition Architecture Team  
**Scope:** Formal corrections applied to Phase 0 deliverables (`docs/Phase_0_Research_Report.md` and `docs/Phase_0_Gate_Review.md`).

---

## Summary of Revisions

This revision pass strictly refined the interpretation scope, policy classifications, and hypothesis framing across the Phase 0 deliverables without modifying any verified numerical findings or starting any Phase 1 implementation.

### 1. Correction 1: France / Country Agreement Interpretation
* **What was corrected:** Removed claims that 100% country agreement in the test set was "confirmed" or that country blocking carries "zero false negative risk".
* **Why it was corrected:** 100% country agreement was empirically verified on training data ground truth (7,638,365 / 7,638,365 links). The test set ground truth is unobserved. Extrapolating a zero false-negative guarantee to the hidden test set is scientifically invalid.
* **Supporting evidence:** `research_scripts/02_scale_and_schema_results.json` (training ground truth analysis) and `research_scripts/06_deep_forensics_results.json` (test distribution analysis).
* **Did underlying finding change:** No. The empirical fact that 100.0% of training ground-truth links share identical country labels remains intact.
* **Effect on Phase 1:** Country-aware candidate generation remains a primary candidate architecture (reducing search space by ~65-75%), but hard country blocking is classified as an empirical hypothesis to be validated against validation folds in Phase 1. France is treated as an open-set string label.

### 2. Correction 2: Model License vs Dependency License Distinction
* **What was corrected:** Replaced statements implying that "all libraries and dependencies must be MIT/Apache 2.0" under official competition rules.
* **Why it was corrected:** Official challenge rules state: *"Final model should be a MIT/Apache 2.0 License model and up to 8 Billion parameters."* The official rule applies specifically to the final model, not to every software build tool or dependency.
* **Supporting evidence:** `dataset/student_resource/README.md` (official competition rules).
* **Did underlying finding change:** No.
* **Effect on Phase 1:** Distinguishes between the **Official Final Model Rule** (MIT/Apache 2.0, $\le 8$B params) and the **Internal Engineering Policy** (auditing packages for license cleanliness and reproducibility). Avoids falsely presenting internal best practices as official competition constraints.

### 3. Correction 3: NFKD / Accent Handling Status
* **What was corrected:** Removed statements presenting Unicode NFKD normalization as the "chosen" or "final" normalization strategy.
* **Why it was corrected:** While NFKD strips accents, raw accented characters may provide valuable discriminative signal for matching French entities. The tradeoff between retrieval recall and discrimination precision has not yet been benchmarked on actual French pairs.
* **Supporting evidence:** `research_scripts/07_leakage_and_script_audit_results.json` (15.72% of France names and 27.83% of France addresses have accents).
* **Did underlying finding change:** No. The observation that accents are localized to France records remains intact.
* **Effect on Phase 1:** Accent handling is classified as a **Provisional Multi-Representation Hypothesis / Deferred Design Decision**. Multiple representations (raw, normalized, accent-folded) will be experimentally compared in Phase 1.

### 4. Correction 4: Indic Transliteration Scope & Precision
* **What was corrected:** Replaced broad statements such as "Indic transliteration is unnecessary" with a precisely scoped statement.
* **Why it was corrected:** The empirical finding is that the observed test dataset contains 0.0% Indic script. This proves transliteration is not needed at test time, but does not preclude training-time transliteration experiments.
* **Supporting evidence:** `research_scripts/07_leakage_and_script_audit_results.json` (0.0% Indic script in test S1 sample).
* **Did underlying finding change:** No.
* **Effect on Phase 1:** Clarifies that Indic transliteration is excluded from the baseline pipeline because the observed test data does not require it. Training-time transliteration remains an optional experimental idea that would require separate empirical justification before adoption.

---

## Quality & Compliance Confirmation
* **Documents Modified:** `docs/Phase_0_Research_Report.md`, `docs/Phase_0_Gate_Review.md`.
* **New Document Created:** `docs/Phase_0_Revision_Notes.md`.
* **Phase 1 Implementation:** Zero implementation code written. No preprocessing, blocking, feature engineering, or modeling scripts have been created or modified.
* **Phase-Gate Status:** Phase 0 remains CLOSED and VERIFIED. Phase 1 remains UNAUTHORIZED.
