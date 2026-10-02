# ML Challenge 2026: Business Entity Resolution Solution Documentation

**Team Name:** Team X  
**Team Members:**  
- **Subhankar Swain** (Government College of Engineering, Kalahandi)  
- **Anurag Swain** (Government College of Engineering, Kalahandi)  
- **Pradyumna Kumar Biswal** (Government College of Engineering, Kalahandi)  
- **Jahanabi Dalai** (Government College of Engineering, Kalahandi)  
**Submission Date:** September 27, 2026  
**Final Leaderboard Macro $F_{0.5}$ Score:** **`0.80539`** (Evaluated Rank: 4245 / 1,732,544 Test Entities)

---

## 1. Executive Summary

We present an enterprise-scale, production-grade Business Entity Resolution (ER) framework developed for the Amazon ML Challenge 2026. The objective is to resolve multi-source commercial entity fragments across three heterogeneous databases under the precision-heavy **Macro $F_{0.5}$** metric. Our architecture marries a high-throughput **7-Channel Polars Priority Union Retrieval Engine** (cutting $3 \times 10^{12}$ comparisons to $\le 40$ candidates per entity, $>99.999\%$ search space reduction) with a **44-Feature GPU-Accelerated Tri-Model Meta-Ensemble** (LightGBM, XGBoost, CatBoost, and an Out-Of-Fold Logistic Regression Stacker). 

To maximize the asymmetric $F_{0.5}$ objective, our pipeline integrates two breakthrough post-processing innovations: (1) a **Tiered Decision Boundary** ($\theta_1 = 0.55$ for primary target recovery, $\theta_2 = 0.65$ for secondary match suppression), and (2) a **Strict Physical Mutual Exclusivity Filter**, which eliminated 12,594 guaranteed false-merge collisions by mathematically enforcing the ground-truth law that real-world business targets are uniquely linked to at most one reference entity ($0.000\%$ multi-parent targets in training). This system achieved a verified score of **`0.80539` Macro $F_{0.5}$** on the official 1.73M-entity test set, operating at $97.2\%$ of the candidate-constrained theoretical ceiling.

---

## 2. Methodology

### 2.1 Problem Analysis & Data Forensic Insights
Exploratory data analysis across the 3 independent data sources revealed four critical systemic noise patterns:
1. **Lexical Churn & Legal Entity Aliasing:** Widespread variations in legal suffixes (`Corporation` vs `Corp`, `Private Limited` vs `Pvt Ltd`, `LLC`, `Inc`), trade names (DBA — "Doing Business As"), parenthetical notations, and punctuation artifacts (`&` vs `and`, hyphens, slashes).
2. **Unstructured & Truncated Address Morphology:** High variance in address representations, including missing postal PIN codes, abbreviated municipal descriptors (`Rd` vs `Road`, `St` vs `Street`, `Bldg`), component reordering (city before street), landmark-relative locations ("Near SBI ATM"), and missing street numbers.
3. **Multilingual Transliteration & Geographic Distribution Shift:** Training records span the US and India, with Indian entities exhibiting heavy Devanagari $\leftrightarrow$ phonetic Latin transliteration divergence. Crucially, the test set introduces an unseen third geographic domain (**France**) requiring open-world domain-invariant representations without hardcoded country filters.
4. **Metric Asymmetry ($F_{0.5}$ Optimization):** In macro-averaged $F_{0.5}$, precision is penalized with **$4\times$ the weight of recall** ($\beta^2 = 0.25$). A single false merge (false positive) severely punishes an entity query. More critically, true singletons (queries with zero matches) receive a score of $1.0$ if correctly predicted empty, but collapse catastrophically to **$0.0$** upon a single false positive prediction.

### 2.2 Solution Strategy
We adopted a decoupled, multi-stage **High-Recall Blocking $\rightarrow$ Multi-Model Pairwise Inference $\rightarrow$ Physical Consensus Post-Processing** architecture:

```
[Raw Sources: S1, S2, S3]
           │
           ▼
[7-Channel Polars Priority Retrieval Engine]  ──► Candidate Set (K <= 40, Reduction > 99.999%)
           │
           ▼
[44-Dimensional Pairwise Feature Extractor]   ──► Lexical, Spatial, Numeric & Domain Features
           │
           ▼
[Tri-Model GPU Meta-Ensemble]                 ──► LightGBM + XGBoost + CatBoost + Stacker
           │
           ▼
[Physical Mutual Exclusivity & Calibration]   ──► Veto Stolen Targets + Tiered Cutoffs
           │
           ▼
[Final Submissions]                           ──► matching_results.tsv (0.80539) & candidate_pairs.tsv
```

---

## 3. Candidate Generation (Blocking)

To process 1,732,544 test queries against millions of Source 2 and Source 3 records without quadratic cost, we designed an ultra-fast vectorized Polars retrieval engine utilizing 7 prioritized deterministic channels:

### 3.1 The 7 Retrieval Channels
1. **Channel 1 — Exact Clean Name (Priority 1):** Lowercased, alphanumeric-only, space-normalized exact match.
2. **Channel 2 — Token-Sorted Suffix-Stripped (Priority 2):** Corporate legal designator stripping followed by alphabetical word-token sorting (captures word transposition e.g., "Apple Retail Inc" $\leftrightarrow$ "Retail Apple").
3. **Channel 3 — Standardized Exact Address (Priority 3):** Standardized street, city, and numeric address string matching.
4. **Channel 4 — Street Prefix & Numeric Anchor (Priority 4):** Combines the first 3 street tokens with the extracted numeric house number.
5. **Channel 5 — Informative Rare Brand Token (Priority 5):** Inverted index over discriminative brand tokens with document frequency $\le 300$ across the corpus.
6. **Channel 6 — Consecutive Name Bigrams (Priority 6):** 2-word sliding name bigrams with document frequency $\le 500$.
7. **Channel 7 — Sorted Address Token Set (Priority 7):** Order-invariant token set intersection across address components.

### 3.2 Candidate Budget & Reduction Metrics
* **Total Raw Candidate Pool:** 5,633,977 pairs retrieved on the 50k validation set (**81.56% unbudgeted link recall**, **96.75% entity coverage**).
* **Budgeted Ranking ($K=40$):** Candidates are ranked by channel priority and token specificity, taking at most 40 candidates per Source 1 entity.
* **Reduction Ratio:** Cuts the search space from $1.73 \times 10^6 \times 7.6 \times 10^6 \approx 1.3 \times 10^{13}$ potential pairs down to an average of **$29.1$ candidate pairs per entity** — achieving a **$>99.9997\%$ reduction ratio** while capturing **$69.33\%$ budgeted link recall**.

---

## 4. Matching Model & Feature Engineering

### 4.1 Feature Engineering (44 Relational Features)
We engineered 44 orthogonal features designed to separate true entity linkages from challenging hard negatives:
1. **Lexical & Fuzzy Similarity (12 features):** Jaro-Winkler distance, Levenshtein distance, token sort ratio, token set ratio, partial token ratio, character 2-gram / 3-gram / 4-gram Jaccard similarity, word token Jaccard similarity, and token overlap coefficients.
2. **Spatial & Address Metrics (6 features):** Address Jaro-Winkler, address token sort/set similarity, character 3-gram address Jaccard, address token overlap, and target address null indicators.
3. **Numeric & Structural Integrity (7 features):** Exact house number equality (`hno_match`), postal code (PIN) exact match, PIN prefix-3 match, PIN presence flags, house number contradiction indicator (`hno_contradiction`), and first-word equality (`fw_match`).
4. **Length Ratios & Cross-Domain Interactions (14 features):** String length absolute differences and ratios across name and address, word count discrepancies, prefix length matching, and multiplicative interaction terms (`name_jw * addr_jw`, `name_sim * pin_match`).
5. **Categorical & Domain Indicators (5 features):** Source origin flag (`is_source3`), India geographic flag, France geographic flag, and channel origin priority ranks.

### 4.2 Model Architecture & GPU Meta-Ensemble
To minimize prediction variance and capture non-linear feature interactions, we deployed a 3-way heterogeneous gradient-boosted ensemble:
* **LightGBM Pointwise Classifier:** 300 trees, leaf-wise Best-First growth, `num_leaves=31`, `lr=0.05`. Achieved individual validation AUC of `0.99931`.
* **XGBoost GPU Classifier:** 300 trees, depth-wise tree growth, `max_depth=6`, GPU histogram method (`tree_method="hist"`). Achieved individual validation AUC of `0.99930`.
* **CatBoost GPU Classifier:** 500 trees, symmetric oblivious tree structure, depth 6, robust handling of sparse numeric flags. Achieved validation AUC of `0.99911`.
* **Meta-Stacker (EXP-16):** Out-of-fold predictions from all three models are integrated via an $L_2$-regularized Logistic Regression meta-learner, producing calibrated posterior probabilities $P(\text{Match} \mid x)$.

### 4.3 Breakthrough Post-Processing: Physical Mutual Exclusivity
A forensic audit of the 7,638,365 training ground-truth links revealed a fundamental physical invariance:
$$\text{Targets associated with }>1\text{ Source 1 entity} = \mathbf{0\ (0.000\%)}$$
Because each Source 2/3 record represents a unique real-world establishment, **no target can simultaneously belong to two different Source 1 businesses**. 

In naive predictions, candidate generation shards caused 12,594 target collision links where unmatched singletons claimed targets already securely matched to reference entities. In `v21`, we enforced **Strict Physical Mutual Exclusivity**:
1. Any target claimed by a high-precision reference match ($v7$) is permanently locked; singletons are strictly forbidden from claiming it.
2. If two singletons compete for an unclaimed target, it is awarded exclusively to the query with the highest posterior probability $\arg\max_q P(\text{Match}_{q,t})$.
3. This single intervention eradicated **12,594 guaranteed false-merge edges**, elevating the final leaderboard score from `0.79918` directly to **`0.80539`**.

---

## 5. Results & Validation Progression

### 5.1 Submission & Validation History

| Iteration | Pipeline Description | Key Innovation | Public $F_{0.5}$ |
|:---|:---|:---|:---:|
| EXP-01 | Baseline Heuristics | Exact match + veto rules | 0.5937 |
| EXP-03 | Standalone LightGBM | 44 features, global cutoff $\theta = 0.55$ | 0.7785 |
| EXP-11 | Tri-Model Ensemble | LightGBM + XGBoost + CatBoost blend | 0.7880 |
| Submission v7 | 7-Channel Fast Retrieval + Ensemble | High-precision reference matching | 0.7880 |
| Submission v12 | v7 + Singleton ML Boost ($\theta \ge 0.60$) | First singleton recall recovery (+109k queries) | 0.7990 |
| Submission v18 | Tiered Boundary ($\theta_1 = 0.55, \theta_2 = 0.65$) | Precision-safe singleton expansion | 0.79918 |
| **Submission v21 (Champion)** | **Tiered + Strict Physical Mutual Exclusivity** | **Pruned 12,594 false collisions with v7** | **`0.80539`** 🏆 |

### 5.2 Error Analysis & Boundary Cases
* **False Merges (False Positives):** Primarily occur among nationwide retail chains (e.g., pharmacy chains, fast-food outlets) sharing identical business names, identical city centers, and differing only in subtle suite/door numbers.
* **False Rejections (False Negatives):** Primarily stem from heavy transliteration distortion on Indian records (phonetic Devanagari spellings without English phonetic anchors) and truncated records lacking both house numbers and road identifiers.

---

## 6. Conclusion

Team X's solution proves that conquering large-scale Entity Resolution under precision-asymmetric metrics ($F_{0.5}$) requires equal mastery of **algorithmic scalability, statistical modeling, and domain physics**:
1. **Scalability:** The Polars 7-channel priority retrieval system successfully compressed $1.3 \times 10^{13}$ potential pairs into a lean, actionable candidate budget with $>99.999\%$ search-space reduction.
2. **Statistical Depth:** A 44-feature GPU ensemble (LightGBM, XGBoost, CatBoost) achieved near-perfect discrimination (AUC $> 0.999$).
3. **Physical Exactness:** Enforcing strict physical mutual exclusivity eliminated 12,594 false merges, propelling our final official score to **`0.80539` Macro $F_{0.5}$**.

The complete, end-to-end pipeline is fully self-contained, reproducible, and compliant with all competition rules and open-source licensing constraints.

---
*Signed by:*  
**Team X — Subhankar Swain, Anurag Swain, Pradyumna Kumar Biswal, Jahanabi Dalai**  
*Government College of Engineering, Kalahandi*

