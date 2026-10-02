# Phase 2 — Technical Research & Literature Log
## Amazon ML Challenge 2026 — Business Entity Resolution

> **Role:** Lead Entity Resolution Researcher & ML Scientist  
> **Phase:** Phase 2 (Matching, Scoring, Ranking, Multi-Source Representation & Architecture Discovery)  
> **Status:** Active & Continuously Updated  
> **Operating Mandate:** Empirical verification of all proposed methodologies against the competition dataset and resource budget.

---

## 1. Research Protocol & Evaluation Criteria

Every external method, paper, and technical capability investigated is cataloged under the following strict schema:
- **Technique Name & Domain:** Algorithmic category and primary mechanism.
- **Source & Reference:** Citation (arXiv, ACL, ACM, IEEE, GitHub, or official documentation).
- **Core Principle:** Mathematical formulation or underlying inductive bias.
- **Dataset Applicability:** Evaluation against dataset specifics (10.3M targets, 2.2M queries, multi-match ground truth, 38.3% name collisions, missing addresses, Devanagari script).
- **Computational Profile:** Memory overhead, latency, hardware requirements (RTX 3050 6GB VRAM, 16GB RAM, i7-13700H).
- **Compliance Status:** Licensing (MIT / Apache 2.0 / Permissive open-weight) and rule compliance (no external web scraping, no external geocoding API).
- **Empirical Status:** `PLANNED` | `PROTOTYPED` | `BENCHMARKED` | `ACCEPTED` | `REJECTED` | `DEFERRED`.

---

## 2. Research Log Entries

### Entry 01: Fast Character N-Gram & Jaro-Winkler String Distance Ensembles
- **Category:** Pairwise String Intelligence & Lexical Scoring
- **Primary References:**
  - Winkler, W. E. (1990). *String Comparator Metrics and Enhanced Decision Rules in the Fellegi-Sunter Model of Record Linkage.* Proceedings of the Section on Survey Research Methods.
  - RapidFuzz Documentation (Max Bachmann, 2024): SIMD-accelerated C++ implementations of Levenshtein, Damerau-Levenshtein, Jaro-Winkler, and token-based ratios.
- **Core Principle:** Levenshtein and Jaro-Winkler capture typographical errors and prefix preservation (critical for business names where suffixes like "Inc", "Pvt Ltd" are removed but root stems remain). Character n-gram cosine similarities capture intra-word transpositions and OCR/scraping noise.
- **Dataset Applicability:** High. S2 and S3 contain severe OCR/token transpositions and OCR noise in Indian and US business names.
- **Compliance & Feasibility:** MIT License. SIMD-vectorized execution allows scoring 100,000 candidate pairs in <0.2 seconds.
- **Status:** `ACCEPTED` — Core pairwise feature family.

### Entry 02: Query-Grouped Learning-to-Rank (LambdaMART / LightGBM Lambdarank)
- **Category:** Learning-to-Rank (LTR) Architecture
- **Primary References:**
  - Burges, C. J. (2010). *From RankNet to LambdaRank to LambdaMART: An Overview.* Microsoft Research Technical Report.
  - Ke, G., et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* NeurIPS 2017.
- **Core Principle:** Standard binary classification treats candidate pairs independently, ignoring intra-query competition. LambdaMART computes gradients proportional to NDCG / MRR gains when swapping candidates within the same S1 query group.
- **Dataset Applicability:** High. S1 queries receive variable candidate lists ($K \in [1, 40]$). In multi-match entity resolution, ranking true positives above hard-negative collisions directly maximizes precision at top-K.
- **Compliance & Feasibility:** MIT License. Highly optimized in LightGBM and XGBoost.
- **Status:** `BENCHMARKED & REJECTED` — EXP-04 (LightGBM LambdaRank: 0.7711) and EXP-05 (XGBoost rank:ndcg: 0.7647) both lost to Pointwise binary classification (0.7786). In competitive entity resolution with multi-match cardinality, listwise ranking distorts global threshold calibration.

### Entry 03: Fellegi-Sunter Probabilistic Record Linkage & Log-Likelihood Ratio Scoring
- **Category:** Classical Statistical Record Linkage
- **Primary References:**
  - Fellegi, I. P., & Sunter, A. B. (1969). *A Theory for Record Linkage.* Journal of the American Statistical Association, 64(328), 1183–1210.
  - Splink (UK Ministry of Justice, 2023): Scalable, unsupervised Fellegi-Sunter implementation using Expectation-Maximization.
- **Core Principle:** Computes agreement weights $w_i = \log_2(m_i / u_i)$ where $m_i = P(\text{agree}_i | \text{match})$ and $u_i = P(\text{agree}_i | \text{non-match})$. Agreement on rare tokens (e.g. rare surname or PIN code) produces massive positive weight, while agreement on frequent tokens ("Enterprises", "New York") yields negligible evidence.
- **Dataset Applicability:** High. 38.31% of S1 queries share business names. A match on "State Bank of India" requires strong address evidence, whereas a match on an idiosyncratic brand name is almost decisive.
- **Compliance & Feasibility:** Mathematical formulation implemented in Polars/NumPy; zero external dependency.
- **Status:** `PLANNED` — Essential baseline and feature generator for GBDT.

### Entry 04: Multilingual Dense Encoders & Cross-Lingual Embedding Retrieval
- **Category:** Neural Dense Retrieval & Semantic Similarity
- **Primary References:**
  - Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* EMNLP 2019.
  - Wang, L., et al. (2022). *Text Embeddings by Weakly-Supervised Contrastive Pre-training (E5).* arXiv:2212.03533.
  - Xiao, S., et al. (2023). *C-Pack: Packaged Resources to Advance General Chinese and Multilingual Information Retrieval (BGE-M3).* arXiv:2309.07597.
- **Core Principle:** Projects business names and addresses into a dense metric space where semantically equivalent representations (e.g., cross-script Devanagari vs Latin, abbreviations, spelling variants) map to high cosine similarity.
- **Dataset Applicability:** High for Indian records with Devanagari script in S2/S3 (13.37% and 7.53% respectively) and phonetic/vernacular name variations.
- **Resource Feasibility:** VRAM budget: RTX 3050 6GB. Models must be $\le 1$B parameters (e.g., `paraphrase-multilingual-MiniLM-L12-v2`, `multilingual-e5-small`, `bge-small-en-v1.5`).
- **Compliance & Feasibility:** Apache 2.0 / MIT.
- **Status:** `PLANNED` — Diagnostic benchmark on cross-script candidate recovery and dense re-ranking.

### Entry 05: Approximate Nearest Neighbor (ANN) Indexing (Faiss / HNSW)
- **Category:** Vector Indexing & Sub-Linear Retrieval
- **Primary References:**
  - Johnson, J., Douze, M., & Jégou, H. (2019). *Billion-scale similarity search with GPUs.* IEEE Transactions on Big Data.
  - Malkov, Y. A., & Yashunin, D. A. (2018). *Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs.* IEEE TPAMI.
- **Core Principle:** Builds a graph-based (HNSW) or inverted-file (IVF-PQ) index over vector embeddings to query top-K nearest neighbors in logarithmic time rather than $O(N)$ exhaustive dot products.
- **Dataset Applicability:** 10.3M target pool records. Exhaustive search requires 10.3M dot products per query; ANN enables real-time vector candidate retrieval.
- **Compliance & Feasibility:** MIT License. CPU / GPU support via `faiss-cpu`.
- **Status:** `PLANNED` — To be evaluated against deterministic blocking for recall-per-second and memory footprint.

### Entry 06: Multi-Source Entity Profiling & Relational Consensus (S2 $\leftrightarrow$ S3)
- **Category:** Relational & Collective Entity Resolution
- **Primary References:**
  - Bhattacharya, I., & Getoor, L. (2007). *Collective Entity Resolution in Relational Data.* ACM Transactions on Knowledge Discovery from Data (TKDD).
  - Dong, X. L., et al. (2014). *Knowledge Vault: A Web-Scale Approach to Probabilistic Knowledge Fusion.* KDD 2014.
- **Core Principle:** Target records in S2 and S3 frequently reference the exact same real-world business entity. If an S1 query retrieves an S2 candidate $A$ and an S3 candidate $B$, and $A$ and $B$ are mutually similar, the cross-source consensus significantly increases the posterior probability of a true match.
- **Dataset Applicability:** Extreme. S2 has 5.03M rows and S3 has 5.29M rows. In training ground truth, 89.02% of S1 entities link to multiple targets across both S2 and S3.
- **Compliance & Feasibility:** Relational feature generation within candidate sets; zero external lookups.
- **Status:** `PLANNED` — Novel architecture design for Phase 2.

### Entry 07: Selective Prediction & Threshold Abstention for Precision-Heavy Evaluation ($F_{0.5}$)
- **Category:** Decision Calibration & Abstention Modeling
- **Primary References:**
  - Geifman, Y., & El-Yaniv, R. (2017). *Selective Classification for Deep Neural Networks.* NeurIPS 2017.
  - Niculescu-Mizil, A., & Caruana, R. (2005). *Predicting Good Probabilities With Supervised Learning.* ICML 2005.
- **Core Principle:** The competition evaluation metric is macro $F_{0.5}$ (precision weighted twice as heavily as recall: $\beta = 0.5$). False positives are severely penalized:
  $$\text{Macro } F_{0.5} = \frac{1}{|S1|} \sum_{q \in S1} \frac{1.25 \cdot P_q \cdot R_q}{0.25 \cdot P_q + R_q}$$
  Abstaining on uncertain or low-margin candidate pairs prevents fatal precision collapses.
- **Dataset Applicability:** 5.585% of train S1 queries have ZERO true matches in S2/S3. Forcing predictions on these queries guarantees precision = 0. Calibrated abstention is mathematically necessary.
- **Status:** `ACCEPTED & VERIFIED` — EXP-03 and EXP-07 confirm that optimal global and country thresholds fall in $[0.55, 0.60]$, protecting Singleton Accuracy at $>90.9\%$ while sustaining Macro F0.5 = 0.7786.

### Entry 08: Feature Dimensionality & Feature Dilution Pathology in GBDTs
- **Category:** Representation Engineering & Tree Regularization
- **Primary References:**
  - Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning.* Chapter 10: Boosting and Additive Trees.
  - Prokhorenkova, L., et al. (2018). *CatBoost: unbiased boosting with categorical features.* NeurIPS 2018.
- **Core Principle:** Adding correlated or noisy weak features dilutes tree split choices, especially when a single dominant feature (e.g., `name_x_addr`) carries the majority of mutual information.
- **Empirical Evidence:** In EXP-06, expanding from 24 to 40 features reduced the primary feature's gain from 57.93% to 5.25%, degrading Macro F0.5 from 0.7786 to 0.7723. In EXP-08 ablation, removing G4/G5/G8 (address flags, postal) had zero negative impact, proving compact, uncorrupted feature sets are superior.
- **Status:** `ACCEPTED` — Core architectural rule: preserve compact, high-SNR feature sets.

### Entry 09: Cost-Sensitive Loss Reweighting vs. Threshold Tuning for Class Imbalance
- **Category:** Class Imbalance & Loss Formulation
- **Primary References:**
  - Elkan, C. (2001). *The Foundations of Cost-Sensitive Learning.* IJCAI 2001.
  - He, H., & Garcia, E. A. (2009). *Learning from Imbalanced Data.* IEEE TKDE.
- **Core Principle:** Modifying loss instance weights changes the posterior probability calibration, effectively shifting the uncalibrated output distribution. In multi-match entity resolution with singletons, artificial loss weighting shifts optimal thresholds into extreme regions ($>0.80$), causing catastrophic precision collapse on zero-match queries.
- **Empirical Evidence:** In EXP-09, applying 2x weight to hard negatives caused Singleton Accuracy to collapse from 91.3% down to 75.8% and Macro F0.5 to drop to 0.7592 (-0.0194 vs champion).
- **Status:** `REJECTED` — Hard negative weighting is harmful; precision must be enforced via calibrated decision thresholds and explicit veto features.
