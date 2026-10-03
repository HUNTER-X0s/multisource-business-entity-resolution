# Multi-Source Business Entity Resolution (Amazon ML Challenge 2026)
### Enterprise-Scale Entity Resolution Across Heterogeneous Databases under Asymmetric Macro $F_{0.5}$

[![Official Score](https://img.shields.io/badge/Official%20Leaderboard%20Score-0.80539%20Macro%20F0.5-brightgreen?style=for-the-badge&logo=amazon)](https://github.com/HUNTER-X0s/multisource-business-entity-resolution)
[![Candidate Recall](https://img.shields.io/badge/Candidate%20Retrieval%20Recall-97.4%25-blue?style=for-the-badge)](https://github.com/HUNTER-X0s/multisource-business-entity-resolution)
[![Search Space Pruning](https://img.shields.io/badge/Search%20Space%20Reduction->99.999%25-orange?style=for-the-badge)](https://github.com/HUNTER-X0s/multisource-business-entity-resolution)
[![Mutual Exclusivity](https://img.shields.io/badge/Collisions%20Eliminated-12,594%20Edges-purple?style=for-the-badge)](https://github.com/HUNTER-X0s/multisource-business-entity-resolution)
[![Python Version](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)](https://github.com/HUNTER-X0s/multisource-business-entity-resolution)
[![Hardware Acceleration](https://img.shields.io/badge/Inference-NVIDIA%20CUDA%20GPU-green?style=for-the-badge&logo=nvidia)](https://github.com/HUNTER-X0s/multisource-business-entity-resolution)

---

> **Team Name:** Team X  
> **Institution:** Government College of Engineering, Kalahandi (GCEK)  
> **Team Members:** Subhankar Swain, Anurag Swain, Pradyumna Kumar Biswal, Jahanabi Dalai  
> **Official Evaluated Leaderboard Metric:** **`0.80539` Macro $F_{0.5}$** (Evaluated Rank: 4245 / 1,732,544 Test Queries)  
> **Dataset Scale:** 1,732,544 Source 1 entities resolved against multi-million Source 2 & Source 3 records across the US, India, and France.

---

## 🏆 Key Engineering & Benchmark Highlights

- ⚡ **97.4% Candidate Retrieval Recall:** Captured 97.4% of all ground-truth matches within a strict, competition-compliant blocking budget of $K \le 40$ candidates per entity.
- 🚀 **>99.999% Search Space Pruning:** Slashed the combinatorial $O(N^2)$ search space from **$3 \times 10^{12}$ pairwise comparisons** down to $\le 40$ candidates per entity using a high-throughput **7-Channel Polars Priority Union Retrieval Engine**.
- 🎯 **Official Evaluated Score of `0.80539` Macro $F_{0.5}$:** Surpassed the competitive `0.80` barrier on 1,732,544 test queries under a metric that heavily penalizes false merges (precision weighted 4× higher than recall), including zero-shot generalization to France with 0 training labels.
- 🛡️ **Mathematical Physical Mutual Exclusivity:** Forensic analysis proved that real-world business targets exhibit a **0.000% multi-parent rate** in training. Enforcing our novel Mutual Exclusivity Layer pruned **12,594 guaranteed false-merge collisions**, driving the score directly from `0.79918` $\rightarrow$ **`0.80539`**.
- 🧠 **44-Feature GPU Tri-Model Meta-Ensemble:** Relational feature space combining lexical distance, phonetic double metaphone, token overlap, numeric ratios, and cross-source interactions trained across **LightGBM, XGBoost, and CatBoost** with out-of-fold L2 regularized logistic probability stacking.
- 🔒 **100% Rules Compliant:** Zero external lookups, zero API cheating, zero data leakage, and total model parameters well below the 8B limit (pure GBDT + stacking ensemble).

---

## 1. System Architecture

The end-to-end resolution pipeline is organized into three decoupled, high-performance stages designed to scale to billions of pairwise comparisons:

```
                  [ Multi-Source Data: S1, S2, S3 ]
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 1: 7-Channel Polars Priority Union Retrieval Engine       │
│ • Exact normalized token inverted index                         │
│ • Normalized business address inverted index                    │
│ • Character 4-gram & word bigram fuzzy indices                  │
│ • Transliteration & rare brand phonetic channels                │
│ • Priority union aggregation with hard truncation (K <= 40)     │
│ ➔ Pruning Efficiency: >99.999% (Output: candidate_pairs.tsv)    │
│ ➔ Candidate Retrieval Recall: 97.4%                             │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 2: 44-Dimensional GPU Relational Feature Engine & Models  │
│ • Lexical: Jaro-Winkler, Levenshtein, Token Sort / Set Ratio    │
│ • N-gram: Character 3-gram, 4-gram & word Jaccard similarities  │
│ • Geographic / Address: Token intersection, numeric consistency │
│ • Model Ensemble: LightGBM (Histogram GPU) + XGBoost + CatBoost │
│ • Meta-Learner: Out-of-Fold L2 Logistic Regression Stacker      │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 3: Decision Calibration & Strict Mutual Exclusivity       │
│ • Tiered Decision Boundaries (θ1 = 0.55 for primary matches)   │
│ • Conservative Secondary Threshold (θ2 = 0.65 for multi-matches)│
│ • Strict Physical Mutual Exclusivity: Vetoes target collisions  │
│   (Pruned 12,594 false merges; 0.000% multi-parent invariant)   │
│ ➔ Final Output: matching_results.tsv (0.80539 Macro F0.5)       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Quantitative Performance & Progression

| Iteration | Retrieval Channels | Model Architecture | Post-Processing | Macro $F_{0.5}$ | Rank | Key Insight |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- |
| **v1 Baseline** | Single-token index | Jaro-Winkler heuristic | Greedy match | `0.65210` | 9200+ | High false positive rate on generic words |
| **v5 LightGBM** | 5-channel union | Single LightGBM (24 feat) | Flat $\theta = 0.50$ | `0.74820` | 7100+ | Learned weights for brand vs address tokens |
| **v8 Retrieval** | 7-channel Priority | GPU LightGBM (32 feat) | Flat $\theta = 0.50$ | `0.77840` | 5800+ | Recall ceiling expanded to >97% |
| **v12 Stacking** | 7-channel Priority | Tri-Model GPU Ensemble | Stratified $\theta$ | `0.79240` | 4900+ | XGB + LGB + CatBoost out-of-fold stacking |
| **v16 Tuned** | 7-channel Priority | Tri-Model + 44 Features | Tiered ($\theta_1=0.55, \theta_2=0.65$) | `0.79918` | 4400+ | Suppressed noisy secondary false positives |
| **v21 Champion** | **7-Channel Priority** | **Tri-Model Meta-Ensemble** | **Strict Mutual Exclusivity** | **`0.80539`** | **4245** | **Vetoed 12,594 multi-parent target collisions** |

---

## 3. Core Technical Innovations

### 3.1 Strict Physical Mutual Exclusivity Layer (The Breakthrough)
In multi-source commercial data, each record in Source 2 or Source 3 represents a physical business entity that can belong to **at most one** true real-world enterprise.
- **Data Forensic Discovery:** Auditing 7.6M training links revealed that targets linked to $>1$ Source 1 entity is **strictly 0.000%**.
- **The Problem:** Independent pointwise ranking allowed multiple Source 1 entities to claim the same high-confidence target. In earlier runs, singletons claimed 12,594 targets that were already claimed with higher confidence by primary entities, collapsing precision.
- **The Solution:** We implemented an optimal bipartite assignment conflict-resolution filter. If target $T$ was claimed by entity $A$ (confidence 0.94) and entity $B$ (confidence 0.58), entity $B$'s edge was revoked. This single constraint eliminated **12,594 false edges**, lifting the score directly across the 0.80 barrier to **`0.80539`**.

### 3.2 7-Channel Polars Priority Union Blocking
Computing all pairwise combinations across 1.73M queries and millions of targets requires $>3 \times 10^{12}$ comparisons. Our Polars-based inverted index generates $\le 40$ candidates per entity in seconds:
1. `exact_token`: Exact match on normalized alphanumeric tokens.
2. `address_token`: Exact match on normalized street and postal tokens.
3. `char_4gram`: Character 4-gram inverted index for handling typos and OCR noise.
4. `word_bigram`: Co-occurring word bigram index for multi-word brands.
5. `transliteration`: Phonetic hash matching (Double Metaphone) for Indic/French names.
6. `rare_brand`: Inverted index on low-frequency tokens ($IDF > 0.85$).
7. `postal_spatial`: Co-location blocking on postal code prefixes.

---

## 4. Repository Structure

```text
multisource-business-entity-resolution/
├── README.md                                       # Comprehensive project overview & reproduction guide
├── Documentation.md                                # Full 12KB technical methodology report
├── Problem Statement.txt                           # Official competition problem description
├── Team X_submission.zip                           # Complete competition-ready submission archive (561 MB)
│
├── Team X_submission/                              # Clean submission bundle
│   ├── Documentation.md                            # Methodology report accredited to GCEK team
│   ├── output/
│   │   ├── candidate_pairs.tsv                     # K <= 40 blocking candidate set (1.30 GB)
│   │   └── matching_results.tsv                    # Final evaluated 0.80539 matches (79.7 MB)
│   └── code/business_entity_resolution/
│       ├── README.md                               # Pipeline reproduction instructions
│       ├── requirements.txt                        # Pinned dependencies
│       └── src/
│           ├── candidate_generation.py             # 7-channel blocking engine
│           ├── feature_engineering.py              # 44 relational features
│           ├── ensemble_pointwise.py               # GPU meta-ensemble & inference
│           ├── validate_submission.py              # Format compliance validator
│           └── models/
│               ├── meta_lgb.pkl                    # Trained LightGBM weights
│               ├── meta_xgb.pkl                    # Trained XGBoost weights
│               ├── meta_cb.pkl                     # Trained CatBoost weights
│               └── meta_stacker.pkl                # Trained L2 Logistic Stacker weights
│
└── Amazon ml challenge/                            # Full research and engineering workspace
    ├── dataset/raw/                                # Full raw datasets (train & test TSVs)
    ├── docs/                                       # 33 comprehensive research and audit logs
    ├── research_scripts/                           # Forensic analysis and integrity scripts
    ├── experiments/                                # Phase 2 experimental benchmarks & models
    ├── output/                                     # Milestone prediction files (v8 through v21)
    └── src/                                        # Modularized production source scripts
```

---

## 5. Quickstart & Reproduction Guide

### Environment Setup
```bash
# Clone the repository
git clone https://github.com/HUNTER-X0s/multisource-business-entity-resolution.git
cd multisource-business-entity-resolution

# Create virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate # Linux / macOS

# Install pinned dependencies
pip install -r "Team X_submission/code/business_entity_resolution/requirements.txt"
```

### End-to-End Pipeline Execution
```bash
# Step 1: Generate K <= 40 blocking candidates (>99.999% search reduction)
python "Team X_submission/code/business_entity_resolution/src/candidate_generation.py"

# Step 2: Extract 44 features and run GPU Meta-Ensemble inference
python "Team X_submission/code/business_entity_resolution/src/ensemble_pointwise.py"

# Step 3: Validate submission format against official competition rules
python "Team X_submission/code/business_entity_resolution/src/validate_submission.py" \
    --matching "Team X_submission/output/matching_results.tsv" \
    --candidate "Team X_submission/output/candidate_pairs.tsv" \
    --test-dir "Amazon ml challenge/dataset/raw/test"
```

---

## 6. Academic Integrity & Fair-Play Statement
This solution was developed in strict compliance with the Amazon ML Challenge 2026 guidelines:
- **No External Lookups:** Zero use of external entity resolution APIs, government databases, geocoders, or internet queries.
- **Model Size Compliance:** Pure GBDT meta-ensemble (LightGBM, XGBoost, CatBoost) totaling <50MB in weights, well below the 8-Billion parameter ceiling.
- **Fully Deterministic & Reproducible:** End-to-end execution without random seed variation or target data leakage.
