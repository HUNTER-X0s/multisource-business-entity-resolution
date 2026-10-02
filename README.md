# Amazon ML Challenge 2026: Multi-Source Business Entity Resolution
## Team X — Official Production Pipeline & Reproduction Guide

> **Team Name:** Team X  
> **Institution:** Government College of Engineering, Kalahandi  
> **Team Members:** Subhankar Swain, Anurag Swain, Pradyumna Kumar Biswal, Jahanabi Dalai  
> **Official Public Leaderboard Score:** **`0.80539` Macro $F_{0.5}$** (Evaluated Rank: 4245)  
> **Primary Objective:** Precision-Heavy Macro $F_{0.5}$ Entity Resolution across 1,732,544 test queries.

---

## 1. Pipeline Architecture Overview

The production pipeline is organized into three decoupled, high-performance stages designed to scale to billions of pairwise comparisons while strictly enforcing the $F_{0.5}$ metric:

```
[dataset/test/ (S1, S2, S3)]
            │
            ▼
[Stage 1: 7-Channel Polars Priority Union Retrieval]
  ├── Candidate generation via token, address, bigram & rare-brand channels
  └── Output: output/candidate_pairs.tsv (K <= 40 per entity, >99.999% reduction)
            │
            ▼
[Stage 2: 44-Dimensional Pairwise Feature Extraction & GPU Inference]
  ├── Relational feature matrix (lexical, geographic, numeric integrity, interaction terms)
  ├── GPU Tri-Model Meta-Ensemble (LightGBM + XGBoost + CatBoost)
  └── Out-of-Fold L2 Logistic Regression Probability Stacker
            │
            ▼
[Stage 3: Decision Calibration & Physical Mutual Exclusivity]
  ├── Tiered Decision Boundaries (theta_1 = 0.55, theta_2 = 0.65)
  ├── Strict Physical Mutual Exclusivity (veto of 12,594 multi-parent target collisions)
  └── Output: output/matching_results.tsv (Score: 0.80539)
```

---

## 2. Directory Layout & Deliverable Inventory

```text
Team X_submission/
├── output/
│   ├── matching_results.tsv           # Final entity matches (Score: 0.80539 Macro F0.5)
│   └── candidate_pairs.tsv            # Final blocking candidate set (K <= 40)
├── code/
│   └── business_entity_resolution/
│       ├── README.md                  # This end-to-end reproduction guide
│       ├── requirements.txt           # Pinned environment dependencies
│       └── src/
│           ├── candidate_generation.py    # 7-channel Polars retrieval engine
│           ├── candidate_evaluator.py     # Recall ceiling & reduction ratio auditor
│           ├── feature_engineering.py     # 44-feature relational extractor
│           ├── fast_10ch_retrieval.py     # Optimized multi-threaded retrieval engine
│           ├── vectorized_pipeline.py     # Production streaming inference coordinator
│           ├── ensemble_pointwise.py      # Meta-ensemble inference coordinator
│           ├── lightgbm_pointwise.py      # LightGBM classifier module
│           ├── evaluate.py                # Official Macro F0.5 evaluation metric
│           ├── schema_validation.py       # Data integrity & column contract checker
│           ├── validate_submission.py     # Official competition submission validator
│           └── models/
│               ├── meta_lgb.pkl           # Trained LightGBM Pointwise Model (300 trees)
│               ├── meta_xgb.pkl           # Trained XGBoost GPU Model (300 trees)
│               ├── meta_cb.pkl            # Trained CatBoost GPU Model (500 trees)
│               ├── meta_stacker.pkl       # Trained L2 Logistic Regression Stacker
│               ├── gpu_model_zoo.py       # Multi-architecture loader & harness
│               └── baseline_heuristic.py  # Deterministic reference baseline
└── Documentation_template.md  # Comprehensive technical methodology report (filled-in template)
```

---

## 3. Environment Setup & Hardware Prerequisites

### 3.1 Hardware Recommendations
* **RAM:** 16 GB minimum (32 GB recommended for full parallel test streaming).
* **GPU:** NVIDIA CUDA-compatible GPU (e.g., RTX 3050 6GB / T4 / V100 / A100) recommended for rapid ensemble scoring. CPU fallback is fully supported via standard Scikit-learn/LightGBM CPU backends.
* **OS:** Cross-platform (Windows PowerShell / Linux Ubuntu 22.04+ / macOS).

### 3.2 Installation Steps
Create an isolated Python 3.10+ virtual environment and install the exact pinned versions:

```bash
# 1. Create and activate virtual environment
python -m venv venv

# On Linux/macOS:
source venv/bin/activate

# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# 2. Upgrade pip and install pinned dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. End-to-End Execution Instructions

All commands are executed from the `code/business_entity_resolution/` directory:

```bash
cd code/business_entity_resolution
```

### Step 1: Candidate Generation (Blocking)
Run the 7-channel prioritized Polars blocking engine over the raw test files:
```bash
python src/candidate_generation.py \
    --test-dir ../../dataset/test \
    --output-dir ../../output \
    --max-candidates 40
```
* **Output:** Generates `output/candidate_pairs.tsv` ($K \le 40$ candidates per Source 1 entity).
* **Guarantees:** Captures $>99.999\%$ search space reduction while retaining link recall.

### Step 2: Feature Engineering & GPU Ensemble Scoring
Extract the 44 pairwise relational features and score candidate pairs using the trained Tri-Model Meta-Ensemble:
```bash
python src/vectorized_pipeline.py \
    --test-dir ../../dataset/test \
    --output-dir ../../output \
    --models-dir src/models \
    --batch-size 100000
```
* **Output:** Generates pre-calibrated candidate scores.

### Step 3: Decision Calibration & Physical Mutual Exclusivity
Apply the Tiered Decision Boundary and Strict Mutual Exclusivity filter to generate the final matches:
```bash
python src/ensemble_pointwise.py \
    --test-dir ../../dataset/test \
    --output-dir ../../output \
    --t1 0.55 \
    --t2 0.65 \
    --enforce-mutual-exclusivity
```
* **Output:** Produces the final, verified `output/matching_results.tsv` (Leaderboard Score: **`0.80539`**).

---

## 5. Verification & Submission Validation

Verify that your output files pass every syntactic, structural, and semantic rule enforced by the competition portal:

```bash
python src/validate_submission.py \
    --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv \
    --test-dir ../../dataset/test
```

### Expected Output:
```text
ML Challenge 2026 — Submission Validator
  test dir: ../../dataset/test
  required S1 entities: 1732544
  matching_results.tsv: 1732544 rows (186008 empty, 1546536 non-empty).
  candidate_pairs.tsv: 1732544 rows (634 empty, 1731910 non-empty).

PASS — no blocking issues found. Safe to submit.
```

---

## 6. Model Licensing & Compliance

* **Base Architectures:** LightGBM (MIT License), XGBoost (Apache 2.0 License), CatBoost (Apache 2.0 License), Scikit-Learn (BSD 3-Clause License).
* **Model Parameters:** Combined ensemble size $< 50\text{ Million}$ parameters (strictly within the competition 8 Billion parameter ceiling).
* **Fair Play & Integrity:** Zero external data lookups, zero external APIs, zero pre-trained closed commercial entity databases. Fully trained from provided data only.

---
*Maintained by Team X | Government College of Engineering, Kalahandi | September 2026*

