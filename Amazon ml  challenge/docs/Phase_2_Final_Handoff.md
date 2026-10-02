# Phase 2 — Final Engineering Handoff & Phase 3 Roadmap
## Amazon ML Challenge 2026 — Multi-Source Business Entity Resolution

> **Role:** Lead Entity Resolution Researcher & ML Scientist  
> **Status:** Production-Ready & Reproducibility Certified  
> **Handoff Target:** Phase 3 (End-to-End Pipeline Scaling & Leaderboard Submission Engineering)  
> **Date:** 2026-09-26

---

## 1. Production Artifacts & Champion Specification

### 1.1 Champion Architecture
- **Model:** Pointwise Convex Ensemble ([EXP-11](file:///z:/Amazon%20ML/experiments/phase2/models/ensemble_pointwise.py))
  $$\hat{P} = 0.50 \cdot P_{\text{LightGBM}} + 0.50 \cdot P_{\text{XGBoost}}$$
- **LightGBM Specification:** `n_estimators=300, num_leaves=31, learning_rate=0.05, objective="binary", random_state=2026`
- **XGBoost Specification:** `n_estimators=300, max_depth=6, learning_rate=0.05, tree_method="hist", objective="binary:logistic", random_state=2026`
- **Optimal Decision Threshold:** $\theta^* = 0.60$ (Global & Country-Invariant)
- **Feature Set:** Compact 24 Pairwise Features (Specified in [Phase_2_Feature_Catalog.md](file:///z:/Amazon%20ML/docs/Phase_2_Feature_Catalog.md))
- **Performance:** Macro $F_{0.5} = \mathbf{0.7733} \dots \mathbf{0.7786}$, Singleton Accuracy = $\mathbf{91.13\%}$, Link Precision = $\mathbf{96.75\%}$.

### 1.2 Core File Map
| Component | Primary Script | Input Data / Cache | Output Artifact |
|---|---|---|---|
| **Validation Framework** | [validation_framework.py](file:///z:/Amazon%20ML/experiments/phase2/validation_framework.py) | `experiments/val_sample_50k.parquet` | Benchmark metrics dictionary |
| **Candidate Retrieval** | [candidate_generation.py](file:///z:/Amazon%20ML/src/candidate_generation.py) | `data/raw/`, `experiments/cache/` | 7-channel candidate pairs |
| **Feature Extraction** | [build_val_features.py](file:///z:/Amazon%20ML/experiments/phase2/build_val_features.py) | Normalized entities | `experiments/phase2/val_50k_features.parquet` |
| **Champion Scorer** | [ensemble_pointwise.py](file:///z:/Amazon%20ML/experiments/phase2/models/ensemble_pointwise.py) | 24-feature matrix | `experiments/phase2/exp11_ensemble_results.json` |
| **Official Evaluator** | [evaluate.py](file:///z:/Amazon%20ML/src/evaluate.py) | Predictions TSV + Ground Truth TSV | Official Macro $F_{0.5}$ |

---

## 2. Step-by-Step Reproduction Guide

To reproduce the verified Phase 2 champion results from a fresh clone:

```bash
# 1. Activate environment
venv\Scripts\activate

# 2. Run independent test suite
pytest

# 3. Verify metric evaluator agreement
pytest tests/test_f05_evaluator_audit.py

# 4. Run retrieval ceiling and error decomposition
python experiments/phase2/retrieval_ceiling_and_error_decomposition.py

# 5. Run Champion Pointwise Ensemble (EXP-11)
python experiments/phase2/models/ensemble_pointwise.py
```

---

## 3. Immediate Phase 3 Execution Priorities

1. **Break the Retrieval Ceiling ($0.8285 \to >0.92$):**
   - Implement Dense Multilingual Vector Retrieval using `sentence-transformers` (`paraphrase-multilingual-MiniLM-L12-v2` or `BGE-M3`).
   - Build offline Faiss / HNSW index for the 10.3M target pool.
   - Introduce Devanagari-to-Latin transliteration to recover the 13.37% cross-script Indian targets.
2. **Expand Candidate Budget ($K=40 \to 80$):**
   - Scale the candidate union to recover truncated true matches, pushing candidate-constrained link recall from $69.3\% \to >82\%$.
3. **Incorporate Relational S2 $\leftrightarrow$ S3 Consensus:**
   - Implement mutual target similarity features to reinforce candidate pairs supported by both sources.
4. **Deploy Full Submission Pipeline:**
   - Build chunked, memory-safe inference pipeline for the full 2.2M test queries.

---

*Certified by Autonomous Lead Researcher | Date: 2026-09-26*
