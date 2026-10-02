# Phase 2 — Cardinality, No-Match & Query-Level Decision Optimization
## Amazon ML Challenge 2026 — Multi-Source Business Entity Resolution

> **Role:** Lead Entity Resolution Researcher & ML Scientist  
> **Diagnostic Focus:** Cardinality Match Bins, No-Match Detection, and Set-Valued Decisions  
> **Evaluation Sample:** N = 15,000 holdout queries  
> **Date:** 2026-09-26

---

## 1. Ground Truth Cardinality Architecture

Unlike traditional 1-to-1 record linkage benchmarks, the Amazon ML Challenge entity resolution dataset is intrinsically **multi-match and multi-source**:

| Match Bin | Cardinality ($|T_i|$) | Query Count (Holdout) | % of Queries | Champion Macro $F_{0.5}$ | Champion Precision | Champion Recall | Diagnostic Interpretation |
|---|---|---|---|---|---|---|---|
| **`match_bin_0`** | **0 (Singletons)** | 796 | **5.31%** | **0.9133** | 0.0000 | 0.0000 | High-risk singleton cliff; 91.33% accuracy avoids F0.5=0 penalty |
| **`match_bin_1`** | **1 (Unique match)** | 821 | **5.47%** | **0.6044** | 0.6015 | 0.6297 | Most fragile slice: 1 false candidate drops precision by 50% |
| **`match_bin_2`** | **2–3 matches** | 6,173 | **41.15%** | **0.7549** | 0.8254 | 0.6335 | Dominant cluster core across S2 and S3 |
| **`match_bin_3`** | **4–6 matches** | 6,580 | **43.87%** | **0.8037** | 0.9069 | 0.6385 | Dense entity profiles; high precision multi-match |
| **`match_bin_4`** | **7+ matches** | 630 | **4.20%** | **0.8039** | 0.9334 | 0.6141 | Large corporate branch networks |

### Key Insight:
- **89.2%** of all entities have cardinality $\ge 2$.
- Entities with true matches almost always have **at least one match in Source 2 AND at least one match in Source 3** (89.02% cross-source linkage).

---

## 2. Independent Binary Decisions vs. Set-Valued Decisions

### The Standard Threshold Policy:
$$\hat{Y}_i = \{ c \in C_i \mid P(\text{match}(i, c)) \ge \theta^* \}$$
Under this independent policy:
1. **Singleton Safety:** If $\max_{c \in C_i} P(i, c) < \theta^*$, the predicted set is empty ($\hat{Y}_i = \emptyset$), correctly recovering the singleton case ($F_{0.5} = 1.0$).
2. **Cardinality Under-Prediction Risk:** If a true entity has 4 matches, and two candidates score $0.62$ and two score $0.53$, a fixed threshold $\theta^* = 0.55$ truncates the set to size 2, dropping recall from $1.0 \to 0.50$ and $F_{0.5}$ from $1.0 \to 0.833$.

### The Weakness in `match_bin_1` ($F_{0.5} = 0.6044$):
For entities with exactly 1 true match:
- If the model accepts 1 true candidate + 1 false candidate:
  $$P = 0.5, \quad R = 1.0 \implies F_{0.5} = \frac{1.25(0.5)(1.0)}{0.25(0.5) + 1.0} = \frac{0.625}{1.125} = \mathbf{0.5556}$$
A single spurious false candidate cuts the query score nearly in half.

---

## 3. Decision Optimization Strategies

### 1. The Dynamic Top-Score Margin Rule
For any query $i$, let $p_{(1)} \ge p_{(2)} \ge \dots \ge p_{(k)}$ be the sorted candidate probabilities:
- If $p_{(1)} < 0.40$: Predict empty set (No match / Singleton).
- If $p_{(1)} \ge 0.65$ and $p_{(1)} - p_{(2)} \ge 0.25$: High confidence single-match; accept only $c_{(1)}$.
- If multiple candidates score $\ge 0.55$: Accept all candidates within a relative margin:
  $$c_{(j)} \in \hat{Y}_i \iff p_{(j)} \ge 0.55 \quad \text{AND} \quad \frac{p_{(j)}}{p_{(1)}} \ge 0.70$$

### 2. Multi-Source Completeness Constraint
Since 89% of true entities link to both Source 2 and Source 3:
- If an S2 candidate scores with extreme confidence ($p \ge 0.85$), the threshold for supporting S3 candidates should be adaptively relaxed from $0.55 \to 0.45$ to capture the corresponding sister record.

---

*Certified by Autonomous Lead Researcher | Date: 2026-09-26*
