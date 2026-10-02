# Phase 1 Design Notes: Data Preparation, Representation Engineering & Candidate Retrieval

**Document:** Architecture Design & Experiment Protocol for Phase 1  
**Project:** Amazon ML Challenge 2026 - Business Entity Resolution Challenge  
**Team:** Senior Autonomous Applied ML Competition Architecture Team  
**Date:** September 25, 2026  

---

## 1. Architectural Strategy & Retrieval Objective

In large-scale entity resolution with $N_1 = 1.73 \times 10^6$ query entities and $N_{target} = 9.97 \times 10^6$ target candidates, full pairwise scoring requires $\approx 1.72 \times 10^{13}$ comparisons.
The retrieval/blocking layer must reduce this comparison space by $> 99.999\%$ while retaining the highest possible proportion of true links.

### The Core Retrieval Principles
1. **Recall is the Upper Bound:** No downstream classifier can recover a true match that candidate generation omits.
2. **Precision / Candidate Volume Tradeoff:** Downstream pairwise scoring cost scales linearly with candidate volume. If $K$ is too large ($K > 50$), memory and feature computation explode. If $K$ is too small ($K < 10$), multi-match entities (which can have up to 11 true links) will suffer severe link truncation.
3. **Multi-Channel Complementarity:** Different corruption mechanisms require different retrieval keys:
   - Suffix stripping $\rightarrow$ Standardized legal suffix representation.
   - Word reordering $\rightarrow$ Token-sorted bag-of-words representation.
   - Typos / OCR / spelling variants $\rightarrow$ Character 3-gram sparse retrieval.
   - Name collisions (chain stores / branches) $\rightarrow$ Address numeric / postal code intersection.
4. **Leakage-Safe Validation:** All vocabularies, IDF frequencies, and token rarity thresholds must be computed strictly on training data folds.
5. **Multi-Representation Hypothesis:** Rather than forcing a single destructive normalization, maintain raw, case/punctuation normalized, legal-suffix stripped, and accent-folded representations to serve both fast retrieval and fine-grained matching.

---

## 2. Metric Specifications for Candidate Evaluation

Every candidate generation configuration must be evaluated against standard metrics:

1. **True Link Recall:**
   $$\text{Link Recall} = \frac{\sum_{i=1}^{N_{S1}} |C(S1_i) \cap T(S1_i)|}{\sum_{i=1}^{N_{S1}} |T(S1_i)|}$$
   where $C(S1_i)$ is the candidate set for entity $S1_i$, and $T(S1_i)$ is the ground-truth target match set.
2. **Entity Coverage:**
   $$\text{Entity Coverage} = \frac{1}{|S1_{\text{non-singleton}}|} \sum_{S1_i \in S1_{\text{non-singleton}}} \mathbb{I}(|C(S1_i) \cap T(S1_i)| > 0)$$
3. **Source-Specific Recall:**
   - Link Recall on S1 $\rightarrow$ S2 true links.
   - Link Recall on S1 $\rightarrow$ S3 true links.
4. **Relationship-Specific Recall:**
   - Recall on entities with matches in S2 only.
   - Recall on entities with matches in S3 only.
   - Recall on entities with matches in both S2 and S3.
5. **Efficiency & Volume Metrics:**
   - Total candidate pairs generated.
   - Mean, median, P90, P95, P99, and max candidates per S1 entity.
   - Reduction ratio:
     $$\text{Reduction Ratio} = 1 - \frac{\text{Total Candidates Generated}}{N_{S1} \times N_{target}}$$
   - Candidate generation throughput (queries per second) and RAM footprint.

---

## 3. Experiment Roadmap

| Exp ID | Focus / Channel | Method Description | Primary Evaluation Question |
|---|---|---|---|
| **EXP-1.0** | Validation Split | 50K stratified S1 queries vs. FULL S2/S3 target pool | Establish leakage-free benchmark harness |
| **EXP-1.1** | Exact & Suffix Inverted Index | Exact cleaned name + suffix-stripped inverted index | What is the base recall of exact lexical matching? |
| **EXP-1.2** | Token-Order Invariant Index | Bag-of-words sorted token index | Does token reordering recover the 5.68% observed permutations? |
| **EXP-1.3** | Rare Token / Inverted Token Index | Multi-token inverted index with IDF rarity thresholds | How much does informative token matching boost recall? |
| **EXP-1.4** | Character 3-Gram Sparse Retrieval | Substring / 3-gram character Jaccard/TF-IDF hashing | Does character-level retrieval capture typos and transliterations? |
| **EXP-1.5** | Address & Postal Auxiliary Channel | PIN/ZIP code + numeric address token intersection | Can address keys resolve hard negatives and recover branch links? |
| **EXP-1.6** | Multi-Pass Union & Budget Tuning | Union of complementary channels with top-$K$ budget ($K \in [15, 20, 25, 30]$) | What is the optimal recall-to-volume Pareto frontier? |
| **EXP-1.7** | French Accent Representation Benchmark | Raw accents vs. NFKD accent-folded vs. multi-rep on France | Does accent folding improve French retrieval recall without collision explosion? |
| **EXP-1.8** | Full-Scale Test Generation | Execute winning architecture on full 1.73M test set | Generate verified `candidate_pairs.tsv` |

---
