# Phase 1 Candidate Generation Evaluation

**Project:** Amazon ML Challenge 2026 — Business Entity Resolution  
**Phase:** 1 — Candidate Generation Evaluation  
**Validation Set:** 50,000 S1 queries (stratified by country and match count bin)  
**True Links in Validation Set:** 172,948  
**Date:** 2026-09-25

---

## Final 7-Channel Pipeline — Full Evaluation

### Unbudgeted (All Retrieved Pairs)

| Metric | Value |
|--------|-------|
| Link Recall | **81.556%** |
| Entity Coverage | **96.75%** |
| S2 Link Recall | 81.838% |
| S3 Link Recall | 81.289% |
| S2-only Entity Recall | 85.97% |
| S3-only Entity Recall | 89.205% |
| Both-source Entity Recall | 96.814% |
| Total Candidate Pairs | 5,633,977 |
| Mean Candidates per S1 | 112.68 |
| P95 Candidates per S1 | 398.0 |
| Max Candidates per S1 | 5,633 |

### Budgeted — Priority+Channel Ranking

| K | Link Recall | Entity Coverage | Total Pairs | Avg/Query | P95/Query |
|---|-------------|-----------------|-------------|-----------|-----------|
| 20 | 65.53% | 86.47% | 810,028 | 16.2 | 20.0 |
| 25 | 66.49% | 87.10% | 978,689 | 19.6 | 25.0 |
| 30 | 67.25% | 87.27% | 1,148,860 | 22.8 | 30.0 |
| 35 | 67.90% | 88.13% | 1,296,235 | 25.9 | 35.0 |
| **40** | **69.33%** | **89.19%** | **1,456,778** | **29.1** | **40.0** |
| 50 | 71.54% | 91.30% | 1,743,907 | 34.9 | 50.0 |

**Chosen: K=40** — best Pareto point for Phase 2 matching model input.

---

## Channel-by-Channel Recall Progression

| Step | Channel Added | Recall | Coverage | Δ Recall |
|------|--------------|--------|----------|----------|
| 1 | Exact Clean Name | 21.48% | 54.22% | +21.48 |
| 2 | + Token-Sorted Suffix-Stripped | 40.78% | 77.38% | +19.30 |
| 3 | + Exact Address | ~50.5% | ~84% | +9.7 |
| 4 | + Street Prefix (3-token+digit) | 57.93% | 88.28% | +7.4 |
| 5 | + Rare Brand Token (freq≤300) | 66.58% | 91.02% | +8.6 |
| 6 | + 2-Word Name Bigram (freq≤500) | 77.71% | 95.48% | +11.1 |
| 7 | + Sorted Address Token Set | **81.56%** | **96.75%** | **+3.8** |

---

## Individual Channel Statistics (Full 6-Channel Benchmark on 50K Validation)

| Channel | Raw Pairs | Standalone Recall | Standalone Coverage |
|---------|-----------|------------------|---------------------|
| Ch1 Exact Name | 484,506 | 21.5% | 54.2% |
| Ch2 Sorted Name | 1,704,573 | 40.8% | 77.4% |
| Ch3 Std Address | 34,700 | ~15% | ~35% |
| Ch4 Sorted Address | 89,876 | 28.3% | 57.1% |
| Ch5 Name Bigram | 1,878,642 | — | — |
| Ch6 Street Prefix | 1,250,988 | — | — |
| Ch7 Brand Token | 1,634,559 | 25.9% | ~60% |

---

## Normalization Collision Audit (100K Sample)

Verified that suffix stripping and token sorting do NOT cause candidate explosion:

| Metric | Value |
|--------|-------|
| Mean group size after normalization | 1.19 |
| P99 group size | 5.0 |
| Max group size | 23 |
| Groups with >50 members | 0 |

Safe to use as retrieval keys without precision collapse.

---

## Validation Split Metadata

| Split | S1 Queries | True Links | Singleton Rate | Strata |
|-------|-----------|------------|----------------|--------|
| val_sample_50k.parquet | 49,998 | 172,948 | 5.584% | Country × match_count_bin |
| val_split_20pct.parquet | 441,363 | — | — | Country × match_count_bin |

Stratification confirmed: singletons, low-match (1-3), medium-match (4-10), and high-match entities proportionally sampled.

---

## Missed Link Forensics

38,547 true links (22.29%) missed by all 7 channels (unbudgeted):

| Failure Mode | Estimated Share | Recoverable? |
|--------------|-----------------|-------------|
| Transliteration (Indian scripts) | ~40% | Partially (via sorted address) |
| Severe OCR/synthetic corruption | ~25% | No (requires fuzzy/edit-distance) |
| Domain-name target format | ~15% | Partially (via address join) |
| Extreme name truncation / noise | ~20% | No (structurally ambiguous) |

**Conclusion:** A theoretical upper bound of ~85-87% unbudgeted recall exists for pure string-matching approaches on this dataset. The remaining 13-15% requires either transliteration models, character-level fuzzy retrieval, or acceptance that competition data contains irreducible noise.

---

## Full Test Set Production Run

| Metric | Value |
|--------|-------|
| S1 queries | 1,732,544 |
| S2 targets | 4,887,273 |
| S3 targets | 5,082,316 |
| Total targets | 9,969,589 |
| Target normalization time | 6.0s |
| Frequency filter build time | 8.8s |
| Avg time per 100K chunk | ~22s |
| Total chunks | 18 |
| Estimated total runtime | ~8 min |
| Peak RAM estimate | ~6 GB |
| Avg candidates per S1 (K=40) | ~30 |
| Output file | `output/candidate_pairs.tsv` |
