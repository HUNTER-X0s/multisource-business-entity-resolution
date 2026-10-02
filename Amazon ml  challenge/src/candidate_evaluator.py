"""
src/candidate_evaluator.py
=========================
Independent, scientifically rigorous candidate evaluation module for Phase 1.
Computes true-link recall, entity coverage, source-specific recall, relationship-specific
recall, candidate volume distributions, and reduction ratio.

Usage:
    from src.candidate_evaluator import CandidateEvaluator
    evaluator = CandidateEvaluator(ground_truth_dict, s1_metadata_df)
    metrics = evaluator.evaluate(candidates_dict)
"""

import time
from typing import Dict, List, Set, Any
import numpy as np


class CandidateEvaluator:
    def __init__(self, ground_truth: Dict[str, Set[str]], s1_metadata: Dict[str, Dict[str, Any]] = None):
        """
        ground_truth: Dict mapping source1_entity_id -> set of true matching target IDs (S2-*, S3-*)
        s1_metadata: Optional dict mapping source1_entity_id -> dict with 'country', etc.
        """
        self.ground_truth = ground_truth
        self.s1_metadata = s1_metadata or {}
        
        # Precompute ground truth breakdown
        self.total_s1 = len(ground_truth)
        self.total_links = sum(len(m) for m in ground_truth.values())
        
        self.s2_true_links = 0
        self.s3_true_links = 0
        self.singleton_entities = set()
        self.s2_only_entities = set()
        self.s3_only_entities = set()
        self.both_entities = set()
        
        for s1_id, matches in ground_truth.items():
            if not matches:
                self.singleton_entities.add(s1_id)
                continue
            has_s2 = any(m.startswith("S2-") for m in matches)
            has_s3 = any(m.startswith("S3-") for m in matches)
            
            s2_cnt = sum(1 for m in matches if m.startswith("S2-"))
            s3_cnt = sum(1 for m in matches if m.startswith("S3-"))
            self.s2_true_links += s2_cnt
            self.s3_true_links += s3_cnt
            
            if has_s2 and has_s3:
                self.both_entities.add(s1_id)
            elif has_s2:
                self.s2_only_entities.add(s1_id)
            elif has_s3:
                self.s3_only_entities.add(s1_id)
                
        self.non_singleton_entities = set(ground_truth.keys()) - self.singleton_entities

    def evaluate(self, candidates: Dict[str, Set[str]], experiment_id: str = "exp", elapsed_seconds: float = 0.0) -> Dict[str, Any]:
        """
        Evaluates a candidate dictionary {s1_id: set_of_candidate_ids} against ground truth.
        """
        t0 = time.time()
        
        recovered_links = 0
        recovered_s2_links = 0
        recovered_s3_links = 0
        
        covered_entities = 0
        recovered_s2_only_entities = 0
        recovered_s3_only_entities = 0
        recovered_both_entities = 0
        
        # Country breakdowns if metadata present
        country_true_links = {}
        country_recovered_links = {}
        
        candidate_counts = []
        singleton_candidate_counts = []
        
        for s1_id, true_matches in self.ground_truth.items():
            cands = candidates.get(s1_id, set())
            n_cands = len(cands)
            candidate_counts.append(n_cands)
            
            country = self.s1_metadata.get(s1_id, {}).get("country", "Unknown")
            if country not in country_true_links:
                country_true_links[country] = 0
                country_recovered_links[country] = 0
            country_true_links[country] += len(true_matches)
            
            if not true_matches:
                singleton_candidate_counts.append(n_cands)
                continue
                
            intersection = true_matches.intersection(cands)
            n_inter = len(intersection)
            recovered_links += n_inter
            country_recovered_links[country] += n_inter
            
            if n_inter > 0:
                covered_entities += 1
                if s1_id in self.s2_only_entities:
                    recovered_s2_only_entities += 1
                elif s1_id in self.s3_only_entities:
                    recovered_s3_only_entities += 1
                elif s1_id in self.both_entities:
                    recovered_both_entities += 1
                    
            for m in intersection:
                if m.startswith("S2-"):
                    recovered_s2_links += 1
                elif m.startswith("S3-"):
                    recovered_s3_links += 1
                    
        # Metrics
        overall_link_recall = recovered_links / self.total_links if self.total_links > 0 else 0.0
        entity_coverage = covered_entities / len(self.non_singleton_entities) if self.non_singleton_entities else 0.0
        s2_link_recall = recovered_s2_links / self.s2_true_links if self.s2_true_links > 0 else 0.0
        s3_link_recall = recovered_s3_links / self.s3_true_links if self.s3_true_links > 0 else 0.0
        
        s2_only_entity_recall = recovered_s2_only_entities / len(self.s2_only_entities) if self.s2_only_entities else 0.0
        s3_only_entity_recall = recovered_s3_only_entities / len(self.s3_only_entities) if self.s3_only_entities else 0.0
        both_entity_recall = recovered_both_entities / len(self.both_entities) if self.both_entities else 0.0
        
        # Volume stats
        cands_arr = np.array(candidate_counts, dtype=np.int32)
        total_candidate_pairs = int(np.sum(cands_arr))
        mean_cands = float(np.mean(cands_arr))
        median_cands = float(np.median(cands_arr))
        p75_cands = float(np.percentile(cands_arr, 75))
        p90_cands = float(np.percentile(cands_arr, 90))
        p95_cands = float(np.percentile(cands_arr, 95))
        p99_cands = float(np.percentile(cands_arr, 99))
        max_cands = int(np.max(cands_arr)) if len(cands_arr) > 0 else 0
        
        singleton_arr = np.array(singleton_candidate_counts, dtype=np.int32)
        singleton_mean_cands = float(np.mean(singleton_arr)) if len(singleton_arr) > 0 else 0.0
        
        country_recalls = {}
        for c, t_links in country_true_links.items():
            if t_links > 0:
                country_recalls[c] = round(country_recovered_links[c] / t_links * 100, 2)
                
        eval_time = round(time.time() - t0, 3)
        
        report = {
            "experiment_id": experiment_id,
            "evaluated_s1_entities": self.total_s1,
            "non_singleton_entities": len(self.non_singleton_entities),
            "singleton_entities": len(self.singleton_entities),
            "total_true_links": self.total_links,
            "recovered_true_links": recovered_links,
            "overall_link_recall_pct": round(overall_link_recall * 100, 3),
            "entity_coverage_pct": round(entity_coverage * 100, 3),
            "s2_link_recall_pct": round(s2_link_recall * 100, 3),
            "s3_link_recall_pct": round(s3_link_recall * 100, 3),
            "s2_only_entity_recall_pct": round(s2_only_entity_recall * 100, 3),
            "s3_only_entity_recall_pct": round(s3_only_entity_recall * 100, 3),
            "both_source_entity_recall_pct": round(both_entity_recall * 100, 3),
            "country_recalls_pct": country_recalls,
            "candidate_volume": {
                "total_candidate_pairs": total_candidate_pairs,
                "mean_per_s1": round(mean_cands, 2),
                "median_per_s1": median_cands,
                "p75": p75_cands,
                "p90": p90_cands,
                "p95": p95_cands,
                "p99": p99_cands,
                "max": max_cands,
                "singleton_mean_cands": round(singleton_mean_cands, 2)
            },
            "runtime_seconds": round(elapsed_seconds, 2),
            "eval_time_seconds": eval_time
        }
        return report
