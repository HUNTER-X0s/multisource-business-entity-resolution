"""
assemble_champion_v9_clean.py
==============================
Assembles Champion v9 by applying optimal precision-focused post-processing
to the 4-model GPU Meta-Ensemble (CatBoost + LightGBM + XGBoost + Stacker)
already computed in shards_v8.

Fixes the two fatal bugs in v8:
1. Discards unverified Address Bypass pairs (prob == 1.0) that flooded false positives.
2. Restores optimal ML threshold (prob >= 0.68) + margin gap fallback (top >= 0.50, gap >= 0.10).
3. Restores per-source cardinality cap (<= 3 from S2, <= 3 from S3).
4. Restores global Mutual Exclusivity (each target claimed by highest confidence query).
"""

import sys
import time
import subprocess
from pathlib import Path
from collections import defaultdict
import pandas as pd

ROOT = Path("z:/Amazon ML")
SHARDS_DIR = ROOT / "output" / "shards_v8"
OUTPUT_DIR = ROOT / "output"

TH_PRIMARY = 0.68     # Optimal cross-validation threshold from meta_ensemble_results.json
TH_MARGIN  = 0.50     # Margin fallback threshold
MIN_GAP    = 0.10     # Minimum gap over runner-up
CAP_PER_SOURCE = 3    # Maximum targets per source (S2 / S3)

def main():
    t0 = time.time()
    print("=" * 70)
    print("ASSEMBLING CHAMPION v9: CLEAN PRECISION ENSEMBLE")
    print(f"Optimal Threshold: {TH_PRIMARY} | Margin: {TH_MARGIN} (gap {MIN_GAP}) | Cap: {CAP_PER_SOURCE}")
    print("=" * 70)

    # 1. Collect all candidate claims across 9 shards
    print("\n[1] Reading 9 shards and applying clean precision filtering...")
    all_claims = []  # (target_id, prob, query_id)
    n_total_queries = 0
    all_qids = []

    for si in range(9):
        p = SHARDS_DIR / f"shard_{si:02d}_matches.tsv"
        if not p.exists():
            print(f"  ERROR: {p} not found!")
            return
        
        t_shard = time.time()
        # Read with pandas
        df = pd.read_csv(p, sep="\t", dtype=str).fillna("")
        n_total_queries += len(df)
        
        shard_accepted = 0
        for _, row in df.iterrows():
            qid = row["source1_entity_id"]
            all_qids.append(qid)
            mids = row["matched_entity_ids"]
            pstr = row.get("probs", "")
            if not mids or not pstr:
                continue
            
            tids = mids.split(",")
            prbs = [float(x) for x in pstr.split("|")]
            
            # Pair targets with probabilities
            # Exclude fake 1.0 bypass matches (only consider genuine ML predictions < 1.0)
            ml_pairs = [(prob, tid) for prob, tid in zip(prbs, tids) if prob < 1.0]
            if not ml_pairs:
                continue
            
            # Sort by probability descending
            ml_pairs.sort(reverse=True, key=lambda x: x[0])
            
            accepted = []
            s2_cnt = 0
            s3_cnt = 0
            
            # Primary rule: ML confidence >= TH_PRIMARY with source capping
            for prob, tid in ml_pairs:
                if prob >= TH_PRIMARY:
                    if tid.startswith("S2-") and s2_cnt < CAP_PER_SOURCE:
                        accepted.append((prob, tid))
                        s2_cnt += 1
                    elif tid.startswith("S3-") and s3_cnt < CAP_PER_SOURCE:
                        accepted.append((prob, tid))
                        s3_cnt += 1
            
            # Margin fallback if no candidates met primary threshold
            if not accepted and ml_pairs:
                top_p, top_tid = ml_pairs[0]
                sec_p = ml_pairs[1][0] if len(ml_pairs) > 1 else 0.0
                if top_p >= TH_MARGIN and (len(ml_pairs) == 1 or (top_p - sec_p >= MIN_GAP)):
                    accepted.append((top_p, top_tid))
            
            for prob, tid in accepted:
                all_claims.append((tid, prob, qid))
                shard_accepted += 1
                
        print(f"  Shard {si+1}/9: {len(df):,} queries processed | {shard_accepted:,} accepted claims in {time.time()-t_shard:.1f}s")

    print(f"\nTotal queries processed: {n_total_queries:,}")
    print(f"Total clean claims: {len(all_claims):,}")

    # 2. Global Mutual Exclusivity: each target assigned to argmax query
    print("\n[2] Applying Global Mutual Exclusivity...")
    best_target_match = {}
    for tid, prob, qid in all_claims:
        if tid not in best_target_match or prob > best_target_match[tid][0]:
            best_target_match[tid] = (prob, qid)

    # Invert: query -> list of claimed targets
    query_matches = defaultdict(list)
    for tid, (prob, qid) in best_target_match.items():
        query_matches[qid].append(tid)

    # 3. Assemble matching_results.tsv
    print("\n[3] Writing final matching_results.tsv...")
    out_file = OUTPUT_DIR / "matching_results.tsv"
    v9_backup = OUTPUT_DIR / "matching_results_v9_clean.tsv"
    
    n_matched = 0
    total_edges = 0
    with open(out_file, "w", encoding="utf-8") as f_out, open(v9_backup, "w", encoding="utf-8") as f_bak:
        header = "source1_entity_id\tmatched_entity_ids\n"
        f_out.write(header)
        f_bak.write(header)
        
        for qid in all_qids:
            tids = query_matches.get(qid, [])
            match_str = ",".join(sorted(tids))
            line = f"{qid}\t{match_str}\n"
            f_out.write(line)
            f_bak.write(line)
            if match_str:
                n_matched += 1
                total_edges += len(tids)

    print(f"  Output saved -> {out_file}")
    print(f"  Matched queries: {n_matched:,} / {n_total_queries:,} ({100*n_matched/n_total_queries:.2f}%)")
    print(f"  Total edges: {total_edges:,} (avg {total_edges/n_matched:.2f} per matched query)")

    # 4. Use v8 candidate pairs (which has 99.96% candidate recall)
    cand_v8 = OUTPUT_DIR / "candidate_pairs_v8.tsv"
    cand_curr = OUTPUT_DIR / "candidate_pairs.tsv"
    if cand_v8.exists():
        print(f"\n[4] Linking high-recall candidate_pairs.tsv (99.96% recall)...")
        import shutil
        shutil.copy2(cand_v8, cand_curr)
        print(f"  candidate_pairs.tsv updated from {cand_v8.name} ({cand_curr.stat().st_size:,} bytes)")

    # 5. Run official validator
    print("\n[5] Running official competition validator...")
    validator = ROOT / "dataset" / "student_resource" / "utils" / "validate_submission.py"
    test_dir  = ROOT / "dataset" / "raw" / "test"
    try:
        r = subprocess.run(
            [sys.executable, str(validator),
             "--matching",   str(out_file),
             "--candidate",  str(cand_curr),
             "--test-dir",   str(test_dir)],
            capture_output=True, text=True, timeout=600,
        )
        print(r.stdout)
        if r.stderr: print("STDERR:", r.stderr)
        print(f"Validator exit code: {r.returncode}")
    except Exception as e:
        print(f"Validator error: {e}")

    print("\n" + "=" * 70)
    print(f"CHAMPION v9 CLEAN COMPLETE IN {time.time()-t0:.1f}s!")
    print("=" * 70)

if __name__ == "__main__":
    main()
