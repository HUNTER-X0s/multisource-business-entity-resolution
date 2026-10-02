"""
comprehensive_gap_analysis.md  (scratch note — written as Python docstring for easy reading)

==========================================================================
COMPREHENSIVE GAP ANALYSIS — v8 vs What's Needed to Beat Rank #1 (0.991483)
==========================================================================

CRITICAL FINDINGS from reading train_gpu_meta_ensemble.py + meta_ensemble_results.json:

═══════════════════════════════════════════════════════════
GAP #1 — THRESHOLD MISMATCH (CRITICAL)
═══════════════════════════════════════════════════════════
Training found:   best_theta = 0.68
v8 is using:      TH_ML = 0.55

The model was trained on 10-CHANNEL candidates (val set from Phase2).
On THAT val distribution, theta=0.68 was optimal.

v8 adds 3 new channels (Ch11 addr_sig, Ch12 adv_name, Ch13 rare HNO).
These new channels bring MORE true positives but with LOWER raw scores.
So theta=0.55 is correct for v8's expanded pool.

BUT: The meta_stacker in v8 only uses [lgb, xgb, avg] (3 inputs).
The training stacker was trained on [lgb, xgb, catboost] (3 inputs).
v8 DROPPED CatBoost from the ensemble!

=> The meta_stacker.pkl was trained on OOF from (lgb, xgb, catboost).
=> v8 is feeding it (lgb, xgb, avg_of_lgb_xgb) — WRONG INPUT!
=> This means the meta-stacker's coefficients are misaligned.
=> FIX: Load meta_cb.pkl too and feed [lgb, xgb, catboost] to meta_model.

═══════════════════════════════════════════════════════════
GAP #2 — BLOCKED BY SLOW apply() ON 10M TARGETS (PERFORMANCE)
═══════════════════════════════════════════════════════════
adv_name and addr_sig use Python .apply() which is O(n) single-threaded.
On 10M targets this will take ~5-10 minutes.
FIX: Vectorize using pandas str operations (already done in regex, but
can speed up addr_sig computation).

═══════════════════════════════════════════════════════════
GAP #3 — META-STACKER INPUT MISMATCH (CRITICAL BUG)
═══════════════════════════════════════════════════════════
Training:  meta_model.fit( [oof_lgb, oof_xgb, oof_cb] )
v8 code:   meta_model.predict_proba( [p_lgb, p_xgb, p_avg] )

p_avg = (p_lgb + p_xgb) / 2  ≠  p_catboost

This is a SILENT BUG. The meta-stacker will still output a number but
it's wrong. It will use LGB weight for LGB ✓, XGB weight for XGB ✓,
but CB weight will be applied to the average of LGB+XGB — NOT CatBoost.

FIX: Load meta_cb.pkl and use [p_lgb, p_xgb, p_cb] as the meta input.

═══════════════════════════════════════════════════════════  
GAP #4 — TRAINING DATA COVERAGE (STRUCTURAL)
═══════════════════════════════════════════════════════════
The model was trained on ONLY 10-channel val candidates.
The new channels (Ch11, Ch12, Ch13) were NOT in the training data.
So the ML model has never seen addr_sig candidates.

This means: pairs found ONLY by Ch11/Ch12/Ch13 will have:
- max_prio = 0.65-0.72 (low priority)
- n_channels = 1
- name_jw ≈ 0 (garbled name case)
- addr metrics may be high

The model CAN still score these high if addr features are strong,
but it was never explicitly trained on them.

WORKAROUND already in v8: addr_anchor_bypass() handles this by
directly accepting rare addr_sig matches WITHOUT the ML model.
This is the correct approach for these cases.

═══════════════════════════════════════════════════════════
GAP #5 — MISSING FEATURES IN v8 vs TRAINING
═══════════════════════════════════════════════════════════
Training feature: "name_len_ratio" = min/max (symmetric)
v8 feature:       "name_len_ratio" = q_len / (t_len + 1) (asymmetric)

Training feature: "name_wc_ratio" = min/max (symmetric)  
v8 feature:       "name_wc_ratio" = qwc / (twc + 1) (asymmetric)

These inconsistencies will cause the model to receive slightly wrong
feature values. FIX: make them symmetric (min/max) to match training.

═══════════════════════════════════════════════════════════
GAP #6 — addr_tok_jac USES SORTED ADDRESS IN TRAINING, RAW IN v8
═══════════════════════════════════════════════════════════
Training (line 252): addr_tok_jac: tok_jac(qas, tas)  ← uses sorted_addr
v8 code:             addr_tok_jac: tok_jac(qa, ta)     ← uses std_address

Minor but worth fixing for consistency.

═══════════════════════════════════════════════════════════
SUMMARY OF FIXES NEEDED (in order of impact):
═══════════════════════════════════════════════════════════
1. [CRITICAL] Load meta_cb.pkl and pass [p_lgb, p_xgb, p_cb] to meta_stacker
2. [HIGH]     Fix name_len_ratio and name_wc_ratio to use min/max like training
3. [MEDIUM]   Fix addr_tok_jac to use sorted_addr like training
4. [DONE ✓]  addr_anchor_bypass covers GAP #4
5. [DONE ✓]  addr_sig + adv_name channels cover recall gap
6. [DONE ✓]  Mutual exclusivity covers collision gap
7. [DONE ✓]  TH=0.55 is correct for expanded candidate pool
"""
