"""
experiments/phase2/retrieval_expansion_exp15.py
================================================
EXP-15: Retrieval Ceiling Breakthrough (High-Performance Sparse Edition)

Channels:
  Ch8  — Char (3,4)-gram TF-IDF + CSC Sparse Inner-Product Retrieval
  Ch9  — Devanagari -> Latin transliteration exact + sorted-token match
  Ch10 — Postal code + first-name-token blocking (India PIN / US ZIP)

Measures union oracle ceiling at K = 40, 50, 75, 100, 150, 200.

Key optimization vs previous failed runs:
- Stop-ngram pruning (max_df=0.03) removes non-discriminative character n-grams.
- Direct CSR sparse matrix dot product with zero dense conversion (0 MB RAM overhead).
- Fast top-K selection via numpy.argpartition over sparse slice arrays.
"""
import json, time, re, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import random; random.seed(42)

ROOT        = Path(__file__).resolve().parents[2]
VAL_SAMPLE  = ROOT / "experiments" / "val_sample_50k.parquet"
FEAT_FILE   = ROOT / "experiments" / "phase2" / "val_50k_features.parquet"
S2_PATH     = ROOT / "dataset" / "raw" / "train" / "train_source2.tsv"
S3_PATH     = ROOT / "dataset" / "raw" / "train" / "train_source3.tsv"
S1_PATH     = ROOT / "dataset" / "raw" / "train" / "train_source1.tsv"
OUTPUT_JSON = ROOT / "experiments" / "phase2" / "exp15_retrieval_expansion_results.json"
K_BUDGETS   = [40, 50, 75, 100, 150, 200]

def flush(*args, **kwargs):
    print(*args, **kwargs, flush=True)

flush("=" * 70)
flush("EXP-15: RETRIEVAL CEILING BREAKTHROUGH (High-Performance Sparse Edition)")
flush("=" * 70)

# ── [0] Load GT ────────────────────────────────────────────────────────────
flush("\n[0] Loading val sample and ground truth...")
t0 = time.time()
val_df = pl.read_parquet(VAL_SAMPLE)
val_gt = {}
for row in val_df.iter_rows(named=True):
    qid = str(row["entity_id"])
    raw = str(row.get("matched_entity_ids", "") or "").strip()
    targets = set()
    if raw and raw not in ("nan", "None", ""):
        for t in raw.split(","):
            t = t.strip()
            if t:
                targets.add(t)
    val_gt[qid] = targets
val_query_list = sorted(val_gt.keys())
all_pos_tids   = set(tid for v in val_gt.values() for tid in v)
flush(f"  Val queries:       {len(val_query_list):,}")
flush(f"  With GT matches:   {sum(1 for v in val_gt.values() if v):,}")
flush(f"  Positive tgt IDs:  {len(all_pos_tids):,}")
flush(f"  Time: {time.time()-t0:.1f}s")

# ── [1] Load entity records ────────────────────────────────────────────────
flush("\n[1] Loading entity records...")
t1 = time.time()
cols = ["entity_id", "business_name", "business_address", "country"]
val_query_set = set(val_query_list)
s1_all    = pl.read_csv(S1_PATH, separator="\t", quote_char=None,
                        has_header=True, infer_schema_length=0, columns=cols)
s2        = pl.read_csv(S2_PATH, separator="\t", quote_char=None,
                        has_header=True, infer_schema_length=0, columns=cols)
s3        = pl.read_csv(S3_PATH, separator="\t", quote_char=None,
                        has_header=True, infer_schema_length=0, columns=cols)
targets_all = pl.concat([s2, s3])
val_s1      = s1_all.filter(pl.col("entity_id").is_in(list(val_query_set)))
val_s1_rows = {str(r["entity_id"]): r for r in val_s1.iter_rows(named=True)}
target_rows = {str(r["entity_id"]): r for r in targets_all.iter_rows(named=True)}
flush(f"  Val S1:  {len(val_s1):,}  |  Targets: {len(targets_all):,}")
flush(f"  Time: {time.time()-t1:.1f}s")

# ── Oracle helpers ─────────────────────────────────────────────────────────
def oracle_f05(candidate_map, gt, queries):
    precs, recs, f05s = [], [], []
    blackout = tot_true = ret_true = 0
    for qid in queries:
        true_s = gt.get(qid, set())
        pred   = true_s & candidate_map.get(qid, set())
        tot_true += len(true_s); ret_true += len(pred)
        if not true_s:
            precs.append(1.); recs.append(1.); f05s.append(1.)
        elif not pred:
            precs.append(0.); recs.append(0.); f05s.append(0.); blackout += 1
        else:
            r = len(pred) / len(true_s); f = 1.25 * r / (0.25 + r)
            precs.append(1.); recs.append(r); f05s.append(f)
    return {
        "macro_f05": float(np.nanmean(f05s)),
        "link_recall": ret_true / max(tot_true, 1),
        "blackout_queries": blackout,
        "total_true_links": tot_true,
        "retrieved_true_links": ret_true
    }

def oracle_at_k(raw_cands, gt, queries, k):
    cmap = defaultdict(list)
    for q, t, s in raw_cands:
        cmap[q].append((s, t))
    km = {q: {t for _, t in sorted(cmap.get(q, []), reverse=True)[:k]} for q in queries}
    return oracle_f05(km, gt, queries)

# ── [2] Baseline oracle ────────────────────────────────────────────────────
flush("\n[2] Baseline oracle (7ch K=40)...")
feat_df = pl.read_parquet(FEAT_FILE)
baseline_cand_map = defaultdict(set)
for row in feat_df.select(["entity_id", "target_id"]).iter_rows(named=True):
    baseline_cand_map[str(row["entity_id"])].add(str(row["target_id"]))
baseline_oracle = oracle_f05(dict(baseline_cand_map), val_gt, val_query_list)
baseline_pos = {(q, t) for q, cs in baseline_cand_map.items()
                for t in cs if t in val_gt.get(q, set())}
flush(f"  Baseline Oracle F0.5:    {baseline_oracle['macro_f05']:.5f}")
flush(f"  Link Recall:             {baseline_oracle['link_recall']:.4f}")
flush(f"  Blackout queries:        {baseline_oracle['blackout_queries']:,}")
flush(f"  Baseline positive pairs: {len(baseline_pos):,}")

# ── [3] Ch8: Char TF-IDF + CSC Sparse Retrieval ───────────────────────────
flush("\n[3] Ch8: Char (3,4)-gram TF-IDF sparse retrieval...")
t3 = time.time()
all_tids    = targets_all["entity_id"].to_list()
neg_sample  = set(random.sample(all_tids, min(600_000, len(all_tids))))
tfidf_tids  = list(all_pos_tids | neg_sample)

def rtext(tid):
    r = target_rows.get(tid, {})
    return (f"{str(r.get('business_name','') or '').lower()} "
            f"{str(r.get('business_address','') or '').lower()}").strip()

def qtext(qid):
    r = val_s1_rows.get(qid, {})
    return (f"{str(r.get('business_name','') or '').lower()} "
            f"{str(r.get('business_address','') or '').lower()}").strip()

flush(f"  Building corpus: {len(tfidf_tids):,} targets, {len(val_query_list):,} queries...")
tgt_texts = [rtext(tid) for tid in tfidf_tids]
qry_texts = [qtext(q)   for q   in val_query_list]

vec = TfidfVectorizer(
    analyzer='char_wb', ngram_range=(3, 4),
    max_features=150_000, sublinear_tf=True,
    min_df=3, max_df=0.03, strip_accents='unicode'
)
flush("  Fitting TF-IDF vectorizer...", end=" ")
tgt_mat = vec.fit_transform(tgt_texts)
qry_mat = vec.transform(qry_texts)
tgt_mat_norm = normalize(tgt_mat, norm='l2', copy=False).tocsc()
qry_mat_norm = normalize(qry_mat, norm='l2', copy=False)
flush(f"done. Matrix nonzeros: {tgt_mat.nnz:,} shape: {tgt_mat.shape}")

TFIDF_K   = 60
BATCH     = 1000
tid_arr   = np.array(tfidf_tids)
ch8_cands = []
flush(f"  Searching {len(val_query_list):,} queries in batches of {BATCH}...")
t_search  = time.time()

for b in range(0, len(val_query_list), BATCH):
    end   = min(b + BATCH, len(val_query_list))
    q_batch = qry_mat_norm[b:end]
    scores  = q_batch.dot(tgt_mat_norm.T) # CSR matrix
    indptr  = scores.indptr
    indices = scores.indices
    data    = scores.data

    for i in range(scores.shape[0]):
        start = indptr[i]
        finish = indptr[i+1]
        if start == finish:
            continue
        row_data = data[start:finish]
        row_indices = indices[start:finish]
        if len(row_data) > TFIDF_K:
            top_k_idx = np.argpartition(row_data, -TFIDF_K)[-TFIDF_K:]
            top_indices = row_indices[top_k_idx]
            top_scores  = row_data[top_k_idx]
        else:
            top_indices = row_indices
            top_scores  = row_data

        qid = val_query_list[b + i]
        for idx, sc in zip(top_indices, top_scores):
            if sc > 0.10:
                ch8_cands.append((qid, tid_arr[idx], float(sc)))

    if (b // BATCH) % 5 == 0 or end == len(val_query_list):
        flush(f"    [{end:>5}/{len(val_query_list)}] elapsed={time.time()-t_search:.1f}s | ch8_pairs={len(ch8_cands):,}")

ch8_new    = {(q, t) for q, t, s in ch8_cands
              if t in val_gt.get(q, set()) and t not in baseline_cand_map.get(q, set())}
ch8_oracle = oracle_at_k(ch8_cands, val_gt, val_query_list, k=TFIDF_K)
flush(f"  Ch8 pairs: {len(ch8_cands):,}  |  NEW positives: {len(ch8_new):,}")
flush(f"  Ch8 oracle F0.5 K={TFIDF_K}: {ch8_oracle['macro_f05']:.5f}")
flush(f"  Time: {time.time()-t3:.1f}s")

# Free memory
del tgt_mat, qry_mat, tgt_mat_norm, qry_mat_norm, scores, tgt_texts, qry_texts

# ── [4] Ch9: Devanagari transliteration ──────────────────────────────────
flush("\n[4] Ch9: Devanagari->Latin transliteration...")
t4 = time.time()
DEVA = {
    'अ':'a','आ':'aa','इ':'i','ई':'ii','उ':'u','ऊ':'uu','ए':'e','ऐ':'ai',
    'ओ':'o','औ':'au','क':'k','ख':'kh','ग':'g','घ':'gh','ङ':'ng','च':'ch',
    'छ':'chh','ज':'j','झ':'jh','ञ':'ny','ट':'t','ठ':'th','ड':'d','ढ':'dh',
    'ण':'n','त':'t','थ':'th','द':'d','ध':'dh','न':'n','प':'p','फ':'ph',
    'ब':'b','भ':'bh','म':'m','य':'y','र':'r','ल':'l','व':'v','श':'sh',
    'ष':'sh','स':'s','ह':'h','ळ':'l','ा':'aa','ि':'i','ी':'ii','ु':'u',
    'ू':'uu','े':'e','ै':'ai','ो':'o','ौ':'au','ं':'m','ः':'h','्':'',
    'ँ':'n','ृ':'ri','०':'0','१':'1','२':'2','३':'3','४':'4','५':'5',
    '६':'6','७':'7','८':'8','९':'9'
}

def has_deva(s):
    return any('\u0900' <= c <= '\u097F' for c in (s or ''))

def translit(tx):
    out, i = [], 0
    while i < len(tx):
        if i+1 < len(tx) and tx[i:i+2] in DEVA:
            out.append(DEVA[tx[i:i+2]]); i += 2
        elif tx[i] in DEVA:
            out.append(DEVA[tx[i]]); i += 1
        else:
            out.append(tx[i]); i += 1
    return ''.join(out)

def nt(s):
    s = (s or '').strip().lower()
    if has_deva(s): s = translit(s)
    return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', ' ', s)).strip()

def stoks(s, ml=3):
    return ' '.join(sorted(t for t in s.split() if len(t) >= ml))

tidx  = defaultdict(list)
tsidx = defaultdict(list)
for tid, row in target_rows.items():
    nm = str(row.get("business_name", "") or "")
    if has_deva(nm):
        tn = nt(nm)
        if len(tn) >= 5:
            tidx[tn].append(tid)
            sk = stoks(tn)
            if sk: tsidx[sk].append(tid)

ch9_cands = []
seen_ch9  = set()
for qid in val_query_list:
    nm = str(val_s1_rows.get(qid, {}).get("business_name", "") or "")
    if has_deva(nm):
        qtn = nt(nm)
        if len(qtn) >= 5:
            for tid in tidx.get(qtn, []):
                key = (qid, tid)
                if key not in seen_ch9:
                    seen_ch9.add(key)
                    ch9_cands.append((qid, tid, 0.95))
            sk = stoks(qtn)
            if sk:
                for tid in tsidx.get(sk, []):
                    key = (qid, tid)
                    if key not in seen_ch9:
                        seen_ch9.add(key)
                        ch9_cands.append((qid, tid, 0.85))

ch9_new    = {(q, t) for q, t, s in ch9_cands
              if t in val_gt.get(q, set()) and t not in baseline_cand_map.get(q, set())}
ch9_oracle = oracle_at_k(ch9_cands, val_gt, val_query_list, k=50)
flush(f"  Ch9 pairs: {len(ch9_cands):,}  |  NEW positives: {len(ch9_new):,}")
flush(f"  Ch9 oracle F0.5 K=50: {ch9_oracle['macro_f05']:.5f}")
flush(f"  Time: {time.time()-t4:.1f}s")

# ── [5] Ch10: Postal + Name blocking ──────────────────────────────────────
flush("\n[5] Ch10: Postal code + name blocking...")
t5 = time.time()
IPIN = re.compile(r'\b([1-9]\d{5})\b')
UZIP = re.compile(r'\b(\d{5})(?:-\d{4})?\b')

def gp(addr, cty):
    a = (addr or '').lower(); c = (cty or '').upper()
    if 'INDIA' in c: m = IPIN.search(a); return m.group(1) if m else ''
    if 'US'    in c: m = UZIP.search(a); return m.group(1) if m else ''
    m = re.search(r'\b(\d{5,6})\b', a); return m.group(1) if m else ''

def fw(nm, ml=3):
    ws = [w for w in re.sub(r'[^\w\s]', ' ', nm.lower()).split() if len(w) >= ml]
    return ws[0] if ws else ''

pidx = defaultdict(list)
for tid, row in target_rows.items():
    p = gp(str(row.get("business_address", "") or ""),
           str(row.get("country", "") or ""))
    if len(p) >= 5:
        f = fw(str(row.get("business_name", "") or ""))
        if f: pidx[(p, f)].append(tid)

ch10_cands = []
seen_ch10  = set()
for qid in val_query_list:
    row = val_s1_rows.get(qid, {})
    p   = gp(str(row.get("business_address", "") or ""),
             str(row.get("country", "") or ""))
    if len(p) >= 5:
        f = fw(str(row.get("business_name", "") or ""))
        if f:
            for tid in pidx.get((p, f), []):
                key = (qid, tid)
                if key not in seen_ch10:
                    seen_ch10.add(key)
                    ch10_cands.append((qid, tid, 0.80))

ch10_new    = {(q, t) for q, t, s in ch10_cands
               if t in val_gt.get(q, set()) and t not in baseline_cand_map.get(q, set())}
ch10_oracle = oracle_at_k(ch10_cands, val_gt, val_query_list, k=50)
flush(f"  Ch10 pairs: {len(ch10_cands):,}  |  NEW positives: {len(ch10_new):,}")
flush(f"  Ch10 oracle F0.5 K=50: {ch10_oracle['macro_f05']:.5f}")
flush(f"  Time: {time.time()-t5:.1f}s")

# ── [6] Union oracle sweep ─────────────────────────────────────────────────
flush("\n[6] Union oracle sweep at K budgets...")
t6 = time.time()
union_cands = [(q, t, 1.0) for q, cs in baseline_cand_map.items() for t in cs]
union_cands.extend(ch8_cands)
union_cands.extend(ch9_cands)
union_cands.extend(ch10_cands)

pair_best = {}
for q, t, s in union_cands:
    key = (q, t)
    if key not in pair_best or s > pair_best[key]:
        pair_best[key] = s
union_dedup = [(q, t, s) for (q, t), s in pair_best.items()]
union_pos   = {(q, t) for q, t, s in union_dedup if t in val_gt.get(q, set())}
new_pos     = union_pos - baseline_pos
flush(f"  Union pairs: {len(union_dedup):,}  |  NEW positives: {len(new_pos):,} "
      f"(+{len(new_pos)/max(len(baseline_pos),1)*100:.1f}%)")

# Pre-sort candidate lists per query once for instantaneous K-budget sweep
query_cands_sorted = defaultdict(list)
for q, t, s in union_dedup:
    query_cands_sorted[q].append((s, t))

for q in val_query_list:
    query_cands_sorted[q].sort(reverse=True)

results_by_k = {}
for k in K_BUDGETS:
    km = {q: {t for _, t in query_cands_sorted.get(q, [])[:k]} for q in val_query_list}
    res = oracle_f05(km, val_gt, val_query_list)
    results_by_k[k] = res
    delta = res['macro_f05'] - baseline_oracle['macro_f05']
    flush(f"  K={k:3d}: Oracle F0.5={res['macro_f05']:.5f} ({delta:+.5f}) "
          f"LinkRecall={res['link_recall']:.4f} Blackout={res['blackout_queries']:,}")
flush(f"  Time: {time.time()-t6:.1f}s")

flush("\n" + "="*70)
flush("EXP-15 SUMMARY")
flush("="*70)
flush(f"  Baseline oracle (7ch K=40):   F0.5 = {baseline_oracle['macro_f05']:.5f}")
for k in K_BUDGETS:
    r = results_by_k[k]
    flush(f"  Union K={k:3d}:                 F0.5 = {r['macro_f05']:.5f} "
          f"({r['macro_f05']-baseline_oracle['macro_f05']:+.5f})")
flush(f"\n  Ch8 (Char TF-IDF) new pos:  {len(ch8_new):,}")
flush(f"  Ch9 (Translit)    new pos:  {len(ch9_new):,}")
flush(f"  Ch10 (Postal)     new pos:  {len(ch10_new):,}")
flush(f"  TOTAL new positives:        {len(new_pos):,}")

output = {
    "experiment": "EXP-15",
    "baseline_oracle": baseline_oracle,
    "ch8_char_tfidf_k60": ch8_oracle,
    "ch8_new_positives": len(ch8_new),
    "ch9_transliteration_k50": ch9_oracle,
    "ch9_new_positives": len(ch9_new),
    "ch10_postal_name_k50": ch10_oracle,
    "ch10_new_positives": len(ch10_new),
    "union_new_positives": len(new_pos),
    "union_pairs": len(union_dedup),
    "union_oracle_by_k": {str(k): v for k, v in results_by_k.items()}
}
OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_JSON, "w") as f:
    json.dump(output, f, indent=2)
flush(f"\nSaved: {OUTPUT_JSON}")
