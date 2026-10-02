import os
import json
import time
import polars as pl

print("Starting Train/Test Overlap & Leakage Audit...", flush=True)

DATASET_DIR = "dataset/raw"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")

start_time = time.time()

# 1. Load S1 train and test
print("Loading train and test S1...", flush=True)
s1_train = pl.read_csv(os.path.join(TRAIN_DIR, "train_source1.tsv"), separator="\t")
s1_test = pl.read_csv(os.path.join(TEST_DIR, "test_source1.tsv"), separator="\t")

print(f"S1 Train shape: {s1_train.shape}, S1 Test shape: {s1_test.shape}", flush=True)

# S1 ID Overlap
s1_train_ids = set(s1_train["entity_id"].to_list())
s1_test_ids = set(s1_test["entity_id"].to_list())
id_overlap = len(s1_train_ids.intersection(s1_test_ids))
print(f"S1 ID Overlap between train and test: {id_overlap}", flush=True)

# 2. Check S2 and S3 ID overlaps
print("Loading S2 and S3 IDs...", flush=True)
s2_train = pl.read_csv(os.path.join(TRAIN_DIR, "train_source2.tsv"), separator="\t", columns=["entity_id"])
s2_test = pl.read_csv(os.path.join(TEST_DIR, "test_source2.tsv"), separator="\t", columns=["entity_id"])
s2_id_overlap = len(set(s2_train["entity_id"].to_list()).intersection(set(s2_test["entity_id"].to_list())))
print(f"S2 ID Overlap: {s2_id_overlap}", flush=True)

s3_train = pl.read_csv(os.path.join(TRAIN_DIR, "train_source3.tsv"), separator="\t", columns=["entity_id"])
s3_test = pl.read_csv(os.path.join(TEST_DIR, "test_source3.tsv"), separator="\t", columns=["entity_id"])
s3_id_overlap = len(set(s3_train["entity_id"].to_list()).intersection(set(s3_test["entity_id"].to_list())))
print(f"S3 ID Overlap: {s3_id_overlap}", flush=True)

# 3. Content Overlap between Train S1 and Test S1
print("Analyzing content overlap between Train S1 and Test S1...", flush=True)
# Lowercase & strip
s1_train_clean = s1_train.with_columns([
    pl.col("business_name").fill_null("").str.to_lowercase().str.strip_chars().alias("clean_name"),
    pl.col("business_address").fill_null("").str.to_lowercase().str.strip_chars().alias("clean_addr"),
    pl.col("country").fill_null("").str.to_lowercase().str.strip_chars().alias("clean_country")
])

s1_test_clean = s1_test.with_columns([
    pl.col("business_name").fill_null("").str.to_lowercase().str.strip_chars().alias("clean_name"),
    pl.col("business_address").fill_null("").str.to_lowercase().str.strip_chars().alias("clean_addr"),
    pl.col("country").fill_null("").str.to_lowercase().str.strip_chars().alias("clean_country")
])

# Exact tuple match (name, addr, country)
train_tuples = set(zip(s1_train_clean["clean_name"], s1_train_clean["clean_addr"], s1_train_clean["clean_country"]))
test_tuples = list(zip(s1_test_clean["clean_name"], s1_test_clean["clean_addr"], s1_test_clean["clean_country"]))

exact_tuple_overlap = sum(1 for t in test_tuples if t in train_tuples)
print(f"Exact (name, address, country) S1 Overlap: {exact_tuple_overlap} / {len(test_tuples)} ({exact_tuple_overlap/len(test_tuples)*100:.3f}%)", flush=True)

# Exact (name, country) match
train_name_country = set(zip(s1_train_clean["clean_name"], s1_train_clean["clean_country"]))
test_name_country = list(zip(s1_test_clean["clean_name"], s1_test_clean["clean_country"]))

name_country_overlap = sum(1 for nc in test_name_country if nc in train_name_country)
print(f"Exact (name, country) S1 Overlap: {name_country_overlap} / {len(test_name_country)} ({name_country_overlap/len(test_name_country)*100:.3f}%)", flush=True)

# 4. Check Non-ASCII characters across entire test S1, S2, S3
print("Auditing non-ASCII / Indic / Accents in Test data...", flush=True)
import unicodedata

def non_ascii_profile(series, name):
    non_ascii_count = 0
    indic_count = 0
    accented_latin_count = 0
    total = len(series)
    for s in series:
        if s is None:
            continue
        has_na = False
        has_ind = False
        has_acc = False
        for ch in s:
            if ord(ch) > 127:
                has_na = True
                # Indic script blocks (Devanagari: 0900-097F, Bengali, Gurmukhi, Gujarati, etc.)
                if 0x0900 <= ord(ch) <= 0x0D7F:
                    has_ind = True
                elif unicodedata.category(ch) in ('Mn', 'Mc', 'Me') or 'LATIN' in unicodedata.name(ch, ''):
                    has_acc = True
        if has_na:
            non_ascii_count += 1
        if has_ind:
            indic_count += 1
        if has_acc:
            accented_latin_count += 1
    return {
        "field": name,
        "total": total,
        "non_ascii_records": non_ascii_count,
        "non_ascii_pct": round(non_ascii_count / total * 100, 3),
        "indic_script_records": indic_count,
        "indic_script_pct": round(indic_count / total * 100, 3),
        "accented_latin_records": accented_latin_count,
        "accented_latin_pct": round(accented_latin_count / total * 100, 3)
    }

test_s1_name_prof = non_ascii_profile(s1_test["business_name"].to_list()[:50000], "test_s1_name_sample50k")
test_s1_addr_prof = non_ascii_profile(s1_test["business_address"].to_list()[:50000], "test_s1_addr_sample50k")

results = {
    "s1_train_count": len(s1_train),
    "s1_test_count": len(s1_test),
    "s1_id_overlap": id_overlap,
    "s2_id_overlap": s2_id_overlap,
    "s3_id_overlap": s3_id_overlap,
    "exact_name_addr_country_overlap_count": exact_tuple_overlap,
    "exact_name_addr_country_overlap_pct": round(exact_tuple_overlap / len(test_tuples) * 100, 4),
    "exact_name_country_overlap_count": name_country_overlap,
    "exact_name_country_overlap_pct": round(name_country_overlap / len(test_name_country) * 100, 4),
    "test_s1_name_char_profile": test_s1_name_prof,
    "test_s1_addr_char_profile": test_s1_addr_prof,
    "elapsed_seconds": round(time.time() - start_time, 2)
}

output_path = "research_scripts/07_leakage_and_script_audit_results.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"Results written to {output_path} in {results['elapsed_seconds']}s", flush=True)
