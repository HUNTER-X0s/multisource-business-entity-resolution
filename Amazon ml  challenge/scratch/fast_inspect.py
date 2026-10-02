import polars as pl

print("Reading small slice of ground truth...")
gt = pl.read_csv("dataset/raw/train/train_ground_truth.tsv", separator="\t", n_rows=200)
gt = gt.filter(pl.col("matched_entity_ids").is_not_null() & (pl.col("matched_entity_ids") != ""))

# Get all referenced IDs
s1_needed = set(gt["source1_entity_id"].to_list())
mids_needed = set()
for raw in gt["matched_entity_ids"]:
    for m in raw.split(","):
        if m.strip(): mids_needed.add(m.strip())

print(f"Needed: {len(s1_needed)} S1 IDs, {len(mids_needed)} target IDs")

# Scan S1, S2, S3 using lazy scan
s1 = pl.scan_csv("dataset/raw/train/train_source1.tsv", separator="\t").filter(pl.col("entity_id").is_in(list(s1_needed))).collect()
s2 = pl.scan_csv("dataset/raw/train/train_source2.tsv", separator="\t").filter(pl.col("entity_id").is_in(list(mids_needed))).collect()
s3 = pl.scan_csv("dataset/raw/train/train_source3.tsv", separator="\t").filter(pl.col("entity_id").is_in(list(mids_needed))).collect()

targets = pl.concat([s2, s3])

s1_dict = {row["entity_id"]: row for row in s1.iter_rows(named=True)}
t_dict = {row["entity_id"]: row for row in targets.iter_rows(named=True)}

print("\n" + "="*80)
print("INSPECTING GROUND TRUTH MATCHES")
print("="*80)

for i, row in enumerate(gt.iter_rows(named=True)):
    if i >= 10: break
    qid = row["source1_entity_id"]
    mids = [x.strip() for x in row["matched_entity_ids"].split(",") if x.strip()]
    q = s1_dict.get(qid, {})
    print(f"\n--- [Example {i+1}] S1: {qid} ---")
    print(f"  NAME   : {q.get('business_name')}")
    print(f"  ADDR   : {q.get('business_address')}")
    print(f"  COUNTRY: {q.get('country')}")
    print(f"  Matches ({len(mids)}):")
    for mid in mids:
        t = t_dict.get(mid, {})
        print(f"    -> {mid}:")
        print(f"       NAME: {t.get('business_name')}")
        print(f"       ADDR: {t.get('business_address')}")
