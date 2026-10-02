"""
scratch/audit_cardinality_and_consensus.py
Analyze ground truth cardinality per source:
Does any query ever match >1 entity in S2 or >1 entity in S3?
"""
from pathlib import Path
import polars as pl
from collections import Counter

ROOT = Path("z:/Amazon ML")
VAL_SAMPLE = ROOT / "experiments" / "val_sample_50k.parquet"
val_df = pl.read_parquet(VAL_SAMPLE)

s2_counts = []
s3_counts = []
total_matches = []

for row in val_df.iter_rows(named=True):
    raw = str(row.get("matched_entity_ids", "") or "").strip()
    if not raw or raw in ("nan", "None", ""):
        total_matches.append(0)
        s2_counts.append(0)
        s3_counts.append(0)
        continue
    tids = [t.strip() for t in raw.split(",") if t.strip()]
    total_matches.append(len(tids))
    s2 = sum(1 for t in tids if t.startswith("S2-"))
    s3 = sum(1 for t in tids if t.startswith("S3-"))
    s2_counts.append(s2)
    s3_counts.append(s3)

print("=" * 60)
print("GROUND TRUTH CARDINALITY AUDIT (50,000 queries)")
print("=" * 60)
print("Total matches distribution:", Counter(total_matches))
print("S2 matches distribution:   ", Counter(s2_counts))
print("S3 matches distribution:   ", Counter(s3_counts))

joint = Counter(zip(s2_counts, s3_counts))
print("\nJoint (S2, S3) distribution:")
for (s2, s3), cnt in sorted(joint.items(), key=lambda x: -x[1]):
    pct = cnt / len(val_df) * 100
    print(f"  S2={s2}, S3={s3}: {cnt:>6,} queries ({pct:>5.1f}%)")
