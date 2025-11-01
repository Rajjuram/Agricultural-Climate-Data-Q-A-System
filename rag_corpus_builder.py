"""
RAG Corpus Builder for Project Samarth
Converts merged agricultural and climate data into text corpus for RAG
"""

import os
import json
from pathlib import Path
import pandas as pd
import hashlib
from datetime import datetime, timezone

print("=" * 60)
print("📚 RAG Corpus Builder - Project Samarth")
print("=" * 60)

# Load the merged dataset
DATA_FILE = Path("data") / "merged_agri_climate_data.csv"
if not DATA_FILE.exists():
    print(f"❌ Error: {DATA_FILE} not found!")
    print("Please run data_preprocessing.py first to generate the merged dataset.")
    exit(1)

print(f"\n📖 Loading merged dataset from: {DATA_FILE}")
df_merged = pd.read_csv(DATA_FILE)
print(f"✅ Loaded merged dataset: {df_merged.shape}")

# Create output directory
OUT_DIR = Path("rag_files")
OUT_DIR.mkdir(exist_ok=True)
print(f"✅ Output directory: {OUT_DIR}")


jsonl_path = OUT_DIR / "rag_corpus.jsonl"
sample_csv_path = OUT_DIR / "rag_corpus_sample.csv"

def mkid(text):
    # stable short id based on text hash
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]

print(f"\n🔄 Converting {len(df_merged)} records to RAG corpus...")

rows_written = 0
rows_skipped = 0

with open(jsonl_path, "w", encoding="utf-8") as fout:
    for idx, row in df_merged.iterrows():
        try:
            # Extract data fields
            year = int(row["year_cleaned"]) if pd.notna(row.get("year_cleaned")) else None
            state = str(row.get("state", "")).strip()
            district = str(row.get("district", "")).strip()
            crop = str(row.get("crop", "")).strip()
            production = row.get("production", None)
            yield_val = row.get("yield", None)
            rainfall = row.get("annual_rainfall", None)

            # Skip if essential fields are missing
            if not state or not district or not crop or year is None:
                rows_skipped += 1
                continue

            # Build factual text - keep it short and informative (1-2 sentences)
            parts = []
            if year:
                parts.append(f"In {year},")
            else:
                parts.append("In an available year,")

            parts.append(f"the district {district} in {state}")
            
            if crop:
                if pd.notna(production):
                    parts.append(f"produced {production:.2f} tonnes of {crop}")
                else:
                    parts.append(f"reported production for {crop}")
            else:
                parts.append("reported production")

            if pd.notna(yield_val):
                parts.append(f"with yield {yield_val:.2f} tonnes per hectare")
            
            if pd.notna(rainfall):
                parts.append(f"and recorded {rainfall:.1f} mm annual rainfall.")

            text = " ".join(parts)
            text = text.strip()

            # Build metadata
            metadata = {
                "state": state,
                "district": district,
                "year": int(year) if year else None,
                "crop": crop,
                "production": float(production) if pd.notna(production) else None,
                "yield": float(yield_val) if pd.notna(yield_val) else None,
                "annual_rainfall": float(rainfall) if pd.notna(rainfall) else None,
                "source": "Ministry of Agriculture (crop) & IMD (rainfall)",
                "created_at": datetime.now(timezone.utc).isoformat()
            }

            doc = {
                "id": mkid(f"{state}|{district}|{crop}|{year}|{production}"),
                "text": text,
                "metadata": metadata
            }

            fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
            rows_written += 1
            
            # Progress indicator
            if rows_written % 10000 == 0:
                print(f"  Processed {rows_written:,} records...")

        except Exception as e:
            rows_skipped += 1
            if rows_written < 10:  # Only print first few errors
                print(f"  Warning: Skipped row {idx}: {str(e)}")

# Save a small CSV sample for quick inspection
print(f"\n💾 Creating sample CSV...")
sample = []
with open(jsonl_path, "r", encoding="utf-8") as fin:
    for i, line in enumerate(fin):
        if i >= 200:
            break
        sample.append(json.loads(line))

sample_df = pd.DataFrame([{"id": s["id"], "text": s["text"], **s["metadata"]} for s in sample])
sample_df.to_csv(sample_csv_path, index=False)

print("=" * 60)
print("✅ RAG Corpus Creation Complete!")
print("=" * 60)
print(f"✅ Created RAG corpus JSONL: {jsonl_path}")
print(f"✅ Sample CSV saved: {sample_csv_path}")
print(f"\n📊 Statistics:")
print(f"  Documents written: {rows_written:,}")
print(f"  Documents skipped: {rows_skipped:,}")
print(f"\n📝 Sample texts (first 5):")
for i, r in enumerate(sample_df["text"].head(5).tolist(), 1):
    print(f"  {i}. {r}")
print("\n✅ Done!")
