"""
Data Preprocessing Script for Project Samarth
Merges crop production and rainfall datasets
"""

import pandas as pd
from pathlib import Path
import os

# Define paths
DATA_DIR = Path("data")
OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("🌾 Data Preprocessing - Project Samarth")
print("=" * 60)

# Step 1: Load crop production data
print("\n📊 Step 1: Loading crop production data...")
crop_file = DATA_DIR / "India Agriculture Crop Production.csv"
df_crop = pd.read_csv(crop_file)
print(f"✅ Loaded crop data: {df_crop.shape}")
print(f"Columns: {list(df_crop.columns)}")

# Step 2: Load rainfall data
print("\n🌧️ Step 2: Loading rainfall data...")
rainfall_file = DATA_DIR / "Sub_Division_IMD_2017.csv"
df_rain = pd.read_csv(rainfall_file)
print(f"✅ Loaded rainfall data: {df_rain.shape}")
print(f"Columns: {list(df_rain.columns)}")

# Step 3: Clean crop data
print("\n🧹 Step 3: Cleaning crop production data...")
df_crop.columns = [col.strip() for col in df_crop.columns]
# Rename to lowercase for consistency
df_crop.columns = [col.lower().replace(" ", "_") for col in df_crop.columns]
print(f"Cleaned columns: {list(df_crop.columns)}")

# Step 4: Clean rainfall data
print("\n🧹 Step 4: Cleaning rainfall data...")
df_rain.columns = [col.strip().upper() for col in df_rain.columns]
# Normalize column names
df_rain.rename(columns={
    "SUBDIVISION": "subdivision",
    "YEAR": "year",
    "ANNUAL": "annual_rainfall"
}, inplace=True)

# Drop rows with missing critical data
df_rain = df_rain.dropna(subset=["subdivision", "year", "annual_rainfall"])
df_rain["year"] = pd.to_numeric(df_rain["year"], errors="coerce").astype(int)
df_rain = df_rain[df_rain["year"] > 1900]  # Filter valid years
print(f"✅ Cleaned rainfall data: {df_rain.shape}")

# Step 5: Map states to IMD subdivisions
print("\n🗺️ Step 5: Mapping states to IMD subdivisions...")
state_to_subdivision = {
    "Andaman and Nicobar Islands": "Andaman & Nicobar Islands",
    "Arunachal Pradesh": "Arunachal Pradesh",
    "Assam": "Assam & Meghalaya",
    "Meghalaya": "Assam & Meghalaya",
    "Nagaland": "Naga Mani Mizo Tripura",
    "Manipur": "Naga Mani Mizo Tripura",
    "Mizoram": "Naga Mani Mizo Tripura",
    "Tripura": "Naga Mani Mizo Tripura",
    "West Bengal": "Gangetic West Bengal",
    "Sikkim": "Sub Himalayan West Bengal & Sikkim",
    "Odisha": "Orissa",
    "Jharkhand": "Jharkhand",
    "Bihar": "Bihar",
    "Uttar Pradesh": "East Uttar Pradesh",
    "Uttarakhand": "West Uttar Pradesh",
    "Haryana": "Haryana Delhi Chandigarh",
    "Delhi": "Haryana Delhi Chandigarh",
    "Punjab": "Punjab",
    "Himachal Pradesh": "Himachal Pradesh",
    "Jammu and Kashmir": "Jammu & Kashmir",
    "Chhattisgarh": "Chhattisgarh",
    "Madhya Pradesh": "West Madhya Pradesh",
    "Gujarat": "Saurashtra Kutch & Diu",
    "Maharashtra": "Konkan & Goa",
    "Goa": "Konkan & Goa",
    "Andhra Pradesh": "Coastal Andhra Pradesh",
    "Telangana": "Telangana",
    "Karnataka": "South Interior Karnataka",
    "Kerala": "Kerala",
    "Tamil Nadu": "Tamil Nadu",
    "Rajasthan": "East Rajasthan"
}

# Add subdivision column to crop data
df_crop["subdivision"] = df_crop["state"].map(state_to_subdivision)

# Extract year from crop data (handling formats like "2001-02")
def extract_year(year_str):
    if pd.isna(year_str):
        return None
    year_str = str(year_str)
    # Extract first 4 digits
    import re
    match = re.search(r'(\d{4})', year_str)
    if match:
        return int(match.group(1))
    return None

df_crop["year_cleaned"] = df_crop["year"].apply(extract_year)
df_crop = df_crop.dropna(subset=["year_cleaned", "subdivision"])

print(f"✅ Mapped subdivisions. Crop data shape: {df_crop.shape}")

# Step 6: Merge datasets
print("\n🔗 Step 6: Merging crop and rainfall data...")
df_merged = pd.merge(
    df_crop,
    df_rain[["subdivision", "year", "annual_rainfall"]],
    left_on=["subdivision", "year_cleaned"],
    right_on=["subdivision", "year"],
    how="inner"
)

# Select and rename columns for final output
final_columns = ["state", "district", "year_cleaned", "crop", "production", "yield", "annual_rainfall"]
# Ensure all columns exist
available_columns = [col for col in final_columns if col in df_merged.columns]
df_merged = df_merged[available_columns].copy()

# Clean up the data
df_merged = df_merged.dropna(subset=["state", "district", "crop", "year_cleaned"])
df_merged["year_cleaned"] = df_merged["year_cleaned"].astype(int)

print(f"✅ Merged dataset shape: {df_merged.shape}")
print(f"Columns: {list(df_merged.columns)}")

# Step 7: Save merged dataset
print("\n💾 Step 7: Saving merged dataset...")
output_file = OUTPUT_DIR / "merged_agri_climate_data.csv"
df_merged.to_csv(output_file, index=False)
print(f"✅ Saved merged dataset to: {output_file}")

# Summary statistics
print("\n" + "=" * 60)
print("📈 Summary Statistics")
print("=" * 60)
print(f"Total records: {len(df_merged):,}")
print(f"Unique states: {df_merged['state'].nunique()}")
print(f"Unique districts: {df_merged['district'].nunique()}")
print(f"Unique crops: {df_merged['crop'].nunique()}")
print(f"Year range: {df_merged['year_cleaned'].min()} - {df_merged['year_cleaned'].max()}")
print(f"\nSample records:")
print(df_merged.head(10).to_string())

print("\n✅ Data preprocessing complete!")

