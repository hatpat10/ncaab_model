"""Run from project root: python diagnose_season.py"""
import pandas as pd
from pathlib import Path

RAW = Path("data/raw")

frames = []
for fname in ["barttorvik_2020_2023.parquet", "barttorvik_2024_2025.parquet"]:
    p = RAW / fname
    if p.exists():
        df = pd.read_parquet(p)
        frames.append(df)

torvik = pd.concat(frames, ignore_index=True)

print("BartTorvik 'year' values:", sorted(torvik["year"].dropna().unique().tolist()))
print(f"Total rows: {len(torvik)}")
print()

# Show what teams look like in BartTorvik for year=2025
y2025 = torvik[torvik["year"] == 2025]
print(f"year=2025 rows: {len(y2025)}")
if len(y2025):
    print("Sample teams (year=2025):", y2025["team"].head(20).tolist())
else:
    print("NO year=2025 data found!")
    print("Max year in data:", torvik["year"].max())
    print("All years:", sorted(torvik["year"].dropna().unique().tolist()))

print()
# Check teams that are failing the join — what do they look like in BartTorvik?
missing = ["South Carolina", "New Mexico", "Montana", "Kentucky", "Pacific",
           "IU Indianapolis", "Tennessee-Martin", "Saint Mary's", "Siena"]
print("BartTorvik team name samples for failing teams:")
for t in missing:
    match = torvik[torvik["team"].str.contains(t.split()[0], case=False, na=False)]["team"].unique()
    print(f"  '{t}' -> {match[:5].tolist()}")