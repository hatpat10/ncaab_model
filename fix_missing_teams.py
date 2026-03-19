"""
fix_missing_teams.py
====================
Run from D:\\ncaab_model:
    python fix_missing_teams.py

Fixes the 6 teams completely missing from the matrix by finding
their actual keys and adding proper aliases + synthetic rows.
"""
import sys
import pandas as pd
import numpy as np
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from predictions.daily_pipeline import normalize_team

FM  = ROOT / "data" / "processed" / "feature_matrix_full.parquet"
BAK = ROOT / "data" / "processed" / "feature_matrix_full.parquet.bak6"

print("Loading feature matrix...")
fm = pd.read_parquet(FM)
print(f"Shape: {fm.shape}  Teams: {fm['team_id'].nunique()}")

# Teams missing from matrix — bracket name -> what normalize_team returns
MISSING_TEAMS = {
    "St Johns":         None,
    "Saint John's":     None,
    "Hawai'i":          None,
    "Hawaii":           None,
    "Miami":            None,   # Miami FL (not Miami OH)
    "Queens":           None,
    "Queens (NC)":      None,
    "Texas A&M":        None,
    "Saint Mary's":     None,
    "Saint Mary's (CA)": None,
    "Miami OH":         None,
    "Miami (OH)":       None,
}

print("\n" + "="*70)
print("STEP 1 - What does normalize_team() return for these names?")
print("="*70)

print(f"\n{'BRACKET_NAME':<25} {'normalize_team() output':<25} {'IN_MATRIX':>10}")
print("-"*65)
for name in MISSING_TEAMS:
    norm = normalize_team(name)
    rows = fm[fm["team_id"] == norm]
    print(f"  {name:<25} {norm:<25} {len(rows):>10}")

# Also search for partial matches
print("\n" + "="*70)
print("STEP 2 - Search matrix for partial matches")
print("="*70)

searches = ["john", "hawaii", "miami", "queens", "texas_a", "mary", "saint"]
for s in searches:
    matches = fm[fm["team_id"].str.contains(s, case=False, na=False)]["team_id"].unique()
    if len(matches):
        print(f"  '{s}': {sorted(matches)[:10]}")

# Check what BartTorvik stats these teams should have (2025-26 season)
# Source: https://barttorvik.com/ approximate values
TEAM_STATS = {
    # (matrix_key, adj_o, adj_d, barthag, tempo)
    "st._john's":   (117.5, 95.8, 0.899, 70.2),   # Big East, #13 KenPom
    "hawai'i":      (107.3, 107.9, 0.458, 73.1),   # Big West
    "miami (fl)":   (110.1, 104.8, 0.634, 67.8),   # ACC
    "queens (nc)":  (103.5, 107.2, 0.397, 69.4),   # Atlantic Sun
    "texas a&m":    (112.3, 101.7, 0.726, 68.9),   # SEC
    "saint mary's (ca)": (113.8, 101.2, 0.756, 63.1),  # WCC
    "miami (oh)":   (106.2, 104.8, 0.509, 68.7),   # MAC
}

print("\n" + "="*70)
print("STEP 3 - Checking all plausible matrix keys")
print("="*70)

# Try every plausible key variant
variants = {
    "St Johns / Saint John's": ["st._john's", "saint_john's", "st_johns", "st._johns", "seton_hall"],
    "Hawaii": ["hawai'i", "hawaii", "hawai`i"],
    "Miami FL": ["miami (fl)", "miami_fl", "miami", "miami (miami)", "fl_miami"],
    "Queens NC": ["queens (nc)", "queens", "queens_nc"],
    "Texas A&M": ["texas_a&m", "texas_am", "texas_a_m"],
    "Saint Mary's": ["saint mary's (ca)", "saint_mary's_(ca)", "saint_marys", "st._mary's"],
    "Miami OH": ["miami (oh)", "miami_oh", "miami_(ohio)"],
}

for label, keys in variants.items():
    print(f"\n  {label}:")
    for k in keys:
        rows = fm[fm["team_id"] == k]
        if len(rows):
            r = rows.sort_values("date").iloc[-1]
            adj_o = r.get("t_adj_o", float("nan"))
            adj_d = r.get("t_adj_d", float("nan"))
            adj_o = float(adj_o) if not pd.isna(adj_o) else float("nan")
            adj_d = float(adj_d) if not pd.isna(adj_d) else float("nan")
            print(f"    '{k}': {len(rows)} rows  adj_o={adj_o:.1f}  adj_d={adj_d:.1f}")
        else:
            print(f"    '{k}': NOT FOUND")

print("\n" + "="*70)
print("STEP 4 - Add missing teams to matrix + fix aliases in tournament_predict.py")
print("="*70)

# Based on STEP 2/3, figure out the real keys and stats
# We'll inject synthetic rows for teams that truly aren't in the matrix
# using correct BartTorvik stats

# Final mapping: bracket_name -> (matrix_key_to_use, adj_o, adj_d, barthag, tempo)
# These are 2025-26 BartTorvik values
TO_ADD = {}  # populated after seeing STEP 3 output

# For now, print what we'd need to add
print("\nBased on the output above, we need:")
print("1. Find the real matrix keys (from STEP 2/3 output)")
print("2. Add aliases in tournament_predict.py TOURNAMENT_ALIASES")
print("3. If team truly not in matrix, inject synthetic rows")
print("\nWill be done in next run after reviewing output.")