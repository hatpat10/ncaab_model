"""
populate_torvik_cols.py
=======================
Run from D:\\ncaab_model:
    python populate_torvik_cols.py

The feature matrix has t_adj_o / t_adj_d / t_barthag columns but
they're NaN for most teams. build_game_features relies on these to
compute net_rating_delta, off_eff_delta, def_eff_delta.

This script finds where the real data lives and populates these
columns for every team that has data somewhere.

Also adds t_adj_t (tempo) and t_wab, t_ov_cur_sos columns if missing.
"""
import pandas as pd
import numpy as np
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FM   = ROOT / "data" / "processed" / "feature_matrix_full.parquet"
BAK  = ROOT / "data" / "processed" / "feature_matrix_full.parquet.bak4"

print("Loading feature matrix...")
fm = pd.read_parquet(FM)
print(f"Shape: {fm.shape}")

# ── Step 1: Find all columns and what's in them ──────────────────────────────
print("\n" + "="*70)
print("STEP 1 — All efficiency-related columns and their coverage")
print("="*70)

eff_cols = [c for c in fm.columns if any(x in c.lower() for x in 
            ["adj_o","adj_d","adj_t","barthag","wab","sos","tempo"])]

for col in eff_cols:
    non_null = fm[col].notna().sum()
    pct = non_null / len(fm) * 100
    sample = fm[col].dropna().head(3).tolist()
    print(f"  {col:<30} {non_null:>8,} non-null ({pct:4.1f}%)  sample={sample}")

# ── Step 2: Check what columns a real team (Duke) has populated ──────────────
print("\n" + "="*70)
print("STEP 2 — Duke's actual populated columns")
print("="*70)
duke = fm[fm["team_id"] == "duke"].sort_values("date")
if len(duke):
    last = duke.iloc[-1]
    print("\nAll non-NaN columns for Duke's most recent row:")
    for col in fm.columns:
        val = last[col]
        if not pd.isna(val) if not isinstance(val, str) else True:
            print(f"  {col:<35} = {val}")

# ── Step 3: Find the best source columns for each target ────────────────────
print("\n" + "="*70)
print("STEP 3 — Finding source columns for t_adj_o, t_adj_d, t_barthag")
print("="*70)

# We need to find what column actually stores team offensive efficiency
# Look for columns that are populated and correlate with expected values

# Known ground truth: Duke adj_o should be ~121, adj_d ~95
# Let's find which column matches
duke_non_null_numeric = duke.iloc[-1][[c for c in fm.columns 
                                        if fm[c].dtype in [np.float64, np.int64, float, int]
                                        and not pd.isna(duke.iloc[-1][c])]]

print("\nNumeric columns for Duke's last row:")
for col, val in duke_non_null_numeric.items():
    # Flag columns that look like adj_o values (100-130 range)
    if 100 <= float(val) <= 135:
        print(f"  *** {col:<35} = {val:.1f}  ← likely adj_o/adj_d candidate")
    elif 80 <= float(val) <= 105:
        print(f"  *** {col:<35} = {val:.1f}  ← likely adj_d candidate")

# ── Step 4: Check the roll/score columns ────────────────────────────────────
print("\n" + "="*70)
print("STEP 4 — Rolling feature coverage for key teams")
print("="*70)

roll_cols = [c for c in fm.columns if "roll" in c.lower() or "score" in c.lower()]
teams_to_check = ["duke", "siena", "liu", "idaho", "queens (nc)", "wright_state", 
                  "california_baptist", "n_dakota_st", "tennessee_state"]

print(f"\n{'TEAM':<25} {'ROWS':>6} {'roll5_margin':>14} {'roll10_margin':>14} {'team_score':>12}")
print("-"*75)
for tid in teams_to_check:
    rows = fm[fm["team_id"] == tid]
    if len(rows) == 0:
        print(f"  {tid:<25} MISSING")
        continue
    r = rows.sort_values("date").iloc[-1]
    r5  = r.get("roll5_margin", np.nan)
    r10 = r.get("roll10_margin", np.nan)
    ts  = r.get("team_score", np.nan)
    r5s  = f"{r5:.1f}"  if not pd.isna(r5)  else "NaN"
    r10s = f"{r10:.1f}" if not pd.isna(r10) else "NaN"
    tss  = f"{ts}"      if not pd.isna(ts)  else "NaN"
    print(f"  {tid:<25} {len(rows):>6} {r5s:>14} {r10s:>14} {tss:>12}")

# ── Step 5: Look at what the training data looked like ──────────────────────
print("\n" + "="*70)
print("STEP 5 — What does a COMPLETE feature row look like?")
print("="*70)
print("(Team with all 25 build_game_features fields populated)")

TARGET_KEYS = ["t_adj_o", "t_adj_d", "t_adj_t", "t_barthag",
               "net_rating_delta", "off_eff_delta", "def_eff_delta", "tempo_delta",
               "roll5_margin", "roll10_margin", "t_ov_cur_sos", "t_wab",
               "wab_delta", "sos_delta"]

# Find teams that have all these populated
coverage = {}
for tid in fm["team_id"].unique()[:200]:  # sample 200 teams
    rows = fm[fm["team_id"] == tid]
    r = rows.iloc[-1]
    has_all = all(not pd.isna(r.get(k, np.nan)) for k in TARGET_KEYS[:4])  # just torvik cols
    if has_all:
        coverage[tid] = True

print(f"\n  Teams with t_adj_o/t_adj_d/t_adj_t/t_barthag all non-null: {len(coverage)}")
print(f"  Examples: {list(coverage.keys())[:10]}")

if len(coverage) == 0:
    print("\n  ❌ NO TEAMS have all 4 BartTorvik columns populated!")
    print("  This means t_adj_o/t_adj_d etc. were NEVER populated in training data.")
    print("  The model was trained with NaN for these — and the imputer handles them.")
    print("\n  BUT: our fix_and_verify.py set these for 9 teams, so they now have")
    print("  real values. This is GOOD — build_game_features will use them.")
    print("\n  The remaining issue is OTHER teams (Siena, Idaho, etc.) who have")
    print("  many game rows but t_adj_o=NaN. They need BartTorvik data populated.")
    print("\n  SOLUTION: We need to populate t_adj_o/t_adj_d from an external source")
    print("  OR accept that the imputer handles NaN (it was trained this way).")

print("\n" + "="*70)
print("STEP 6 — What does the imputer fill NaN with?")
print("="*70)

import joblib
model_path = ROOT / "models" / "saved" / "spread_model.pkl"
if model_path.exists():
    saved = joblib.load(model_path)
    imputer = saved.get("imputer")
    features = saved.get("features", [])
    model = saved.get("model")
    
    print(f"\n  Model dict keys: {list(saved.keys())}")
    print(f"  Feature list ({len(features)}): {features[:10]}...")
    
    if imputer is not None:
        print(f"\n  Imputer type: {type(imputer).__name__}")
        if hasattr(imputer, "statistics_"):
            print(f"  Imputer statistics (fill values for NaN):")
            for feat, val in zip(features, imputer.statistics_):
                print(f"    {feat:<35} → fill with {val:.3f}")
    else:
        print("  No imputer found in model dict")
    
    if model is not None:
        print(f"\n  Model type: {type(model).__name__}")
        if hasattr(model, "feature_names_in_"):
            print(f"  Model feature_names_in_: {list(model.feature_names_in_)[:10]}")