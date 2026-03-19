"""
fix_parquet_numerics.py
=======================
Run from D:\\ncaab_model:
    python fix_parquet_numerics.py

The previous backfill did .astype(str) on ALL object columns to satisfy pyarrow,
but this accidentally stringified numeric columns that pandas had categorized as
object dtype (e.g., 'margin', 'score', etc. stored as mixed types).

This script:
1. Identifies which numeric-looking columns got stringified
2. Converts them back to float where possible
3. Leaves truly categorical string columns (team_id, date, etc.) as str
4. Saves cleanly without corrupting dtypes

Also identifies which tournament teams have null features (model spread = 0.0).
"""

import pandas as pd
import numpy as np
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FM   = ROOT / "data" / "processed" / "feature_matrix_full.parquet"
BAK  = ROOT / "data" / "processed" / "feature_matrix_full.parquet.bak3"

# Known string columns that should stay as str
STRING_COLS = {
    "team_id", "game_id", "date", "home_team", "away_team",
    "season", "conference", "opponent", "game_type", "location",
    "neutral_site", "is_home", "is_neutral",
}

print("Loading feature matrix...")
fm = pd.read_parquet(FM)
print(f"Shape: {fm.shape}")

# ── Step 1: Find columns that look numeric but are stored as object/str ───────
print("\n" + "="*70)
print("STEP 1 — Finding stringified numeric columns")
print("="*70)

to_convert = []
truly_string = []

for col in fm.select_dtypes("object").columns:
    if col in STRING_COLS:
        truly_string.append(col)
        continue
    
    # Sample non-null values
    sample = fm[col].dropna()
    if len(sample) == 0:
        continue
    
    # Try converting a sample to float
    sample_strs = sample.astype(str).head(50)
    try:
        converted = pd.to_numeric(sample_strs, errors="raise")
        to_convert.append(col)
    except (ValueError, TypeError):
        # Check if it's mixed - some numeric some not
        n_numeric = pd.to_numeric(sample_strs, errors="coerce").notna().sum()
        if n_numeric > len(sample_strs) * 0.8:  # 80%+ numeric
            to_convert.append(col)
        else:
            truly_string.append(col)

print(f"\n  Columns to convert back to numeric ({len(to_convert)}): {to_convert}")
print(f"\n  True string columns ({len(truly_string)}): {truly_string}")

# ── Step 2: Check the specific problematic column ─────────────────────────────
print("\n" + "="*70)
print("STEP 2 — Checking h2h_avg_margin and margin columns specifically")
print("="*70)

margin_cols = [c for c in fm.columns if "margin" in c.lower() or "score" in c.lower()]
for col in margin_cols:
    dtype = fm[col].dtype
    sample = fm[col].dropna().head(5).tolist()
    print(f"  {col:<35} dtype={dtype}  sample={sample}")

# ── Step 3: Back up and fix ───────────────────────────────────────────────────
print("\n" + "="*70)
print("STEP 3 — Fixing column dtypes")
print("="*70)

shutil.copy2(FM, BAK)
print(f"  Backed up to {BAK}")

fm_fixed = fm.copy()
fixed_count = 0
for col in to_convert:
    try:
        fm_fixed[col] = pd.to_numeric(fm_fixed[col], errors="coerce")
        fixed_count += 1
    except Exception as e:
        print(f"  WARNING: Could not convert {col}: {e}")

print(f"  Converted {fixed_count} columns back to numeric")

# For remaining string columns, make sure they're proper strings (not mixed)
for col in truly_string:
    try:
        fm_fixed[col] = fm_fixed[col].astype(str).replace("nan", np.nan)
    except Exception:
        pass

# ── Step 4: Verify the margin column is fixed ─────────────────────────────────
print("\n  Verifying margin columns after fix:")
for col in margin_cols:
    dtype = fm_fixed[col].dtype
    sample = fm_fixed[col].dropna().head(3).tolist()
    print(f"  {col:<35} dtype={dtype}  sample={sample}")

# ── Step 5: Save ──────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("STEP 4 — Saving")
print("="*70)

fm_fixed.to_parquet(FM, index=False)
print(f"  Saved: {len(fm_fixed):,} rows")

# Verify it loads cleanly
fm_check = pd.read_parquet(FM)
print(f"  Verified: {len(fm_check):,} rows loaded cleanly")

# ── Step 6: Check which tournament teams have null key features ───────────────
print("\n" + "="*70)
print("STEP 5 — Tournament team feature check")
print("="*70)

TOURNAMENT_TEAMS = {
    # key in matrix -> (expected bracket name, expected barthag range)
    "duke":             ("Duke #1",          0.90),
    "arizona":          ("Arizona #1",       0.88),
    "michigan":         ("Michigan #1",      0.89),
    "florida":          ("Florida #1",       0.87),
    "uconn":            ("UConn #2",         0.85),
    "purdue":           ("Purdue #2",        0.84),
    "iowa_state":       ("Iowa State #2",    0.87),
    "houston":          ("Houston #2",       0.82),
    "michigan_st":      ("Michigan St #3",   0.84),
    "gonzaga":          ("Gonzaga #3",       0.83),
    "illinois":         ("Illinois #3",      0.82),
    "virginia":         ("Virginia #3",      0.80),
    "kansas":           ("Kansas #4",        0.81),
    "alabama":          ("Alabama #4",       0.80),
    "tennessee":        ("Tennessee #6",     0.83),
    "ohio_st":          ("Ohio St #8",       0.73),
    "california_baptist": ("Cal Baptist #13", 0.48),
    "n_dakota_st":      ("N Dakota St #14",  0.41),
    "tennessee_state":  ("Tennessee St #15", 0.33),
    "mcneese":          ("McNeese #12",      0.50),
    "kennesaw_st":      ("Kennesaw St #14",  0.39),
    "penn":             ("Penn #14",         0.44),
    "liu":              ("LIU #16",          0.30),
}

print(f"\n  {'MATRIX_KEY':<25} {'BRACKET_NAME':<20} {'ROWS':>6} {'t_adj_o':>8} {'t_adj_d':>8} {'t_barthag':>10} {'STATUS'}")
print("  " + "-"*95)

for key, (name, exp_barthag) in TOURNAMENT_TEAMS.items():
    rows = fm_check[fm_check["team_id"] == key]
    if len(rows) == 0:
        print(f"  {key:<25} {name:<20} {'MISSING':>6}  ❌ NOT IN MATRIX")
        continue
    
    r = rows.sort_values("date", ascending=False).iloc[0] if "date" in rows.columns else rows.iloc[0]
    
    adj_o   = float(r["t_adj_o"])   if "t_adj_o"   in r.index and not pd.isna(r["t_adj_o"])   else float("nan")
    adj_d   = float(r["t_adj_d"])   if "t_adj_d"   in r.index and not pd.isna(r["t_adj_d"])   else float("nan")
    barthag = float(r["t_barthag"]) if "t_barthag" in r.index and not pd.isna(r["t_barthag"]) else float("nan")
    
    has_features = not (np.isnan(adj_o) or np.isnan(adj_d))
    status = "✓" if has_features else "❌ NaN FEATURES"
    
    o_str = f"{adj_o:.1f}" if not np.isnan(adj_o) else "NaN"
    d_str = f"{adj_d:.1f}" if not np.isnan(adj_d) else "NaN"
    b_str = f"{barthag:.3f}" if not np.isnan(barthag) else "NaN"
    
    print(f"  {key:<25} {name:<20} {len(rows):>6}  {o_str:>8}  {d_str:>8}  {b_str:>10}  {status}")

print("""
======================================================================
NEXT STEPS
======================================================================
If teams above show NaN features, the feature matrix looks up features
by the MOST RECENT row for each team. The synthetic rows we added have
date='2026-03-15', but if build_game_features uses a different date
filter (e.g., only games up to target_date), it might skip synthetic rows.

Run: python tournament_predict.py --bracket bracket.json
If you still see model spread = 0.0 for some teams, the feature lookup
is filtering out the synthetic rows by date. We'll need to check
build_game_features in daily_pipeline.py.
""")