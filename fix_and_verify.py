"""
fix_and_verify.py
=================
Run from D:\\ncaab_model:
    python fix_and_verify.py

Does three things:
1. Finds the REAL column names for adj_o, adj_d, barthag, tempo in the matrix
2. Rebuilds all synthetic rows using the correct column names
3. Verifies the fix worked and prints what features the models will actually see

Usage:
    python fix_and_verify.py          # dry run - show what will change
    python fix_and_verify.py --apply  # apply the fix
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

APPLY = "--apply" in sys.argv
ROOT  = Path(__file__).resolve().parent
FM    = ROOT / "data" / "processed" / "feature_matrix_full.parquet"
BAK   = ROOT / "data" / "processed" / "feature_matrix_full.parquet.bak2"

# ── Real stats (BartTorvik, Selection Sunday 2026) ──────────────────────────
TEAM_STATS = {
    # Synthetic teams injected earlier (wrong column names)
    "ohio_st":            {"adj_o": 116.8, "adj_d": 100.4, "barthag": 0.732, "tempo": 69.8},
    "california_baptist": {"adj_o": 108.4, "adj_d": 104.8, "barthag": 0.481, "tempo": 63.2},
    "michigan_st":        {"adj_o": 121.1, "adj_d":  96.3, "barthag": 0.841, "tempo": 69.4},
    "n_dakota_st":        {"adj_o": 105.2, "adj_d": 105.1, "barthag": 0.412, "tempo": 63.8},
    "uconn":              {"adj_o": 122.3, "adj_d":  97.1, "barthag": 0.856, "tempo": 68.3},
    "kennesaw_st":        {"adj_o": 104.1, "adj_d": 106.7, "barthag": 0.388, "tempo": 65.3},
    "penn":               {"adj_o": 107.9, "adj_d": 105.2, "barthag": 0.441, "tempo": 66.1},
    # Missing teams (not in matrix at all)
    "tennessee_state":    {"adj_o": 102.1, "adj_d": 107.3, "barthag": 0.331, "tempo": 68.1},
    "mcneese":            {"adj_o": 106.8, "adj_d": 103.2, "barthag": 0.501, "tempo": 71.2},
}

print("Loading feature matrix...")
fm = pd.read_parquet(FM)
print(f"Shape: {fm.shape}  |  Unique teams: {fm['team_id'].nunique()}")

# ── Step 1: Find real column names ──────────────────────────────────────────
print("\n" + "="*70)
print("STEP 1 — Finding real column names")
print("="*70)

# Look for adj_o variants
adj_o_cols   = [c for c in fm.columns if "adj_o" in c.lower() or "adjoe" in c.lower() or ("off_eff" in c.lower() and "delta" not in c.lower())]
adj_d_cols   = [c for c in fm.columns if "adj_d" in c.lower() or "adjde" in c.lower() or ("def_eff" in c.lower() and "delta" not in c.lower())]
barthag_cols = [c for c in fm.columns if "barthag" in c.lower() and "delta" not in c.lower() and "rank" not in c.lower()]
tempo_cols   = [c for c in fm.columns if "tempo" in c.lower() and "delta" not in c.lower()]

print(f"  adj_o variants:   {adj_o_cols}")
print(f"  adj_d variants:   {adj_d_cols}")
print(f"  barthag variants: {barthag_cols}")
print(f"  tempo variants:   {tempo_cols}")

# Pick the best column for each stat (prefer t_ prefix = team stats)
def pick_col(candidates, prefer_prefix="t_"):
    for c in candidates:
        if c.startswith(prefer_prefix):
            return c
    return candidates[0] if candidates else None

COL_ADJ_O   = pick_col(adj_o_cols)
COL_ADJ_D   = pick_col(adj_d_cols)
COL_BARTHAG = pick_col(barthag_cols)
COL_TEMPO   = pick_col(tempo_cols)

print(f"\n  → Using: adj_o='{COL_ADJ_O}'  adj_d='{COL_ADJ_D}'  barthag='{COL_BARTHAG}'  tempo='{COL_TEMPO}'")

# ── Step 2: Show what good teams look like ──────────────────────────────────
print("\n" + "="*70)
print("STEP 2 — Reference values from real teams")
print("="*70)

REFS = {"duke": (125.3, 94.8, 0.951), "arizona": (120.1, 96.4, 0.891), "iowa_state": (119.8, 97.2, 0.873)}
for team, (exp_o, exp_d, exp_b) in REFS.items():
    rows = fm[fm["team_id"] == team]
    if len(rows) == 0:
        print(f"  {team}: NOT FOUND")
        continue
    r = rows.sort_values("date", ascending=False).iloc[0]
    act_o = r.get(COL_ADJ_O, np.nan) if COL_ADJ_O else np.nan
    act_d = r.get(COL_ADJ_D, np.nan) if COL_ADJ_D else np.nan
    act_b = r.get(COL_BARTHAG, np.nan) if COL_BARTHAG else np.nan
    print(f"  {team:<20} adj_o={act_o:.1f} (exp {exp_o})  adj_d={act_d:.1f} (exp {exp_d})  barthag={act_b:.3f} (exp {exp_b})")

# ── Step 3: Verify/fix each synthetic team ──────────────────────────────────
print("\n" + "="*70)
print("STEP 3 — Checking and fixing synthetic teams")
print("="*70)

if not APPLY:
    print("  DRY RUN — run with --apply to save changes\n")

fm_updated = fm.copy()
teams_fixed = []
teams_added = []

for tid, stats in TEAM_STATS.items():
    rows = fm_updated[fm_updated["team_id"] == tid]
    
    if len(rows) == 0:
        status = "MISSING — will add synthetic rows"
        teams_added.append(tid)
    else:
        # Check if the key columns are NaN
        r = rows.iloc[0]
        cur_o = r.get(COL_ADJ_O, np.nan) if COL_ADJ_O else np.nan
        cur_d = r.get(COL_ADJ_D, np.nan) if COL_ADJ_D else np.nan
        
        if np.isnan(cur_o) or np.isnan(cur_d):
            status = f"EXISTS but features are NaN — will fix {len(rows)} rows"
            teams_fixed.append(tid)
        else:
            o_ok = abs(cur_o - stats["adj_o"]) < 5
            d_ok = abs(cur_d - stats["adj_d"]) < 5
            status = f"OK — adj_o={cur_o:.1f} adj_d={cur_d:.1f}" if (o_ok and d_ok) else f"WRONG — adj_o={cur_o:.1f} adj_d={cur_d:.1f} (expected {stats['adj_o']} / {stats['adj_d']})"
    
    print(f"  {tid:<25} {status}")

print(f"\n  Teams to fix (NaN features): {teams_fixed}")
print(f"  Teams to add (missing):      {teams_added}")

if not APPLY:
    print("\n  Run with --apply to apply fixes.")
    sys.exit(0)

# ── Apply fixes ─────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("STEP 4 — Applying fixes")
print("="*70)

import shutil
shutil.copy2(FM, BAK)
print(f"  Backed up to {BAK}")

# Build a reference row from a real mid-major team (for non-key feature defaults)
# Use a team at roughly the median KenPom level
median_team = fm.groupby("team_id")[COL_BARTHAG].mean().dropna().sort_values()
median_idx = len(median_team) // 2
median_team_id = median_team.index[median_idx]
median_row = fm[fm["team_id"] == median_team_id].iloc[0].copy()
print(f"  Median reference team: {median_team_id} (barthag={median_team.iloc[median_idx]:.3f})")

# For each tier, find a comparable real team to clone non-key features from
def find_comparable(target_barthag, tolerance=0.05):
    """Find a real team with similar barthag to use as feature template"""
    team_avg = fm.groupby("team_id")[COL_BARTHAG].mean().dropna()
    candidates = team_avg[(team_avg - target_barthag).abs() < tolerance]
    if len(candidates) == 0:
        candidates = team_avg
    # Pick the closest
    best_tid = (candidates - target_barthag).abs().idxmin()
    return fm[fm["team_id"] == best_tid].iloc[0].copy()

def fix_rows(df, team_id, stats, comparable_row):
    """Update the key efficiency columns in rows for team_id"""
    mask = df["team_id"] == team_id
    n = mask.sum()
    
    if n == 0:
        # Build synthetic rows using comparable team as template
        new_rows = []
        for i in range(5):
            row = comparable_row.copy()
            row["team_id"] = team_id
            row["date"] = "2026-03-15"
            row["game_id"] = f"synthetic_{team_id}_2026_{i}"
            # Set key features
            if COL_ADJ_O:   row[COL_ADJ_O]   = stats["adj_o"]
            if COL_ADJ_D:   row[COL_ADJ_D]   = stats["adj_d"]
            if COL_BARTHAG: row[COL_BARTHAG]  = stats["barthag"]
            if COL_TEMPO:   row[COL_TEMPO]    = stats["tempo"]
            new_rows.append(row)
        return pd.concat([df] + [pd.DataFrame([r for r in new_rows])], ignore_index=True), n, "added"
    else:
        # Fix existing rows
        if COL_ADJ_O:   df.loc[mask, COL_ADJ_O]   = stats["adj_o"]
        if COL_ADJ_D:   df.loc[mask, COL_ADJ_D]   = stats["adj_d"]
        if COL_BARTHAG: df.loc[mask, COL_BARTHAG]  = stats["barthag"]
        if COL_TEMPO:   df.loc[mask, COL_TEMPO]    = stats["tempo"]
        return df, n, "fixed"

for tid, stats in TEAM_STATS.items():
    comp_row = find_comparable(stats["barthag"])
    fm_updated, n, action = fix_rows(fm_updated, tid, stats, comp_row)
    print(f"  {tid:<25} {action} {n} rows  (adj_o={stats['adj_o']} adj_d={stats['adj_d']} barthag={stats['barthag']:.3f})")

# Cast object columns to str to avoid pyarrow issues
for col in fm_updated.select_dtypes("object").columns:
    fm_updated[col] = fm_updated[col].astype(str)

fm_updated.to_parquet(FM, index=False)
print(f"\n  Saved: {len(fm_updated):,} rows, {fm_updated['team_id'].nunique()} unique teams")

# ── Step 5: Verify ───────────────────────────────────────────────────────────
print("\n" + "="*70)
print("STEP 5 — Verification")
print("="*70)

fm_check = pd.read_parquet(FM)
print(f"{'TEAM':<25} {'ADJ_O':>8} {'ADJ_D':>8} {'BARTHAG':>8} {'STATUS'}")
print("-"*65)
for tid, stats in TEAM_STATS.items():
    rows = fm_check[fm_check["team_id"] == tid]
    if len(rows) == 0:
        print(f"  {tid:<25} STILL MISSING ❌")
        continue
    r = rows.iloc[0]
    act_o = float(r[COL_ADJ_O])   if COL_ADJ_O   and not pd.isna(r.get(COL_ADJ_O))   else float("nan")
    act_d = float(r[COL_ADJ_D])   if COL_ADJ_D   and not pd.isna(r.get(COL_ADJ_D))   else float("nan")
    act_b = float(r[COL_BARTHAG]) if COL_BARTHAG and not pd.isna(r.get(COL_BARTHAG)) else float("nan")
    ok = not np.isnan(act_o) and abs(act_o - stats["adj_o"]) < 1
    status = "✓ FIXED" if ok else f"❌ STILL WRONG (expected {stats['adj_o']})"
    print(f"  {tid:<25} {act_o:>8.1f} {act_d:>8.1f} {act_b:>8.3f}  {status}")

print("\nDone. Now run: python tournament_predict.py --bracket bracket.json")
print("Check that Tennessee State / McNeese / Cal Baptist outputs are now sane.")