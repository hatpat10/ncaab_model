"""
fix_tournament_torvik.py
========================
Run from D:\\ncaab_model:
    python fix_tournament_torvik.py

For each tournament team, checks if their most recent row has t_adj_o populated.
If not, finds their most recent row that does have it, and copies those values
to ALL rows for that team (since BartTorvik ratings are season-level, not game-level).

This ensures build_game_features always finds a valid BartTorvik row.
"""
import pandas as pd
import numpy as np
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FM   = ROOT / "data" / "processed" / "feature_matrix_full.parquet"
BAK  = ROOT / "data" / "processed" / "feature_matrix_full.parquet.bak5"

# All 64 tournament teams (matrix keys after alias resolution)
# These are the teams we need to verify and fix
TOURNAMENT_TEAMS = [
    # East
    "duke", "siena", "ohio_st", "tcu", "northern_iowa", "saint_john's",
    "california_baptist", "kansas", "south_florida", "louisville",
    "n_dakota_st", "michigan_st", "ucf", "ucla", "furman", "uconn",
    # West
    "arizona", "liu", "utah_state", "villanova", "high_point", "wisconsin",
    "hawai'i", "arkansas", "nc_state", "byu", "kennesaw_st", "gonzaga",
    "missouri", "miami", "queens (nc)", "purdue",
    # South
    "florida", "lehigh", "iowa", "clemson", "mcneese", "vanderbilt",
    "troy", "nebraska", "vcu", "north_carolina", "penn", "illinois",
    "texas_a&m", "saint mary's (ca)", "idaho", "houston",
    # Midwest
    "michigan", "umbc", "saint_louis", "georgia", "akron", "texas_tech",
    "hofstra", "alabama", "miami (oh)", "tennessee", "wright_state", "virginia",
    "santa_clara", "kentucky", "tennessee_state", "iowa_state",
]

TORVIK_COLS = ["t_adj_o", "t_adj_d", "t_adj_t", "t_barthag",
               "t_adj_o_rank", "t_adj_d_rank", "t_barthag_rank",
               "t_wab", "t_ov_cur_sos", "t_nc_cur_sos",
               "o_adj_o", "o_adj_d", "o_adj_t", "o_barthag",
               "o_wab", "o_ov_cur_sos", "o_nc_cur_sos",
               "net_rating_delta", "off_eff_delta", "def_eff_delta",
               "tempo_delta", "wab_delta", "sos_delta"]

print("Loading feature matrix...")
fm = pd.read_parquet(FM)
print(f"Shape: {fm.shape}")

# First, understand the problem
print("\n" + "="*70)
print("STEP 1 - Diagnosing: most recent row date vs most recent torvik row date")
print("="*70)

print(f"\n{'TEAM':<25} {'TOTAL':>6} {'W_TORVIK':>10} {'LAST_DATE':>12} {'LAST_TORVIK':>12} {'STATUS'}")
print("-"*80)

teams_needing_fix = []
for tid in TOURNAMENT_TEAMS:
    rows = fm[fm["team_id"] == tid].copy()
    if len(rows) == 0:
        print(f"  {tid:<25} MISSING")
        continue
    
    # Sort by date
    rows["_date"] = pd.to_datetime(rows["date"], errors="coerce")
    rows = rows.sort_values("_date")
    
    torvik_rows = rows[rows["t_adj_o"].notna()]
    last_date = rows["_date"].max()
    last_torvik_date = torvik_rows["_date"].max() if len(torvik_rows) > 0 else pd.NaT
    
    has_recent_torvik = (
        len(torvik_rows) > 0 and 
        not pd.isna(last_torvik_date) and
        not pd.isna(last_date) and
        (last_date - last_torvik_date).days <= 30  # torvik data within 30 days of last game
    )
    
    status = "OK" if has_recent_torvik else "NEEDS_FIX"
    if not has_recent_torvik:
        teams_needing_fix.append(tid)
    
    ld_str  = last_date.strftime("%Y-%m-%d") if not pd.isna(last_date) else "N/A"
    ltd_str = last_torvik_date.strftime("%Y-%m-%d") if not pd.isna(last_torvik_date) else "NONE"
    
    print(f"  {tid:<25} {len(rows):>6} {len(torvik_rows):>10} {ld_str:>12} {ltd_str:>12}  {status}")

print(f"\n  Teams needing fix: {len(teams_needing_fix)}")
print(f"  {teams_needing_fix}")

# Fix: for teams where the most recent torvik data is stale or missing,
# find their best available torvik data and forward-fill it to recent rows
print("\n" + "="*70)
print("STEP 2 - Applying fix")
print("="*70)

shutil.copy2(FM, BAK)
print(f"Backed up to {BAK}")

fm_fixed = fm.copy()
fixed_teams = []

for tid in teams_needing_fix:
    rows_idx = fm_fixed[fm_fixed["team_id"] == tid].index
    rows = fm_fixed.loc[rows_idx].copy()
    rows["_date"] = pd.to_datetime(rows["date"], errors="coerce")
    rows = rows.sort_values("_date")
    
    torvik_rows = rows[rows["t_adj_o"].notna()]
    
    if len(torvik_rows) == 0:
        print(f"  {tid}: No torvik data at all - skipping (needs manual entry)")
        continue
    
    # Get the best (most recent) torvik values
    best_torvik = torvik_rows.iloc[-1]
    
    # Forward-fill torvik columns to ALL rows for this team
    # (BartTorvik ratings are season-level, so using most recent is correct)
    cols_to_fill = [c for c in TORVIK_COLS if c in fm_fixed.columns]
    
    for col in cols_to_fill:
        val = best_torvik.get(col, np.nan)
        if not pd.isna(val):
            fm_fixed.loc[rows_idx, col] = val
    
    adj_o = best_torvik.get("t_adj_o", "?")
    adj_d = best_torvik.get("t_adj_d", "?")
    fixed_teams.append(tid)
    print(f"  {tid}: forward-filled from {best_torvik.get('_date', 'unknown')} "
          f"(adj_o={adj_o:.1f}, adj_d={adj_d:.1f})")

print(f"\n  Fixed {len(fixed_teams)} teams")

# Save
fm_fixed.to_parquet(FM, index=False)
print(f"\nSaved: {len(fm_fixed):,} rows")

# Verify
print("\n" + "="*70)
print("STEP 3 - Verification")
print("="*70)

fm_check = pd.read_parquet(FM)
print(f"\n{'TEAM':<25} {'t_adj_o':>8} {'t_adj_d':>8} {'t_barthag':>10} {'STATUS'}")
print("-"*60)

for tid in TOURNAMENT_TEAMS:
    rows = fm_check[fm_check["team_id"] == tid]
    if len(rows) == 0:
        print(f"  {tid:<25} MISSING")
        continue
    
    # Check most recent row
    rows_sorted = rows.copy()
    rows_sorted["_d"] = pd.to_datetime(rows_sorted["date"], errors="coerce")
    r = rows_sorted.sort_values("_d").iloc[-1]
    
    adj_o   = float(r["t_adj_o"])   if "t_adj_o"   in r.index and not pd.isna(r["t_adj_o"])   else float("nan")
    adj_d   = float(r["t_adj_d"])   if "t_adj_d"   in r.index and not pd.isna(r["t_adj_d"])   else float("nan")
    barthag = float(r["t_barthag"]) if "t_barthag" in r.index and not pd.isna(r["t_barthag"]) else float("nan")
    
    ok = not np.isnan(adj_o)
    status = "OK" if ok else "STILL NaN"
    
    o_str = f"{adj_o:.1f}"   if not np.isnan(adj_o)   else "NaN"
    d_str = f"{adj_d:.1f}"   if not np.isnan(adj_d)   else "NaN"
    b_str = f"{barthag:.3f}" if not np.isnan(barthag) else "NaN"
    
    flag = "" if ok else " <-- PROBLEM"
    print(f"  {tid:<25} {o_str:>8} {d_str:>8} {b_str:>10}  {status}{flag}")

print("""
Done. Now run:
  python tournament_predict.py --bracket bracket.json

Check that previously-failing teams (Siena, LIU, Idaho, Queens, etc.)
now show non-zero spreads instead of model spread = 0.0.
""")