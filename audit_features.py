"""
audit_features.py
Run from D:\ncaab_model with venv active:
  python audit_features.py

Checks what features synthetic tournament teams got,
and compares them to real KenPom/BartTorvik values.
Flags any teams whose model-facing features are badly wrong.
"""

import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FM_PATH = ROOT / "data" / "processed" / "feature_matrix_full.parquet"

fm = pd.read_parquet(FM_PATH)
print(f"Feature matrix: {fm.shape[0]:,} rows, {fm['team_id'].nunique()} unique teams\n")

# ── Real stats from BartTorvik / KenPom (as of selection sunday 2026) ──────
# These are the values we INJECTED in the backfill
INJECTED = {
    "ohio_st":            {"adj_o": 116.8, "adj_d": 100.4, "barthag": 0.732, "tempo": 69.8},
    "california_baptist": {"adj_o": 108.4, "adj_d": 104.8, "barthag": 0.481, "tempo": 63.2},
    "michigan_st":        {"adj_o": 121.1, "adj_d":  96.3, "barthag": 0.841, "tempo": 69.4},
    "n_dakota_st":        {"adj_o": 105.2, "adj_d": 105.1, "barthag": 0.412, "tempo": 63.8},
    "uconn":              {"adj_o": 122.3, "adj_d":  97.1, "barthag": 0.856, "tempo": 68.3},
    "kennesaw_st":        {"adj_o": 104.1, "adj_d": 106.7, "barthag": 0.388, "tempo": 65.3},
    "penn":               {"adj_o": 107.9, "adj_d": 105.2, "barthag": 0.441, "tempo": 66.1},
}

# All tournament teams to check (including suspects)
ALL_SUSPECTS = {
    "tennessee_st":   {"adj_o": 102.1, "adj_d": 107.3, "barthag": 0.331, "tempo": 68.1, "note": "SUSPECT - may collide with 'tennessee'"},
    "mcneese_state":  {"adj_o": 106.8, "adj_d": 103.2, "barthag": 0.501, "tempo": 71.2, "note": "SUSPECT - Sun Belt, overrated by model"},
    "mcneese":        {"adj_o": 106.8, "adj_d": 103.2, "barthag": 0.501, "tempo": 71.2, "note": "alt key for mcneese"},
}
ALL_SUSPECTS.update(INJECTED)

print("=" * 80)
print(f"{'TEAM':<25} {'IN_MATRIX':>10} {'ADJ_O':>8} {'ADJ_D':>8} {'BARTHAG':>8} {'STATUS'}")
print("=" * 80)

# Also check Tennessee variants
tn_variants = fm[fm["team_id"].str.contains("tennessee", na=False, case=False)]["team_id"].unique()
print(f"\nAll Tennessee variants in matrix: {sorted(tn_variants)}")
mcn_variants = fm[fm["team_id"].str.contains("mcneese", na=False, case=False)]["team_id"].unique()
print(f"All McNeese variants in matrix:   {sorted(mcn_variants)}\n")

for team, expected in ALL_SUSPECTS.items():
    rows = fm[fm["team_id"] == team]
    note = expected.get("note", "")
    
    if len(rows) == 0:
        print(f"  {team:<25} {'MISSING':>10}  -- NOT IN MATRIX -- {note}")
        continue
    
    r = rows.iloc[0]
    
    # Try multiple column name variants
    def get_col(row, *names, default=np.nan):
        for n in names:
            if n in row.index and not pd.isna(row[n]):
                return float(row[n])
        return default
    
    adj_o   = get_col(r, "adj_o", "adjoe", "off_eff", "team_adj_o")
    adj_d   = get_col(r, "adj_d", "adjde", "def_eff", "team_adj_d")
    barthag = get_col(r, "barthag", "team_barthag")
    
    exp_o = expected["adj_o"]
    exp_d = expected["adj_d"]
    exp_b = expected["barthag"]
    
    # Flag if values are badly off
    o_ok = abs(adj_o - exp_o) < 3 if not np.isnan(adj_o) else False
    d_ok = abs(adj_d - exp_d) < 3 if not np.isnan(adj_d) else False
    b_ok = abs(barthag - exp_b) < 0.05 if not np.isnan(barthag) else False
    
    status = "✓ OK" if (o_ok and d_ok) else f"❌ WRONG (expected O={exp_o} D={exp_d} B={exp_b:.3f})"
    n_rows = len(rows)
    
    print(f"  {team:<25} {n_rows:>10}  {adj_o:>7.1f}  {adj_d:>7.1f}  {barthag:>8.3f}  {status}")
    if note:
        print(f"    NOTE: {note}")

print("\n" + "=" * 80)
print("\nKey feature columns present in matrix:")
adj_cols = [c for c in fm.columns if any(x in c.lower() for x in ["adj_o", "adj_d", "adjoe", "adjde", "barthag", "tempo", "off_eff", "def_eff"])]
print(f"  {adj_cols[:20]}")

print("\nSample row for a known good team (duke):")
duke = fm[fm["team_id"] == "duke"]
if len(duke):
    r = duke.iloc[0]
    for col in adj_cols[:10]:
        if col in r.index:
            print(f"  {col}: {r[col]}")

print("\nSample row for tennessee (to compare vs tennessee_st):")
tenn = fm[fm["team_id"] == "tennessee"]
if len(tenn):
    r = tenn.iloc[0]
    for col in adj_cols[:10]:
        if col in r.index:
            print(f"  {col}: {r[col]}")

print("\nSample row for tennessee_st:")
tst = fm[fm["team_id"] == "tennessee_st"]
if len(tst):
    r = tst.iloc[0]
    for col in adj_cols[:10]:
        if col in r.index:
            print(f"  {col}: {r[col]}")
else:
    print("  NOT FOUND IN MATRIX")