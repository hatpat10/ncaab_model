"""
audit_tournament_teams.py
=========================
Checks which Round 1 tournament teams exist in your feature matrix
and prints the key format so you can fix TOURNAMENT_ALIASES correctly.

Run from D:\\ncaab_model:
    python audit_tournament_teams.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from predictions.daily_pipeline import load_feature_matrix, normalize_team

fm = load_feature_matrix()

# Get all team keys from the matrix
if hasattr(fm, 'index') and fm.index.dtype == object:
    all_keys = sorted(fm.index.tolist())
else:
    col = next((c for c in fm.columns if 'team' in c.lower()), None)
    all_keys = sorted(fm[col].tolist()) if col else []

print(f"\nFeature matrix: {len(all_keys)} team keys")
print(f"Key format sample (first 10):")
for k in all_keys[:10]:
    print(f"  '{k}'")

# Every Round 1 team — check which ones resolve correctly
BRACKET = [
    ("East",    1,  "Duke"),
    ("East",    16, "Siena"),
    ("East",    8,  "Ohio St"),
    ("East",    9,  "TCU"),
    ("East",    5,  "St Johns"),
    ("East",    12, "Northern Iowa"),
    ("East",    4,  "Kansas"),
    ("East",    13, "Cal Baptist"),
    ("East",    6,  "Louisville"),
    ("East",    11, "South Florida"),
    ("East",    3,  "Michigan St"),
    ("East",    14, "N Dakota St"),
    ("East",    7,  "UCLA"),
    ("East",    10, "UCF"),
    ("East",    2,  "UConn"),
    ("East",    15, "Furman"),
    ("West",    1,  "Arizona"),
    ("West",    16, "Long Island"),
    ("West",    8,  "Villanova"),
    ("West",    9,  "Utah State"),
    ("West",    5,  "Wisconsin"),
    ("West",    12, "High Point"),
    ("West",    4,  "Arkansas"),
    ("West",    13, "Hawaii"),
    ("West",    6,  "BYU"),
    ("West",    11, "NC State"),
    ("West",    3,  "Gonzaga"),
    ("West",    14, "Kennesaw St"),
    ("West",    7,  "Miami"),
    ("West",    10, "Missouri"),
    ("West",    2,  "Purdue"),
    ("West",    15, "Queens"),
    ("South",   1,  "Florida"),
    ("South",   16, "Lehigh"),
    ("South",   8,  "Clemson"),
    ("South",   9,  "Iowa"),
    ("South",   5,  "Vanderbilt"),
    ("South",   12, "McNeese"),
    ("South",   4,  "Nebraska"),
    ("South",   13, "Troy"),
    ("South",   6,  "North Carolina"),
    ("South",   11, "VCU"),
    ("South",   3,  "Illinois"),
    ("South",   14, "Penn"),
    ("South",   7,  "Saint Mary's"),
    ("South",   10, "Texas A&M"),
    ("South",   2,  "Houston"),
    ("South",   15, "Idaho"),
    ("Midwest", 1,  "Michigan"),
    ("Midwest", 16, "UMBC"),
    ("Midwest", 8,  "Georgia"),
    ("Midwest", 9,  "Saint Louis"),
    ("Midwest", 5,  "Texas Tech"),
    ("Midwest", 12, "Akron"),
    ("Midwest", 4,  "Alabama"),
    ("Midwest", 13, "Hofstra"),
    ("Midwest", 6,  "Tennessee"),
    ("Midwest", 11, "Miami OH"),
    ("Midwest", 3,  "Virginia"),
    ("Midwest", 14, "Wright State"),
    ("Midwest", 7,  "Kentucky"),
    ("Midwest", 10, "Santa Clara"),
    ("Midwest", 2,  "Iowa State"),
    ("Midwest", 15, "Tennessee State"),
]

print(f"\n{'REGION':<8} {'SEED':>4}  {'BRACKET NAME':<20} {'NORM KEY':<25} {'IN MATRIX?'}")
print("─" * 80)

missing = []
found = []
for region, seed, name in BRACKET:
    norm = normalize_team(name)
    in_matrix = norm in all_keys
    status = "YES" if in_matrix else "MISSING"
    print(f"{region:<8} #{seed:>2}   {name:<20} {norm:<25} {status}")
    if not in_matrix:
        # Find closest match
        frags = [f for f in norm.replace('_',' ').split() if len(f) > 3]
        close = [k for k in all_keys if any(f in k for f in frags)]
        missing.append((name, norm, close[:5]))
    else:
        found.append((name, norm))

print(f"\n{'─'*80}")
print(f"Found: {len(found)}/64   Missing: {len(missing)}/64")

if missing:
    print(f"\n{'─'*80}")
    print("MISSING TEAMS — add these to TOURNAMENT_ALIASES:")
    print("─"*80)
    for name, norm, close in missing:
        print(f"\n  '{name}' → normalize='{norm}'")
        if close:
            print(f"  Closest matrix keys: {close}")
        else:
            print(f"  No close match found — team not in matrix at all")