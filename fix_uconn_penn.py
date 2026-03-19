"""
fix_uconn_penn.py
=================
python fix_uconn_penn.py

Fixes two alias issues:
- uconn -> normalize_team returns 'connecticut', real data is at 'connecticut'
- penn  -> normalize_team returns 'pennsylvania', real data is at 'pennsylvania'

Also verifies every bracket team resolves to a real matrix key with torvik data.
"""
import sys, re, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

TP  = ROOT / "tournament_predict.py"
BAK = ROOT / "tournament_predict.py.bak3"

src = TP.read_text(encoding="utf-8")
shutil.copy2(TP, BAK)

# Fix 1: uconn alias should point to 'connecticut' (what normalize_team returns)
# But actually the cleaner fix is: alias 'uconn' -> 'connecticut' directly
# so normalize_team('connecticut') -> 'connecticut' (passthrough)

# Fix 2: penn alias should point to 'pennsylvania'

# Find and replace the uconn and penn alias lines
fixes = [
    # (old_text, new_text)
    ('"uconn":                 "uconn"',
     '"uconn":                 "connecticut"'),
    ('"connecticut":           "uconn"',
     '"connecticut":           "connecticut"'),
    ('"penn":                  "penn"',
     '"penn":                  "pennsylvania"'),
    ('"pennsylvania":          "penn"',
     '"pennsylvania":          "pennsylvania"'),
]

changes = 0
for old, new in fixes:
    if old in src:
        src = src.replace(old, new)
        print(f"Fixed: {old!r} -> {new!r}")
        changes += 1

if changes == 0:
    print("No exact matches found - checking current alias values...")
    # Find what uconn and penn currently map to
    uconn_match = re.search(r'"uconn"\s*:\s*"([^"]+)"', src)
    penn_match   = re.search(r'"penn"\s*:\s*"([^"]+)"', src)
    print(f"  uconn currently maps to: {uconn_match.group(1) if uconn_match else 'NOT FOUND'}")
    print(f"  penn currently maps to:  {penn_match.group(1) if penn_match else 'NOT FOUND'}")
    
    # Do targeted replacements
    if uconn_match and uconn_match.group(1) != "connecticut":
        src = src.replace(f'"uconn":                 "{uconn_match.group(1)}"',
                          '"uconn":                 "connecticut"')
        print("Fixed uconn")
        changes += 1
    
    if penn_match and penn_match.group(1) != "pennsylvania":
        src = src.replace(f'"penn":                  "{penn_match.group(1)}"',
                          '"penn":                  "pennsylvania"')
        print("Fixed penn")
        changes += 1

TP.write_text(src, encoding="utf-8")
print(f"\nSaved with {changes} changes")

# Now verify ALL 32 bracket teams resolve correctly
print("\n" + "="*70)
print("FULL BRACKET VERIFICATION")
print("="*70)

import importlib
import tournament_predict as tp
importlib.reload(tp)
from predictions.daily_pipeline import normalize_team
import pandas as pd

fm = pd.read_parquet(ROOT / "data" / "processed" / "feature_matrix_full.parquet")

# All 32 R64 matchups from bracket.json
BRACKET_TEAMS = [
    # East
    "Siena", "Duke", "TCU", "Ohio St", "Northern Iowa", "St Johns",
    "California Baptist", "Kansas", "South Florida", "Louisville",
    "North Dakota State", "Michigan St", "UCF", "UCLA", "Furman", "UConn",
    # West
    "Arizona", "LIU", "Utah State", "Villanova", "High Point", "Wisconsin",
    "Hawaii", "Arkansas", "NC State", "BYU", "Kennesaw St", "Gonzaga",
    "Missouri", "Miami", "Queens", "Purdue",
    # South
    "Florida", "Lehigh", "Iowa", "Clemson", "McNeese", "Vanderbilt",
    "Troy", "Nebraska", "VCU", "North Carolina", "Penn", "Illinois",
    "Texas A&M", "Saint Mary's", "Idaho", "Houston",
    # Midwest
    "Michigan", "UMBC", "Saint Louis", "Georgia", "Akron", "Texas Tech",
    "Hofstra", "Alabama", "Miami OH", "Tennessee", "Wright State", "Virginia",
    "Santa Clara", "Kentucky", "Tennessee State", "Iowa State",
]

print(f"\n{'BRACKET':<22} {'ALIAS':<22} {'NORM':<22} {'ROWS':>6} {'TORVIK':>7} {'STATUS'}")
print("-"*90)

all_ok = True
for name in BRACKET_TEAMS:
    alias = tp.TOURNAMENT_ALIASES.get(name.lower(), name)
    norm  = normalize_team(alias)
    rows  = fm[fm["team_id"] == norm]
    torvik = rows[rows["t_adj_o"].notna()] if len(rows) else rows
    
    ok = len(rows) > 0 and len(torvik) > 0
    status = "OK" if ok else "PROBLEM"
    flag = " <--" if not ok else ""
    if not ok:
        all_ok = False
    
    print(f"  {name:<22} {alias:<22} {norm:<22} {len(rows):>6} {len(torvik):>7}  {status}{flag}")

print("\n" + "="*70)
if all_ok:
    print("ALL 64 TEAMS RESOLVE CORRECTLY")
    print("Run: python tournament_predict.py --bracket bracket.json")
else:
    print("SOME TEAMS STILL HAVE ISSUES - check flagged rows above")