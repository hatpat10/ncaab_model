"""
patch_final_aliases.py
======================
Run from D:\\ncaab_model:
    python patch_final_aliases.py

Adds the final 7 team aliases to TOURNAMENT_ALIASES in tournament_predict.py.
All these teams ARE in the matrix but their normalize_team() output doesn't
match the matrix key exactly (parentheses, special chars, etc.)
"""
import re, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TP   = ROOT / "tournament_predict.py"
BAK  = ROOT / "tournament_predict.py.bak2"

src = TP.read_text(encoding="utf-8")
shutil.copy2(TP, BAK)
print(f"Backed up to {BAK}")

# The new aliases to add — maps bracket JSON names to exact matrix team_id keys
# normalize_team() is called AFTER alias resolution, so we need to map to the
# final key that exists in the feature matrix.
NEW_ENTRIES = '''
    # ── Teams where normalize_team() output doesn't match matrix key ──────────
    # St John's — matrix key is 'st_johns'
    "st johns":              "st_johns",
    "st. johns":             "st_johns",
    "st. john's":            "st_johns",
    "saint johns":           "st_johns",

    # Hawaii — normalize returns 'hawaii', matrix has 'hawaii' ✓
    # (already resolves correctly via normalize_team, but add explicit alias)
    "hawai'i":               "hawaii",

    # Miami FL — matrix key is 'miami_fl'
    "miami fl":              "miami_fl",
    "miami (fl)":            "miami_fl",
    "miami florida":         "miami_fl",

    # Queens NC — matrix key is 'queens'
    "queens (nc)":           "queens",
    "queens nc":             "queens",
    "queens (n.c.)":         "queens",

    # Texas A&M — matrix key is 'texas_am'
    "texas a&m":             "texas_am",
    "texas a&amp;m":         "texas_am",

    # Saint Mary's CA — matrix key is 'saint_marys'
    "saint mary's (ca)":     "saint_marys",
    "saint mary's":          "saint_marys",
    "saint marys":           "saint_marys",
    "st. mary's":            "saint_marys",
    "st. marys":             "saint_marys",

    # Miami OH — matrix key is 'miami_oh'
    "miami (oh)":            "miami_oh",
    "miami oh":              "miami_oh",
    "miami ohio":            "miami_oh",
    "miami (ohio)":          "miami_oh",
'''

# Find the closing } of TOURNAMENT_ALIASES and insert before it
# Look for the last entry before the closing brace
alias_end = re.search(r"(\s*)\}", src[src.find("TOURNAMENT_ALIASES"):])
if not alias_end:
    print("ERROR: Could not find end of TOURNAMENT_ALIASES")
    exit(1)

# Find the absolute position of the closing brace
dict_start = src.find("TOURNAMENT_ALIASES")
dict_end_rel = alias_end.start()
dict_end_abs = dict_start + dict_end_rel

# Insert new entries before the closing brace
src = src[:dict_end_abs] + NEW_ENTRIES + src[dict_end_abs:]
print("Inserted 7 team alias groups ✓")

TP.write_text(src, encoding="utf-8")
print(f"Saved {TP}")

# Verify
print("\n" + "="*70)
print("VERIFICATION — Last 40 lines of TOURNAMENT_ALIASES")
print("="*70)
src_check = TP.read_text(encoding="utf-8")
alias_block = re.search(r"TOURNAMENT_ALIASES.*?\}", src_check, re.DOTALL)
if alias_block:
    lines = alias_block.group(0).split("\n")
    for line in lines[-45:]:
        print(f"  {line}")

# Also verify normalize_team resolves aliases correctly
print("\n" + "="*70)
print("ALIAS RESOLUTION CHECK")
print("="*70)
import sys
sys.path.insert(0, str(ROOT))
from predictions.daily_pipeline import normalize_team

# Load the updated aliases
import importlib
import tournament_predict as tp
importlib.reload(tp)

test_cases = [
    ("St Johns",          "st_johns"),
    ("saint john's",      "st_johns"),
    ("Hawai'i",           "hawaii"),
    ("Miami",             "miami_fl"),  # from bracket
    ("Queens",            "queens"),    # from bracket
    ("queens (nc)",       "queens"),
    ("Texas A&M",         "texas_am"),
    ("saint mary's (ca)", "saint_marys"),
    ("Miami OH",          "miami_oh"),
    ("miami (oh)",        "miami_oh"),
    # Previously fixed
    ("tennessee state",   "tennessee_state"),
    ("mcneese",           "mcneese"),
    ("cal baptist",       "california_baptist"),
]

import pandas as pd
fm = pd.read_parquet(ROOT / "data" / "processed" / "feature_matrix_full.parquet")

print(f"\n{'BRACKET_INPUT':<30} {'AFTER_ALIAS':>20} {'IN_MATRIX':>12} {'STATUS'}")
print("-"*70)
for bracket_name, expected_key in test_cases:
    # Simulate what tournament_predict.py does
    alias_resolved = tp.TOURNAMENT_ALIASES.get(bracket_name.lower(), bracket_name)
    norm = normalize_team(alias_resolved)
    in_matrix = len(fm[fm["team_id"] == norm]) > 0
    
    status = "OK" if in_matrix else f"MISSING (got '{norm}')"
    flag = "" if in_matrix else " <-- PROBLEM"
    print(f"  {bracket_name:<30} {alias_resolved:>20} {str(in_matrix):>12}  {status}{flag}")

print("\nDone. Run: python tournament_predict.py --bracket bracket.json")