"""
audit_predictor.py
==================
Run from D:\\ncaab_model:
    python audit_predictor.py

Traces exactly what features tournament_predict.py builds for each
suspect team. Prints the actual feature vector going into the model
and flags anything that looks wrong.
"""

import sys
import json
import importlib
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

FM_PATH = ROOT / "data" / "processed" / "feature_matrix_full.parquet"
BRACKET_PATH = ROOT / "bracket.json"

print("Loading feature matrix...")
fm = pd.read_parquet(FM_PATH)
print(f"Shape: {fm.shape}  |  Teams: {fm['team_id'].nunique()}")

# ── 1. Find what columns the model actually needs ────────────────────────────
print("\n" + "="*70)
print("STEP 1 — What columns does the feature matrix have?")
print("="*70)

# Group columns by type
eff_cols  = sorted([c for c in fm.columns if any(x in c.lower() for x in ["adj_o","adj_d","barthag","off_eff","def_eff"])])
tempo_col = sorted([c for c in fm.columns if "tempo" in c.lower()])
rank_cols = sorted([c for c in fm.columns if "rank" in c.lower()])
delta_cols = sorted([c for c in fm.columns if "delta" in c.lower()])
other_cols = [c for c in fm.columns if c not in eff_cols + tempo_col + rank_cols + delta_cols]

print(f"\n  Efficiency cols: {eff_cols}")
print(f"  Tempo cols:      {tempo_col}")
print(f"  Rank cols:       {rank_cols[:10]}...")
print(f"  Delta cols:      {delta_cols[:10]}...")
print(f"  Other cols ({len(other_cols)}): {other_cols[:20]}...")

# ── 2. Check how tournament_predict.py looks up teams ───────────────────────
print("\n" + "="*70)
print("STEP 2 — How does tournament_predict.py fetch team features?")
print("="*70)

tp_path = ROOT / "tournament_predict.py"
if tp_path.exists():
    src = tp_path.read_text(encoding="utf-8", errors="replace")
    
    # Find feature lookup section
    lines = src.split("\n")
    lookup_lines = [(i+1, l) for i, l in enumerate(lines) 
                    if any(x in l for x in ["team_id", "feature_matrix", "fm[", "get_team", "lookup", "normalize_team", "ALIAS"])]
    
    print(f"\n  Relevant lines in tournament_predict.py ({len(lookup_lines)} found):")
    for lineno, line in lookup_lines[:40]:
        print(f"    {lineno:4d}: {line.rstrip()}")
    
    # Find the feature columns the model uses
    feat_lines = [(i+1, l) for i, l in enumerate(lines)
                  if any(x in l for x in ["FEATURE_COLS", "feature_cols", "X_cols", "features =", "feature_names"])]
    print(f"\n  Feature column definitions:")
    for lineno, line in feat_lines[:20]:
        print(f"    {lineno:4d}: {line.rstrip()}")
    
    # Find alias/normalize section
    alias_lines = [(i+1, l) for i, l in enumerate(lines)
                   if any(x in l for x in ["TOURNAMENT_ALIASES", "tennessee", "mcneese", "normalize"])]
    print(f"\n  Alias/normalize section:")
    for lineno, line in alias_lines[:30]:
        print(f"    {lineno:4d}: {line.rstrip()}")
else:
    print("  tournament_predict.py not found!")

# ── 3. Simulate what features a team would get ───────────────────────────────
print("\n" + "="*70)
print("STEP 3 — Simulate feature lookup for suspect teams")
print("="*70)

SUSPECTS = {
    "tennessee_st":   "tennessee_state",   # alias needed
    "tennessee state": "tennessee_state",
    "mcneese state":  "mcneese",
    "mcneese_state":  "mcneese",
    "california baptist": "california_baptist",
    "cal baptist":    "california_baptist",
}

# What does the model actually look for?
# Try to import and use normalize_team if available
try:
    # Try importing directly
    spec = importlib.util.spec_from_file_location("tournament_predict", tp_path)
    mod  = importlib.util.module_from_spec(spec)
    # Don't execute the whole module, just get the normalize function
    src_trimmed = src[:src.find("if __name__")] if "if __name__" in src else src
    print("  (importing normalize_team from tournament_predict...)")
except Exception as e:
    print(f"  Could not import: {e}")

# Manual check - look up each suspect key in the matrix
check_keys = [
    "tennessee_st", "tennessee_state", "tennessee",
    "mcneese_state", "mcneese",
    "california_baptist", "cal_baptist",
    "n_dakota_st", "north_dakota_state",
]

print(f"\n  {'KEY':<30} {'IN_MATRIX':>12} {'t_adj_o':>8} {'t_adj_d':>8} {'t_barthag':>10}")
print("  " + "-"*70)
for key in check_keys:
    rows = fm[fm["team_id"] == key]
    if len(rows) == 0:
        print(f"  {key:<30} {'MISSING':>12}")
    else:
        r = rows.iloc[0]
        adj_o   = r.get("t_adj_o",   np.nan)
        adj_d   = r.get("t_adj_d",   np.nan)
        barthag = r.get("t_barthag", np.nan)
        adj_o   = float(adj_o)   if not pd.isna(adj_o)   else float("nan")
        adj_d   = float(adj_d)   if not pd.isna(adj_d)   else float("nan")
        barthag = float(barthag) if not pd.isna(barthag) else float("nan")
        nan_str = lambda x: f"{x:.1f}" if not np.isnan(x) else "NaN"
        nan_str3= lambda x: f"{x:.3f}" if not np.isnan(x) else "NaN"
        print(f"  {key:<30} {len(rows):>12}  {nan_str(adj_o):>8}  {nan_str(adj_d):>8}  {nan_str3(barthag):>10}")

# ── 4. Check TOURNAMENT_ALIASES in predictor ─────────────────────────────────
print("\n" + "="*70)
print("STEP 4 — Parse TOURNAMENT_ALIASES from tournament_predict.py")
print("="*70)

if tp_path.exists():
    # Extract the alias dict
    import re
    alias_match = re.search(r"TOURNAMENT_ALIASES\s*=\s*\{([^}]+)\}", src, re.DOTALL)
    if alias_match:
        alias_src = alias_match.group(0)
        print(f"\n  Found TOURNAMENT_ALIASES:")
        print(f"  {alias_src[:800]}")
        
        # Check if Tennessee State / McNeese are in there
        for check in ["tennessee", "mcneese", "cal_baptist", "california_baptist", "n_dakota"]:
            if check in alias_src.lower():
                print(f"  ✓ '{check}' IS in aliases")
            else:
                print(f"  ❌ '{check}' NOT in aliases — may fail lookup")
    else:
        print("  TOURNAMENT_ALIASES not found in tournament_predict.py")
        
        # Look for any alias/mapping dict
        any_alias = re.findall(r"['\"]tennessee[^'\"]*['\"]", src[:5000], re.IGNORECASE)
        print(f"  Tennessee mentions: {any_alias[:5]}")

# ── 5. Check what columns the trained model expects ──────────────────────────
print("\n" + "="*70)
print("STEP 5 — What columns does the trained spread_model expect?")
print("="*70)

import joblib
model_path = ROOT / "models" / "spread_model.pkl"
if model_path.exists():
    try:
        model = joblib.load(model_path)
        if hasattr(model, "feature_names_in_"):
            feat_names = list(model.feature_names_in_)
            print(f"\n  Model expects {len(feat_names)} features:")
            # Look for efficiency-related
            eff_feats = [f for f in feat_names if any(x in f.lower() for x in ["adj","barthag","tempo","eff"])]
            print(f"  Efficiency features: {eff_feats}")
            print(f"  First 20 features: {feat_names[:20]}")
            
            # Check overlap with matrix
            matrix_cols = set(fm.columns)
            model_cols  = set(feat_names)
            missing_from_matrix = model_cols - matrix_cols
            if missing_from_matrix:
                print(f"\n  ❌ Model needs these cols NOT in matrix: {sorted(missing_from_matrix)[:20]}")
            else:
                print(f"\n  ✓ All model feature columns exist in matrix")
        else:
            print(f"  Model type: {type(model).__name__} — no feature_names_in_ attribute")
            # Try pipeline
            if hasattr(model, "steps"):
                for name, step in model.steps:
                    if hasattr(step, "feature_names_in_"):
                        print(f"  Pipeline step '{name}' features: {list(step.feature_names_in_)[:10]}")
    except Exception as e:
        print(f"  Error loading model: {e}")
else:
    print("  spread_model.pkl not found")
    # Try other locations
    for p in ROOT.glob("models/*.pkl"):
        print(f"  Found: {p}")

print("\n" + "="*70)
print("SUMMARY OF ISSUES TO FIX")
print("="*70)
print("""
Based on this audit:

1. If 'tennessee_st' is not aliased to 'tennessee_state' in TOURNAMENT_ALIASES
   → tournament_predict.py will fall back to 'tennessee' features (wrong team)
   → FIX: Add 'tennessee st' / 'tennessee_st' → 'tennessee_state' to aliases

2. If 'mcneese_state' is not aliased to 'mcneese'  
   → predictor can't find the team's features
   → FIX: Add 'mcneese state' / 'mcneese_state' → 'mcneese' to aliases

3. If the model's feature columns don't match 't_adj_o'/'t_adj_d'/'t_barthag'
   → features are silently NaN → model predicts garbage
   → FIX: verify the model was trained on the same column names

4. If delta/rank columns are NaN for synthetic teams
   → the model will use the feature matrix values, not computed matchup deltas
   → these get computed at prediction time using both teams' stats
   → this is actually fine IF the base stats (t_adj_o etc.) are correct
""")