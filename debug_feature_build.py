"""
debug_feature_build.py
======================
Run from D:\\ncaab_model:
    python debug_feature_build.py

Traces exactly why build_game_features returns 0.0 spread for
teams like Siena, UConn, LIU, Kennesaw St, etc.
Prints the actual feature dict that goes into the model.
"""
import sys, json
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Load what tournament_predict.py loads
from predictions.daily_pipeline import build_game_features, load_feature_matrix
import joblib

print("Loading feature matrix and models...")
fm = load_feature_matrix()
models = {
    "spread_model":   joblib.load(ROOT / "models" / "saved" / "spread_model.pkl"),
    "win_prob_model": joblib.load(ROOT / "models" / "saved" / "win_prob_model.pkl"),
    "totals_model":   joblib.load(ROOT / "models" / "saved" / "totals_model.pkl"),
}
spread_model = models["spread_model"]

# What features does the spread model expect?
if hasattr(spread_model, "feature_names_in_"):
    MODEL_FEATURES = list(spread_model.feature_names_in_)
elif hasattr(spread_model, "named_steps"):
    for name, step in spread_model.named_steps.items():
        if hasattr(step, "feature_names_in_"):
            MODEL_FEATURES = list(step.feature_names_in_)
            break
    else:
        MODEL_FEATURES = []
else:
    MODEL_FEATURES = []

print(f"Model expects {len(MODEL_FEATURES)} features")
print(f"Key efficiency features: {[f for f in MODEL_FEATURES if any(x in f for x in ['adj','barthag','tempo','eff'])]}")

# Test pairs: (home, away) for games that return model spread = 0
TEST_GAMES = [
    ("Duke",       "Siena",         "duke",          "siena"),
    ("uconn",      "Furman",        "uconn",         "furman"),
    ("Arizona",    "liu",           "arizona",       "liu"),
    ("Gonzaga",    "kennesaw_st",   "gonzaga",       "kennesaw_st"),
    ("Purdue",     "queens (nc)",   "purdue",        "queens (nc)"),
    ("Houston",    "Idaho",         "houston",       "idaho"),
    ("Virginia",   "wright_state",  "virginia",      "wright_state"),
    ("Iowa State", "tennessee_state", "iowa_state",  "tennessee_state"),
    # A game that DOES work for comparison
    ("Kansas",     "california_baptist", "kansas",   "california_baptist"),
]

print("\n" + "="*80)
print("FEATURE BUILD TRACE")
print("="*80)

for home_name, away_name, home_key, away_key in TEST_GAMES:
    print(f"\n{'─'*60}")
    print(f"  {away_name} @ {home_name}")
    
    # Check if teams exist in matrix
    home_rows = fm[fm["team_id"] == home_key]
    away_rows = fm[fm["team_id"] == away_key]
    print(f"  Home ({home_key}): {len(home_rows)} rows in matrix")
    print(f"  Away ({away_key}): {len(away_rows)} rows in matrix")
    
    if len(home_rows):
        r = home_rows.iloc[0]
        print(f"  Home t_adj_o={r.get('t_adj_o', 'MISSING')}  t_adj_d={r.get('t_adj_d', 'MISSING')}")
    if len(away_rows):
        r = away_rows.iloc[0]
        print(f"  Away t_adj_o={r.get('t_adj_o', 'MISSING')}  t_adj_d={r.get('t_adj_d', 'MISSING')}")
    
    # Try building features
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            feat = build_game_features(
                home_team=home_name,
                away_team=away_name,
                is_neutral=1,
                target_date="2026-03-20",
                feature_matrix=fm,
                odds_row=None,
            )
        
        if feat is None:
            print(f"  ❌ build_game_features returned None")
            continue
        
        # Check which model features are NaN
        if MODEL_FEATURES:
            nan_feats = [f for f in MODEL_FEATURES if f in feat.columns and feat[f].isna().any()]
            ok_feats  = [f for f in MODEL_FEATURES if f in feat.columns and not feat[f].isna().any()]
            missing   = [f for f in MODEL_FEATURES if f not in feat.columns]
            
            print(f"  Features: {len(ok_feats)} OK, {len(nan_feats)} NaN, {len(missing)} missing")
            if nan_feats:
                print(f"  NaN features: {nan_feats[:15]}")
            if missing:
                print(f"  Missing from output: {missing[:10]}")
            
            # Try prediction
            try:
                pred = float(spread_model.predict(feat[MODEL_FEATURES].fillna(0))[0])
                pred_nan = float(spread_model.predict(feat[MODEL_FEATURES])[0])
                print(f"  Model spread (fillna=0): {pred:.1f}  (with NaN): {pred_nan:.1f}")
            except Exception as e:
                print(f"  Prediction error: {e}")
        else:
            print(f"  Features shape: {feat.shape}")
            # Show key columns
            key_cols = [c for c in feat.columns if any(x in c for x in ["adj","barthag","tempo","eff","delta"])]
            for col in key_cols[:10]:
                val = feat[col].iloc[0] if len(feat) else "N/A"
                print(f"    {col}: {val}")
    
    except Exception as e:
        print(f"  ❌ Exception: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("""
If features return None: build_game_features can't find the team in the matrix
  → Fix: check normalize_team() output for each team name

If features have many NaN: the team has rows but delta/rolling features are NaN
  → Fix: build_game_features needs a fallback for teams with sparse history
  → The spread model predicts 0.0 when key features are all NaN

If prediction is 0.0 with NaN but non-zero with fillna(0):
  → The model is getting NaN input and outputting 0.0 by default
  → Fix: either impute NaN with team's known efficiency stats before predicting,
         or use a direct KenPom-based spread calculation as fallback
""")