"""
diagnose_zero_spread.py
=======================
python diagnose_zero_spread.py
"""
import sys, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

from predictions.daily_pipeline import build_game_features, load_feature_matrix, predict_game
import joblib
from pathlib import Path
import numpy as np

ROOT = Path('.').resolve()
fm = load_feature_matrix()

# Load models
models = {}
for name in ['spread_model', 'win_prob_model', 'totals_model']:
    p = ROOT / 'models' / 'saved' / f'{name}.pkl'
    if p.exists():
        models[name] = joblib.load(p)

spread_saved = models.get('spread_model', {})
model   = spread_saved.get('model')
imputer = spread_saved.get('imputer')
feats   = spread_saved.get('features', [])

print(f"Model feature list ({len(feats)}):")
print(f"  {feats}")

# Get features for a failing game (LIU vs Arizona)
feat = build_game_features('Arizona', 'liu', 1, '2026-03-20', fm, None)
print(f"\nbuild_game_features output ({len(feat)} features):")
print(f"  Keys: {list(feat.index)}")

# Check overlap
model_set = set(feats)
feat_set  = set(feat.index)
in_both   = model_set & feat_set
only_model = model_set - feat_set
only_feat  = feat_set - model_set

print(f"\nIn both model and features: {sorted(in_both)}")
print(f"In model but NOT in features (will be NaN->imputed): {sorted(only_model)}")
print(f"In features but NOT in model (will be ignored): {sorted(only_feat)}")

# Simulate predict_game step by step
import pandas as pd
X_series = feat.reindex(feats)
print(f"\nAfter reindex to model features:")
for f, v in X_series.items():
    nan = " <-- NaN" if pd.isna(v) else ""
    print(f"  {f}: {v}{nan}")

X = X_series.values.reshape(1, -1)
X_imp = imputer.transform(X)
print(f"\nAfter imputer.transform:")
for f, v in zip(feats, X_imp[0]):
    print(f"  {f}: {v:.4f}")

pred = float(model.predict(X_imp)[0])
print(f"\nSpread prediction: {pred:.2f}")

# Also test a game that WORKS (Kansas vs Cal Baptist)
print("\n" + "="*60)
print("WORKING GAME: Kansas vs Cal Baptist")
feat2 = build_game_features('Kansas', 'California Baptist', 1, '2026-03-20', fm, None)
X2 = feat2.reindex(feats).values.reshape(1, -1)
X2_imp = imputer.transform(X2)
pred2 = float(model.predict(X2_imp)[0])
print(f"Spread prediction: {pred2:.2f}")
print(f"net_rating_delta: {feat2.get('net_rating_delta')}")