"""
models/model_trainer.py
========================
NCAAB ML Prediction System — Phase 5: Model Training
2025-2026 Season • February 23, 2026

Trains three XGBoost models from the feature matrix.
Automatically uses feature_matrix_full.parquet (6 seasons, ~60k rows)
when available, falling back to feature_matrix.parquet (current season only).

  1. Spread model     — XGBoost regression → predicted point margin
  2. Win prob model   — XGBoost classifier → P(home team wins), calibrated
  3. Totals model     — XGBoost regression → predicted combined score

Time-series split (strict no-lookahead):
  Train:      everything before Jan 1 2026  (includes 2020-2025 if full matrix)
  Validation: Jan 1 2026 – Jan 31 2026
  Test:       Feb 1 2026 – present

Usage:
    python -m models.model_trainer              # train all 3 models
    python -m models.model_trainer --model spread|winprob|totals
    python -m models.model_trainer --eval       # eval saved models on test set
    python -m models.model_trainer --feature-path data/processed/feature_matrix_full.parquet
"""

from __future__ import annotations

import argparse
import logging
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    roc_auc_score,
    log_loss,
    accuracy_score,
    brier_score_loss,
)
import xgboost as xgb

warnings.filterwarnings("ignore", category=UserWarning)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parent.parent
FULL_PATH    = ROOT / "data" / "processed" / "feature_matrix_full.parquet"
CURRENT_PATH = ROOT / "data" / "processed" / "feature_matrix.parquet"
MODELS_DIR   = ROOT / "models" / "saved"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── Time-series split boundaries ─────────────────────────────────────────────
VAL_START  = "2026-01-01"
TEST_START = "2026-02-01"

# ──────────────────────────────────────────────────────────────────────────────
# FEATURE SETS
# ──────────────────────────────────────────────────────────────────────────────

SPREAD_FEATURES = [
    # BartTorvik efficiency deltas — top predictors
    "off_eff_delta",           # team AdjO − opp AdjD
    "def_eff_delta",           # team AdjD − opp AdjO (inverted)
    "net_rating_delta",        # (AdjO−AdjD) diff between teams
    "tempo_delta",             # pace matchup
    "def_suppression_factor",  # opponent defense tier
    "wab_delta",               # wins above bubble delta
    "sos_delta",               # strength of schedule delta

    # BartTorvik absolute ratings (team + opponent)
    "t_adj_o", "t_adj_d", "t_adj_t", "t_barthag",
    "o_adj_o", "o_adj_d", "o_adj_t", "o_barthag",

    # Rolling form
    "roll5_pts",    "roll5_margin",  "roll5_win_streak",
    "roll10_pts",   "roll10_margin",

    # Situational
    "is_home",       "is_neutral",     "days_rest",
    "rest_advantage","is_back_to_back","is_short_rest",
    "games_played",

    # Head-to-head history
    "h2h_avg_margin", "h2h_win_rate",

    # SOS context
    "t_ov_cur_sos", "o_ov_cur_sos",
    "t_nc_cur_sos", "o_nc_cur_sos",
]

# Totals adds market signal when available
TOTALS_FEATURES = SPREAD_FEATURES + ["vegas_total"]

# Win prob uses same features as spread
WIN_PROB_FEATURES = SPREAD_FEATURES

# Targets
TARGET_SPREAD = "margin"       # team_score − opp_score
TARGET_WIN    = "win"          # 1 = team won
TARGET_TOTAL  = "total_score"  # combined score

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def resolve_feature_path(override=None):
    """
    Pick the best feature matrix:
      1. Explicit --feature-path override
      2. feature_matrix_full.parquet  (6 seasons, preferred)
      3. feature_matrix.parquet       (current season fallback)
    """
    if override and Path(override) != CURRENT_PATH:
        log.info("Using specified feature path: %s", override)
        return Path(override)
    if FULL_PATH.exists():
        log.info("Multi-season matrix found: %s", FULL_PATH.name)
        return FULL_PATH
    log.warning("Multi-season matrix not found (%s). Using current season only.", FULL_PATH.name)
    log.warning("To build it: python -m processing.historical_feature_builder")
    return CURRENT_PATH


def load_features(path):
    """Load feature matrix, add derived targets, validate."""
    log.info("Loading feature matrix: %s", path)
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    score_col     = next((c for c in ["team_score", "score", "pts"] if c in df.columns), None)
    opp_score_col = next((c for c in ["opp_score", "opp_pts"]       if c in df.columns), None)

    if score_col and opp_score_col:
        df["margin"]      = df[score_col] - df[opp_score_col]
        df["win"]         = (df["margin"] > 0).astype(int)
        df["total_score"] = df[score_col] + df[opp_score_col]
    else:
        raise ValueError(
            f"Cannot find score columns. Available: "
            f"{[c for c in df.columns if 'score' in c.lower() or 'pts' in c.lower()]}"
        )

    before = len(df)
    df = df.dropna(subset=["margin", "win", "total_score"])
    log.info("Rows with valid targets: %d (dropped %d without scores)", len(df), before - len(df))
    log.info("Feature matrix: %d rows, %d cols | %s -> %s",
             len(df), len(df.columns),
             df["date"].min().date(), df["date"].max().date())
    return df


def time_series_split(df):
    """Strict temporal train/val/test split."""
    df      = df.sort_values("date")
    val_dt  = pd.to_datetime(VAL_START)
    test_dt = pd.to_datetime(TEST_START)

    train = df[df["date"] <  val_dt]
    val   = df[(df["date"] >= val_dt) & (df["date"] < test_dt)]
    test  = df[df["date"] >= test_dt]

    for name, split in [("Train", train), ("Val", val), ("Test", test)]:
        if len(split):
            log.info("  %-6s %6d rows  (%s -> %s)",
                     name + ":", len(split),
                     split["date"].min().date(), split["date"].max().date())
        else:
            log.warning("  %s: 0 rows", name)

    return train, val, test


def get_X_y(df, feature_cols, target):
    """Extract X, y — dedup feature list, skip missing columns."""
    seen, deduped = set(), []
    for c in feature_cols:
        if c not in seen:
            seen.add(c)
            deduped.append(c)

    available = [c for c in deduped if c in df.columns]
    missing   = [c for c in deduped if c not in df.columns]
    if missing:
        log.debug("Features not in matrix (skipping): %s", missing)

    return df[available].copy(), df[target].copy(), available


def impute(X_train, X_val, X_test=None):
    """Fit median imputer on train, transform all splits."""
    imp  = SimpleImputer(strategy="median")
    X_tr = imp.fit_transform(X_train)
    X_v  = imp.transform(X_val)
    X_te = imp.transform(X_test) if X_test is not None and len(X_test) else None
    return imp, X_tr, X_v, X_te


def print_section(title):
    log.info("=" * 60)
    log.info("  %s", title)
    log.info("=" * 60)


def _xgb_params(n_train):
    """Scale XGBoost hyperparams based on dataset size."""
    if n_train > 30_000:
        # 6-season full matrix (~60k rows) — deeper trees, more estimators
        return dict(
            n_estimators=1500, learning_rate=0.02, max_depth=6,
            min_child_weight=5, subsample=0.8, colsample_bytree=0.7,
            colsample_bylevel=0.8, reg_alpha=0.2, reg_lambda=2.0,
            gamma=0.1, early_stopping_rounds=75,
        )
    elif n_train > 10_000:
        # 2-3 seasons
        return dict(
            n_estimators=1000, learning_rate=0.025, max_depth=5,
            min_child_weight=4, subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.15, reg_lambda=1.5, gamma=0.05,
            early_stopping_rounds=60,
        )
    else:
        # Single season fallback
        return dict(
            n_estimators=600, learning_rate=0.03, max_depth=5,
            min_child_weight=3, subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.1, reg_lambda=1.5, early_stopping_rounds=50,
        )


# ──────────────────────────────────────────────────────────────────────────────
# MODEL 1: SPREAD (POINT MARGIN REGRESSION)
# ──────────────────────────────────────────────────────────────────────────────

def train_spread_model(train, val, test):
    print_section("SPREAD MODEL (Point Margin Regression)")

    X_train, y_train, feats = get_X_y(train, SPREAD_FEATURES, TARGET_SPREAD)
    X_val,   y_val,   _     = get_X_y(val,   feats,           TARGET_SPREAD)
    X_test,  y_test,  _     = get_X_y(test,  feats,           TARGET_SPREAD)

    log.info("Features used: %d / %d requested", len(feats), len(SPREAD_FEATURES))
    log.info("Train: %d  Val: %d  Test: %d", len(X_train), len(X_val), len(X_test))

    imp, X_tr, X_v, X_te = impute(X_train, X_val, X_test if len(X_test) else None)

    params = _xgb_params(len(X_train))
    log.info("XGBoost: n_estimators=%d  lr=%.3f  max_depth=%d",
             params["n_estimators"], params["learning_rate"], params["max_depth"])

    model = xgb.XGBRegressor(**params, eval_metric="mae",
                              random_state=42, n_jobs=-1, verbosity=0)
    model.fit(X_tr, y_train, eval_set=[(X_v, y_val)], verbose=100)

    val_pred = model.predict(X_v)
    val_mae  = mean_absolute_error(y_val, val_pred)
    baseline = y_val.abs().mean()
    val_side = ((val_pred > 0) == (y_val > 0)).mean()

    log.info("Val MAE:               %.2f pts", val_mae)
    log.info("Baseline MAE (0):      %.2f pts", baseline)
    log.info("Improvement:           %.1f%%", 100 * (1 - val_mae / baseline))
    log.info("Val win-side accuracy: %.1f%%", 100 * val_side)

    if X_te is not None and len(X_te):
        test_pred = model.predict(X_te)
        log.info("Test MAE:              %.2f pts", mean_absolute_error(y_test, test_pred))
        log.info("Test win-side acc:     %.1f%%", 100 * ((test_pred > 0) == (y_test > 0)).mean())

    fi = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False)
    log.info("Top 10 features (spread):\n%s", fi.head(10).to_string())

    path = MODELS_DIR / "spread_model.pkl"
    with open(path, "wb") as f:
        pickle.dump({"model": model, "imputer": imp, "features": feats}, f)
    log.info("Saved: %s", path)

    return {"val_mae": val_mae, "baseline_mae": baseline,
            "val_side_acc": val_side, "feature_importance": fi}


# ──────────────────────────────────────────────────────────────────────────────
# MODEL 2: WIN PROBABILITY (BINARY CLASSIFICATION)
# ──────────────────────────────────────────────────────────────────────────────

def train_win_prob_model(train, val, test):
    print_section("WIN PROBABILITY MODEL (Binary Classification)")

    X_train, y_train, feats = get_X_y(train, WIN_PROB_FEATURES, TARGET_WIN)
    X_val,   y_val,   _     = get_X_y(val,   feats,             TARGET_WIN)
    X_test,  y_test,  _     = get_X_y(test,  feats,             TARGET_WIN)

    log.info("Class balance — Train wins: %.1f%%", 100 * y_train.mean())
    log.info("Class balance — Val wins:   %.1f%%", 100 * y_val.mean())
    log.info("Train: %d  Val: %d  Test: %d", len(X_train), len(X_val), len(X_test))

    imp, X_tr, X_v, X_te = impute(X_train, X_val, X_test if len(X_test) else None)

    params = _xgb_params(len(X_train))
    params["max_depth"] = max(params["max_depth"] - 1, 4)  # classifiers prefer shallower

    model = xgb.XGBClassifier(**params, eval_metric="logloss",
                               use_label_encoder=False,
                               random_state=42, n_jobs=-1, verbosity=0)
    model.fit(X_tr, y_train, eval_set=[(X_v, y_val)], verbose=100)

    calibrated = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
    calibrated.fit(X_v, y_val)

    val_proba   = calibrated.predict_proba(X_v)[:, 1]
    val_pred    = (val_proba > 0.5).astype(int)
    val_auc     = roc_auc_score(y_val, val_proba)
    val_logloss = log_loss(y_val, val_proba)
    val_brier   = brier_score_loss(y_val, val_proba)
    val_acc     = accuracy_score(y_val, val_pred)
    baseline_ll = log_loss(y_val, np.full(len(y_val), y_train.mean()))

    log.info("Val AUC-ROC:        %.4f  (>0.65=good)", val_auc)
    log.info("Val Log-Loss:       %.4f  (baseline: %.4f)", val_logloss, baseline_ll)
    log.info("Val Brier Score:    %.4f", val_brier)
    log.info("Val Accuracy:       %.1f%%", 100 * val_acc)

    if X_te is not None and len(X_te):
        test_proba = calibrated.predict_proba(X_te)[:, 1]
        log.info("Test AUC-ROC:       %.4f", roc_auc_score(y_test, test_proba))
        log.info("Test Log-Loss:      %.4f", log_loss(y_test, test_proba))
        log.info("Test Accuracy:      %.1f%%", 100 * accuracy_score(y_test, (test_proba > 0.5)))

    fi = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False)
    log.info("Top 10 features (win prob):\n%s", fi.head(10).to_string())

    path = MODELS_DIR / "win_prob_model.pkl"
    with open(path, "wb") as f:
        pickle.dump({"model": calibrated, "imputer": imp, "features": feats}, f)
    log.info("Saved: %s", path)

    return {"val_auc": val_auc, "val_logloss": val_logloss,
            "val_accuracy": val_acc, "feature_importance": fi}


# ──────────────────────────────────────────────────────────────────────────────
# MODEL 3: GAME TOTALS (COMBINED SCORE REGRESSION)
# ──────────────────────────────────────────────────────────────────────────────

def train_totals_model(train, val, test):
    print_section("TOTALS MODEL (Combined Score Regression)")

    X_train, y_train, feats = get_X_y(train, TOTALS_FEATURES, TARGET_TOTAL)
    X_val,   y_val,   _     = get_X_y(val,   feats,           TARGET_TOTAL)
    X_test,  y_test,  _     = get_X_y(test,  feats,           TARGET_TOTAL)

    log.info("Target distribution — mean: %.1f  std: %.1f", y_train.mean(), y_train.std())
    log.info("Train: %d  Val: %d  Test: %d", len(X_train), len(X_val), len(X_test))

    imp, X_tr, X_v, X_te = impute(X_train, X_val, X_test if len(X_test) else None)

    params = _xgb_params(len(X_train))
    model = xgb.XGBRegressor(**params, eval_metric="mae",
                              random_state=42, n_jobs=-1, verbosity=0)
    model.fit(X_tr, y_train, eval_set=[(X_v, y_val)], verbose=100)

    val_pred = model.predict(X_v)
    val_mae  = mean_absolute_error(y_val, val_pred)
    baseline = (y_val - y_train.mean()).abs().mean()

    log.info("Val MAE:             %.2f pts", val_mae)
    log.info("Baseline MAE (mean): %.2f pts", baseline)
    log.info("Improvement:         %.1f%%", 100 * (1 - val_mae / baseline))

    if "vegas_total" in val.columns:
        vt       = val["vegas_total"].reindex(X_val.index)
        has_line = vt.notna()
        if has_line.sum() > 0:
            ou_acc = ((val_pred[has_line] > vt[has_line].values) ==
                      (y_val[has_line].values > vt[has_line].values)).mean()
            log.info("O/U accuracy vs Vegas: %.1f%% (%d games with line)",
                     100 * ou_acc, has_line.sum())

    if X_te is not None and len(X_te):
        log.info("Test MAE:            %.2f pts", mean_absolute_error(y_test, model.predict(X_te)))

    n_imp = len(model.feature_importances_)
    fi = pd.Series(model.feature_importances_, index=feats[:n_imp]).sort_values(ascending=False)
    log.info("Top 10 features (totals):\n%s", fi.head(10).to_string())

    path = MODELS_DIR / "totals_model.pkl"
    with open(path, "wb") as f:
        pickle.dump({"model": model, "imputer": imp, "features": feats}, f)
    log.info("Saved: %s", path)

    return {"val_mae": val_mae, "baseline_mae": baseline, "feature_importance": fi}


# ──────────────────────────────────────────────────────────────────────────────
# EVAL-ONLY MODE
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_saved_models(test):
    print_section("EVALUATING SAVED MODELS ON TEST SET")

    for model_name, feat_cols, target, mode in [
        ("spread_model",   SPREAD_FEATURES,  TARGET_SPREAD, "regression"),
        ("win_prob_model", WIN_PROB_FEATURES, TARGET_WIN,    "classification"),
        ("totals_model",   TOTALS_FEATURES,   TARGET_TOTAL,  "regression"),
    ]:
        path = MODELS_DIR / f"{model_name}.pkl"
        if not path.exists():
            log.warning("  %s not found — skipping", path)
            continue

        with open(path, "rb") as f:
            saved = pickle.load(f)

        X_test, y_test, _ = get_X_y(test, saved["features"], target)
        if len(X_test) == 0:
            log.warning("  %s: no test rows", model_name)
            continue

        X_imp = saved["imputer"].transform(X_test)
        model = saved["model"]

        if mode == "regression":
            pred = model.predict(X_imp)
            mae  = mean_absolute_error(y_test, pred)
            side = ((pred > 0) == (y_test > 0)).mean() if target == TARGET_SPREAD else None
            if side:
                log.info("  %-22s  Test MAE: %.2f  Side acc: %.1f%%", model_name, mae, 100*side)
            else:
                log.info("  %-22s  Test MAE: %.2f", model_name, mae)
        else:
            proba = model.predict_proba(X_imp)[:, 1]
            log.info("  %-22s  AUC: %.4f  LogLoss: %.4f  Acc: %.1f%%",
                     model_name, roc_auc_score(y_test, proba),
                     log_loss(y_test, proba),
                     100 * accuracy_score(y_test, (proba > 0.5)))


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NCAAB Model Trainer — Phase 5")
    parser.add_argument("--model", choices=["spread", "winprob", "totals", "all"],
                        default="all", help="Which model to train (default: all)")
    parser.add_argument("--eval", action="store_true",
                        help="Evaluate saved models on test set without retraining")
    parser.add_argument("--feature-path", type=Path, default=None,
                        help="Override feature matrix path (default: auto-detect)")
    args = parser.parse_args()

    feat_path = resolve_feature_path(args.feature_path)
    df = load_features(feat_path)

    log.info("Time-series split:")
    train, val, test = time_series_split(df)

    if len(train) < 100:
        log.error("Not enough training data (%d rows).", len(train))
        log.error("Run: python -m processing.historical_feature_builder")
        log.error("Falling back to proportional split...")
        cut65 = df["date"].quantile(0.65)
        cut85 = df["date"].quantile(0.85)
        train = df[df["date"] <= cut65]
        val   = df[(df["date"] > cut65) & (df["date"] <= cut85)]
        test  = df[df["date"] > cut85]

    if args.eval:
        evaluate_saved_models(test)
        return

    results = {}

    if args.model in ("spread", "all"):
        results["spread"] = train_spread_model(train, val, test)

    if args.model in ("winprob", "all"):
        results["winprob"] = train_win_prob_model(train, val, test)

    if args.model in ("totals", "all"):
        results["totals"] = train_totals_model(train, val, test)

    print_section("TRAINING COMPLETE — SUMMARY")
    if "spread" in results:
        r = results["spread"]
        log.info("  Spread   — Val MAE: %.2f pts  (baseline: %.2f)  Side acc: %.1f%%",
                 r["val_mae"], r["baseline_mae"], 100 * r["val_side_acc"])
    if "winprob" in results:
        r = results["winprob"]
        log.info("  Win Prob — Val AUC: %.4f  LogLoss: %.4f  Acc: %.1f%%",
                 r["val_auc"], r["val_logloss"], 100 * r["val_accuracy"])
    if "totals" in results:
        r = results["totals"]
        log.info("  Totals   — Val MAE: %.2f pts  (baseline: %.2f)",
                 r["val_mae"], r["baseline_mae"])

    log.info("Models saved to: %s", MODELS_DIR)
    log.info("Feature matrix used: %s", feat_path.name)


if __name__ == "__main__":
    main()