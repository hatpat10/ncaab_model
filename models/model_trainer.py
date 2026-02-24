"""
models/model_trainer.py
========================
NCAAB ML Prediction System — Phase 5: Model Training
2025-2026 Season • February 23, 2026

Trains three models from data/processed/feature_matrix.parquet:
  1. Spread model     — XGBoost regression → predicted point margin
  2. Win prob model   — XGBoost classification → P(home team wins)
  3. Totals model     — XGBoost regression → predicted combined score

All models use a strict time-series split (no lookahead):
  Train:      Nov 2025 – Dec 31 2025
  Validation: Jan 1 2026 – Jan 31 2026
  Test:       Feb 1 2026 – present

Usage:
    python -m models.model_trainer              # train all 3 models
    python -m models.model_trainer --model spread
    python -m models.model_trainer --model winprob
    python -m models.model_trainer --model totals
    python -m models.model_trainer --eval       # eval only (no retrain)
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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
ROOT       = Path(__file__).resolve().parent.parent
FEATURE_PATH = ROOT / "data" / "processed" / "feature_matrix.parquet"
MODELS_DIR = ROOT / "models" / "saved"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── Time-series split boundaries ─────────────────────────────────────────────
VAL_START  = "2026-01-01"
TEST_START = "2026-02-01"

# ──────────────────────────────────────────────────────────────────────────────
# FEATURE SETS
# ──────────────────────────────────────────────────────────────────────────────

# Core features available in every game row (from feature_builder.py)
SPREAD_FEATURES = [
    # BartTorvik efficiency deltas — most predictive signals
    "off_eff_delta",          # team AdjO − opp AdjD
    "def_eff_delta",          # team AdjD − opp AdjO (inverted)
    "net_rating_delta",       # (AdjO−AdjD) team − (AdjO−AdjD) opp
    "tempo_delta",            # AdjT difference → pace matchup
    "def_suppression_factor", # opponent defense tier adjustment
    "wab_delta",              # wins above bubble delta
    "sos_delta",              # strength of schedule delta

    # BartTorvik absolute ratings
    "t_adj_o",   "t_adj_d",   "t_adj_t",   "t_barthag",
    "o_adj_o",   "o_adj_d",   "o_adj_t",   "o_barthag",

    # Rolling form
    "roll5_pts",     "roll5_margin",  "roll5_win_streak",
    "roll10_pts",    "roll10_margin",

    # Situational
    "is_home",       "is_neutral",    "days_rest",
    "rest_advantage","is_back_to_back","is_short_rest",
    "games_played",

    # Head-to-head
    "h2h_avg_margin", "h2h_win_rate",

    # SOS context
    "t_ov_cur_sos",  "o_ov_cur_sos",
    "t_nc_cur_sos",  "o_nc_cur_sos",
]

# Totals model adds market signal when available (tempo already in SPREAD_FEATURES)
TOTALS_FEATURES = SPREAD_FEATURES + [
    "vegas_total",  # market signal when available
]

# Win prob uses same features as spread (different target)
WIN_PROB_FEATURES = SPREAD_FEATURES

# Target variables
TARGET_SPREAD   = "margin"      # actual_score − opp_score (team perspective)
TARGET_WIN      = "win"         # 1 if team won, 0 if lost
TARGET_TOTAL    = "total_score" # combined final score

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def load_features() -> pd.DataFrame:
    """Load feature matrix, add derived targets, basic validation."""
    log.info("Loading feature matrix: %s", FEATURE_PATH)
    df = pd.read_parquet(FEATURE_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # ── Derive targets ────────────────────────────────────────────────────────
    # margin = team_score − opp_score (positive = team won)
    score_col = next((c for c in ["team_score", "score", "pts"] if c in df.columns), None)
    opp_score_col = next((c for c in ["opp_score", "opp_pts"] if c in df.columns), None)

    if score_col and opp_score_col:
        df["margin"] = df[score_col] - df[opp_score_col]
        df["win"]    = (df["margin"] > 0).astype(int)
        df["total_score"] = df[score_col] + df[opp_score_col]
    else:
        # Fallback: try roll5_pts as proxy — but warn
        log.warning("team_score/opp_score not found. Checking available score columns: %s",
                    [c for c in df.columns if "score" in c.lower() or "pts" in c.lower()])
        raise ValueError(
            "Cannot find score columns to build targets. "
            "Need 'team_score' and 'opp_score' in feature matrix."
        )

    # Drop rows without valid targets
    before = len(df)
    df = df.dropna(subset=["margin", "win", "total_score"])
    log.info("Rows with valid targets: %d (dropped %d without scores)", len(df), before - len(df))

    log.info("Feature matrix: %d rows, %d cols | %s → %s",
             len(df), len(df.columns),
             df["date"].min().date(), df["date"].max().date())
    return df


def time_series_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Strict temporal train/val/test split."""
    df = df.sort_values("date")
    val_dt  = pd.to_datetime(VAL_START)
    test_dt = pd.to_datetime(TEST_START)

    train = df[df["date"] <  val_dt]
    val   = df[(df["date"] >= val_dt) & (df["date"] < test_dt)]
    test  = df[df["date"] >= test_dt]

    for name, split in [("Train", train), ("Val", val), ("Test", test)]:
        if len(split):
            log.info("  %-6s %5d rows  (%s → %s)",
                     name + ":", len(split),
                     split["date"].min().date(), split["date"].max().date())
        else:
            log.warning("  %s: 0 rows", name)

    return train, val, test


def get_X_y(df: pd.DataFrame, feature_cols: list[str], target: str):
    """Extract X, y — only use features that actually exist, deduped."""
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for c in feature_cols:
        if c not in seen:
            seen.add(c)
            deduped.append(c)

    available = [c for c in deduped if c in df.columns]
    missing   = [c for c in deduped if c not in df.columns]
    if missing:
        log.debug("Features not in matrix (will skip): %s", missing)

    X = df[available].copy()
    y = df[target].copy()
    return X, y, available


def build_pipeline(model) -> Pipeline:
    """Wrap model in imputer → scaler pipeline."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   model),
    ])


def print_section(title: str):
    log.info("=" * 60)
    log.info("  %s", title)
    log.info("=" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# MODEL 1: SPREAD (POINT MARGIN REGRESSION)
# ──────────────────────────────────────────────────────────────────────────────

def train_spread_model(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> dict:
    print_section("SPREAD MODEL (Point Margin Regression)")

    X_train, y_train, feats = get_X_y(train, SPREAD_FEATURES, TARGET_SPREAD)
    X_val,   y_val,   _     = get_X_y(val,   feats,           TARGET_SPREAD)
    X_test,  y_test,  _     = get_X_y(test,  feats,           TARGET_SPREAD)

    log.info("Features used: %d / %d requested", len(feats), len(SPREAD_FEATURES))
    log.info("Train: %d  Val: %d  Test: %d", len(X_train), len(X_val), len(X_test))

    # Impute before fitting XGBoost (handles NaN internally but pipeline is cleaner)
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp   = imputer.transform(X_val)
    X_test_imp  = imputer.transform(X_test)

    model = xgb.XGBRegressor(
        n_estimators          = 600,
        learning_rate         = 0.03,
        max_depth             = 5,
        min_child_weight      = 3,
        subsample             = 0.8,
        colsample_bytree      = 0.7,
        reg_alpha             = 0.1,
        reg_lambda            = 1.5,
        early_stopping_rounds = 50,
        eval_metric           = "mae",
        random_state          = 42,
        n_jobs                = -1,
        verbosity             = 0,
    )

    model.fit(
        X_train_imp, y_train,
        eval_set=[(X_val_imp, y_val)],
        verbose=100,
    )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    val_pred  = model.predict(X_val_imp)
    val_mae   = mean_absolute_error(y_val, val_pred)
    baseline  = y_val.abs().mean()  # MAE of always predicting 0

    log.info("Val MAE:           %.2f pts", val_mae)
    log.info("Baseline MAE (0):  %.2f pts", baseline)
    log.info("Improvement:       %.1f%%", 100 * (1 - val_mae / baseline))

    # ATS accuracy on val (did model pick the right side?)
    val_correct_side = ((val_pred > 0) == (y_val > 0)).mean()
    log.info("Val win-side accuracy: %.1f%%", 100 * val_correct_side)

    if len(X_test_imp):
        test_pred = model.predict(X_test_imp)
        test_mae  = mean_absolute_error(y_test, test_pred)
        test_side = ((test_pred > 0) == (y_test > 0)).mean()
        log.info("Test MAE:              %.2f pts", test_mae)
        log.info("Test win-side acc:     %.1f%%", 100 * test_side)

    # Feature importance
    fi = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False)
    log.info("Top 10 features (spread):\n%s", fi.head(10).to_string())

    # Save
    out = {"model": model, "imputer": imputer, "features": feats}
    path = MODELS_DIR / "spread_model.pkl"
    with open(path, "wb") as f:
        pickle.dump(out, f)
    log.info("Saved: %s", path)

    return {"val_mae": val_mae, "baseline_mae": baseline,
            "val_side_acc": val_correct_side, "feature_importance": fi}


# ──────────────────────────────────────────────────────────────────────────────
# MODEL 2: WIN PROBABILITY (BINARY CLASSIFICATION)
# ──────────────────────────────────────────────────────────────────────────────

def train_win_prob_model(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> dict:
    print_section("WIN PROBABILITY MODEL (Binary Classification)")

    X_train, y_train, feats = get_X_y(train, WIN_PROB_FEATURES, TARGET_WIN)
    X_val,   y_val,   _     = get_X_y(val,   feats,             TARGET_WIN)
    X_test,  y_test,  _     = get_X_y(test,  feats,             TARGET_WIN)

    log.info("Class balance — Train wins: %.1f%%", 100 * y_train.mean())
    log.info("Class balance — Val wins:   %.1f%%", 100 * y_val.mean())

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp   = imputer.transform(X_val)
    X_test_imp  = imputer.transform(X_test) if len(X_test) else X_test_imp

    model = xgb.XGBClassifier(
        n_estimators          = 500,
        learning_rate         = 0.05,
        max_depth             = 4,
        min_child_weight      = 3,
        subsample             = 0.8,
        colsample_bytree      = 0.7,
        reg_alpha             = 0.1,
        reg_lambda            = 1.0,
        early_stopping_rounds = 50,
        eval_metric           = "logloss",
        use_label_encoder     = False,
        random_state          = 42,
        n_jobs                = -1,
        verbosity             = 0,
    )

    model.fit(
        X_train_imp, y_train,
        eval_set=[(X_val_imp, y_val)],
        verbose=100,
    )

    # Calibrate so P(win)=0.65 actually means 65% of the time
    calibrated = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
    calibrated.fit(X_val_imp, y_val)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    val_proba = calibrated.predict_proba(X_val_imp)[:, 1]
    val_pred  = (val_proba > 0.5).astype(int)

    val_auc      = roc_auc_score(y_val, val_proba)
    val_logloss  = log_loss(y_val, val_proba)
    val_brier    = brier_score_loss(y_val, val_proba)
    val_acc      = accuracy_score(y_val, val_pred)
    baseline_ll  = log_loss(y_val, np.full(len(y_val), y_train.mean()))

    log.info("Val AUC-ROC:        %.4f  (0.5 = random, 1.0 = perfect)", val_auc)
    log.info("Val Log-Loss:       %.4f  (baseline: %.4f)", val_logloss, baseline_ll)
    log.info("Val Brier Score:    %.4f", val_brier)
    log.info("Val Accuracy:       %.1f%%", 100 * val_acc)

    if len(X_test_imp):
        test_proba = calibrated.predict_proba(X_test_imp)[:, 1]
        log.info("Test AUC-ROC:       %.4f", roc_auc_score(y_test, test_proba))
        log.info("Test Log-Loss:      %.4f", log_loss(y_test, test_proba))

    fi = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False)
    log.info("Top 10 features (win prob):\n%s", fi.head(10).to_string())

    # Save
    out = {"model": calibrated, "imputer": imputer, "features": feats}
    path = MODELS_DIR / "win_prob_model.pkl"
    with open(path, "wb") as f:
        pickle.dump(out, f)
    log.info("Saved: %s", path)

    return {"val_auc": val_auc, "val_logloss": val_logloss,
            "val_accuracy": val_acc, "feature_importance": fi}


# ──────────────────────────────────────────────────────────────────────────────
# MODEL 3: GAME TOTALS (COMBINED SCORE REGRESSION)
# ──────────────────────────────────────────────────────────────────────────────

def train_totals_model(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> dict:
    print_section("TOTALS MODEL (Combined Score Regression)")

    X_train, y_train, feats = get_X_y(train, TOTALS_FEATURES, TARGET_TOTAL)
    X_val,   y_val,   _     = get_X_y(val,   feats,           TARGET_TOTAL)
    X_test,  y_test,  _     = get_X_y(test,  feats,           TARGET_TOTAL)

    log.info("Target distribution — mean: %.1f  std: %.1f",
             y_train.mean(), y_train.std())

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp   = imputer.transform(X_val)

    model = xgb.XGBRegressor(
        n_estimators          = 600,
        learning_rate         = 0.03,
        max_depth             = 5,
        min_child_weight      = 3,
        subsample             = 0.8,
        colsample_bytree      = 0.7,
        reg_alpha             = 0.1,
        reg_lambda            = 1.5,
        early_stopping_rounds = 50,
        eval_metric           = "mae",
        random_state          = 42,
        n_jobs                = -1,
        verbosity             = 0,
    )

    model.fit(
        X_train_imp, y_train,
        eval_set=[(X_val_imp, y_val)],
        verbose=100,
    )

    val_pred = model.predict(X_val_imp)
    val_mae  = mean_absolute_error(y_val, val_pred)
    baseline = (y_val - y_train.mean()).abs().mean()  # MAE of predicting mean

    log.info("Val MAE:             %.2f pts", val_mae)
    log.info("Baseline MAE (mean): %.2f pts", baseline)
    log.info("Improvement:         %.1f%%", 100 * (1 - val_mae / baseline))

    # Over/Under accuracy (vs vegas total when available)
    if "vegas_total" in val.columns:
        vt = val["vegas_total"].reindex(X_val.index)
        has_line = vt.notna()
        if has_line.sum() > 0:
            ou_correct = ((val_pred[has_line] > vt[has_line]) ==
                         (y_val[has_line] > vt[has_line])).mean()
            log.info("O/U accuracy vs Vegas: %.1f%% (%d games with line)",
                     100 * ou_correct, has_line.sum())

    if len(X_test) and len(X_test.dropna()) > 0:
        X_test_imp = imputer.transform(X_test)
        test_mae   = mean_absolute_error(y_test, model.predict(X_test_imp))
        log.info("Test MAE:            %.2f pts", test_mae)

    # Guard against length mismatch (can happen if XGBoost drops constant features)
    n_imp = len(model.feature_importances_)
    fi = pd.Series(model.feature_importances_, index=feats[:n_imp]).sort_values(ascending=False)
    log.info("Top 10 features (totals):\n%s", fi.head(10).to_string())

    out = {"model": model, "imputer": imputer, "features": feats}
    path = MODELS_DIR / "totals_model.pkl"
    with open(path, "wb") as f:
        pickle.dump(out, f)
    log.info("Saved: %s", path)

    return {"val_mae": val_mae, "baseline_mae": baseline, "feature_importance": fi}


# ──────────────────────────────────────────────────────────────────────────────
# EVAL-ONLY MODE
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_saved_models(test: pd.DataFrame):
    """Load saved models and evaluate on test set."""
    print_section("EVALUATING SAVED MODELS ON TEST SET")

    for model_name, feat_cols, target, mode in [
        ("spread_model",   SPREAD_FEATURES,   TARGET_SPREAD, "regression"),
        ("win_prob_model", WIN_PROB_FEATURES,  TARGET_WIN,    "classification"),
        ("totals_model",   TOTALS_FEATURES,    TARGET_TOTAL,  "regression"),
    ]:
        path = MODELS_DIR / f"{model_name}.pkl"
        if not path.exists():
            log.warning("  %s not found — skipping", path)
            continue

        with open(path, "rb") as f:
            saved = pickle.load(f)

        feats   = saved["features"]
        imputer = saved["imputer"]
        model   = saved["model"]

        X_test, y_test, _ = get_X_y(test, feats, target)
        if len(X_test) == 0:
            log.warning("  %s: no test rows", model_name)
            continue

        X_imp = imputer.transform(X_test)

        if mode == "regression":
            pred = model.predict(X_imp)
            mae  = mean_absolute_error(y_test, pred)
            log.info("  %-20s  Test MAE: %.2f", model_name, mae)
        else:
            proba = model.predict_proba(X_imp)[:, 1]
            auc   = roc_auc_score(y_test, proba)
            ll    = log_loss(y_test, proba)
            log.info("  %-20s  AUC: %.4f  LogLoss: %.4f", model_name, auc, ll)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NCAAB Model Trainer — Phase 5")
    parser.add_argument("--model", choices=["spread", "winprob", "totals", "all"],
                        default="all", help="Which model to train")
    parser.add_argument("--eval", action="store_true",
                        help="Evaluate saved models on test set (no training)")
    parser.add_argument("--feature-path", type=Path, default=FEATURE_PATH,
                        help="Path to feature_matrix.parquet")
    args = parser.parse_args()

    # ── Load data ─────────────────────────────────────────────────────────────
    df = load_features()
    log.info("Time-series split:")
    train, val, test = time_series_split(df)

    if len(train) < 100:
        log.error("Not enough training data (%d rows). Need at least 100.", len(train))
        log.error("Your feature matrix only covers the current season (Nov 2025–present).")
        log.error("To train properly, add historical seasons to hoopR data and rebuild.")
        log.error("For now, using all data for training (no val/test split).")
        # Fallback: use 80/20 split within current season
        split_date = df["date"].quantile(0.65)
        train = df[df["date"] <= split_date]
        val   = df[(df["date"] > split_date) & (df["date"] <= df["date"].quantile(0.85))]
        test  = df[df["date"]  > df["date"].quantile(0.85)]
        log.info("Fallback split — Train: %d  Val: %d  Test: %d",
                 len(train), len(val), len(test))

    if args.eval:
        evaluate_saved_models(test)
        return

    results = {}

    # ── Train selected model(s) ───────────────────────────────────────────────
    if args.model in ("spread", "all"):
        results["spread"] = train_spread_model(train, val, test)

    if args.model in ("winprob", "all"):
        results["winprob"] = train_win_prob_model(train, val, test)

    if args.model in ("totals", "all"):
        results["totals"] = train_totals_model(train, val, test)

    # ── Summary ───────────────────────────────────────────────────────────────
    print_section("TRAINING COMPLETE — SUMMARY")
    if "spread" in results:
        r = results["spread"]
        log.info("  Spread  — Val MAE: %.2f pts  (baseline: %.2f)  Side acc: %.1f%%",
                 r["val_mae"], r["baseline_mae"], 100 * r["val_side_acc"])
    if "winprob" in results:
        r = results["winprob"]
        log.info("  Win Prob— Val AUC: %.4f  LogLoss: %.4f  Acc: %.1f%%",
                 r["val_auc"], r["val_logloss"], 100 * r["val_accuracy"])
    if "totals" in results:
        r = results["totals"]
        log.info("  Totals  — Val MAE: %.2f pts  (baseline: %.2f)",
                 r["val_mae"], r["baseline_mae"])

    log.info("Models saved to: %s", MODELS_DIR)


if __name__ == "__main__":
    main()