"""
predictions/daily_pipeline.py
==============================
NCAAB ML â€” Daily Prediction Pipeline
2025-2026 Season â€¢ February 2026

Run each morning to generate predictions for tonight's slate.
Fetches today's schedule from ESPN, builds features for each game,
runs all three models, compares to Vegas lines, and saves outputs.

Usage:
    python -m predictions.daily_pipeline               # today
    python -m predictions.daily_pipeline --date 2026-02-24
    python -m predictions.daily_pipeline --date 2026-02-24 --pretty

Output:
    predictions/YYYY-MM-DD_predictions.json
    predictions/YYYY-MM-DD_predictions.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sqlite3
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

# â”€â”€ Logging â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# â”€â”€ Paths â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
ROOT         = Path(__file__).resolve().parent.parent
MODELS_DIR   = ROOT / "models" / "saved"
PRED_DIR     = ROOT / "predictions"
DATA_PROC    = ROOT / "data" / "processed"
ALIASES_PATH = ROOT / "data" / "team_aliases.json"
DB_PATH      = ROOT / "data" / "ncaab.db"
CONF_LOOKUP_PATH = ROOT / "data" / "processed" / "espn_conf_lookup.json"

_CONF_LOOKUP: dict = {}

def _load_conf_lookup() -> dict:
    global _CONF_LOOKUP
    if _CONF_LOOKUP:
        return _CONF_LOOKUP
    if CONF_LOOKUP_PATH.exists():
        with open(CONF_LOOKUP_PATH) as f:
            _CONF_LOOKUP = json.load(f)
    return _CONF_LOOKUP

def get_game_tier(home_team: str, away_team: str) -> str:
    """Return the WEAKEST tier among both teams (low-major contaminates the game)."""
    lookup = _load_conf_lookup()
    home_tier = lookup.get(home_team, {}).get("tier", "unknown")
    away_tier = lookup.get(away_team, {}).get("tier", "unknown")
    tier_rank = {"low": 0, "mid": 1, "high": 2, "unknown": 3}
    weakest = min(home_tier, away_tier, key=lambda t: tier_rank.get(t, 3))
    return weakest
PRED_DIR.mkdir(parents=True, exist_ok=True)

# â”€â”€ ESPN API â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
ESPN_BASE    = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"
ESPN_HEADERS = {"User-Agent": "Mozilla/5.0"}

# â”€â”€ Odds API â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# TEAM ALIASES
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_ALIASES: dict = {}

def _load_aliases() -> dict:
    global _ALIASES
    if _ALIASES:
        return _ALIASES
    if ALIASES_PATH.exists():
        with open(ALIASES_PATH) as f:
            _ALIASES = json.load(f)
    return _ALIASES

def normalize_team(name: str) -> str:
    """
    Normalize a team name to canonical slug using aliases.
    Handles ESPN full display names like 'Kansas Jayhawks' -> 'kansas'.
    Falls back to progressive mascot-stripping if exact match not found.
    """
    aliases = _load_aliases()
    if not isinstance(name, str):
        return str(name)
    name = name.strip()

    # 1. Direct alias lookup
    if name in aliases:
        return aliases[name]

    # 2. Case-insensitive lookup
    lower = name.lower()
    for k, v in aliases.items():
        if k.lower() == lower:
            return v

    # 3. Progressive mascot stripping: drop last 1, 2, 3 words
    #    "Kansas Jayhawks" -> "Kansas" -> aliases["Kansas"] = "kansas"
    words = name.split()
    for n_drop in range(1, min(4, len(words))):
        shorter = " ".join(words[:-n_drop])
        if shorter in aliases:
            return aliases[shorter]
        for k, v in aliases.items():
            if k.lower() == shorter.lower():
                return v

    # 4. Slug fallback (remove punctuation, spaces -> underscores)
    slug = lower.replace(" ", "_").replace("-", "_").replace(".", "").replace("'", "").replace("&", "")
    if slug in aliases:
        return aliases[slug]

    # 5. Return the slug as-is (best effort)
    return slug


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: FETCH TODAY'S SCHEDULE FROM ESPN
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def fetch_todays_schedule(date_str: str) -> list[dict]:
    """
    Fetch upcoming/completed games from ESPN for a given date.
    date_str: YYYY-MM-DD
    Returns list of game dicts.
    """
    espn_date = date_str.replace("-", "")
    url = f"{ESPN_BASE}/scoreboard"
    params = {"limit": 200, "dates": espn_date, "groups": 50}

    log.info("Fetching ESPN schedule for %s...", date_str)
    try:
        r = requests.get(url, params=params, headers=ESPN_HEADERS, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log.error("ESPN fetch failed: %s", e)
        return []

    events = data.get("events", [])
    games = []
    for event in events:
        try:
            game = _parse_espn_event(event)
            if game:
                games.append(game)
        except Exception as e:
            log.debug("Error parsing event %s: %s", event.get("id"), e)

    log.info("Found %d games on ESPN for %s", len(games), date_str)
    return games


def _parse_espn_event(event: dict) -> dict | None:
    """Parse a single ESPN event into a flat game dict."""
    comp   = event["competitions"][0]
    home   = next((t for t in comp["competitors"] if t["homeAway"] == "home"), None)
    away   = next((t for t in comp["competitors"] if t["homeAway"] == "away"), None)
    if not home or not away:
        return None

    status    = event["status"]["type"]
    completed = status.get("completed", False)
    state     = status.get("name", "")  # "STATUS_SCHEDULED", "STATUS_IN_PROGRESS", etc.

    return {
        "game_id":     str(event["id"]),
        "date":        event["date"][:10],
        "tipoff_time": event["date"],
        "home_team":   home["team"]["displayName"],
        "away_team":   away["team"]["displayName"],
        "home_team_id": str(home["id"]),
        "away_team_id": str(away["id"]),
        "home_score":  int(home.get("score", 0) or 0),
        "away_score":  int(away.get("score", 0) or 0),
        "neutral":     int(comp.get("neutralSite", False)),
        "completed":   int(completed),
        "status":      state,
        "venue":       comp.get("venue", {}).get("fullName", ""),
        "home_team_norm": normalize_team(home["team"]["displayName"]),
        "away_team_norm": normalize_team(away["team"]["displayName"]),
    }


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2: FETCH CURRENT ODDS FROM THE ODDS API
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def fetch_current_odds() -> pd.DataFrame:
    """
    Pull current NCAAB odds from The Odds API.
    Returns DataFrame with columns: home_team, away_team, spread, total, home_ml, away_ml
    """
    if not ODDS_API_KEY:
        log.warning("ODDS_API_KEY not set â€” skipping odds fetch")
        return pd.DataFrame()

    url = "https://api.the-odds-api.com/v4/sports/basketball_ncaab/odds"
    params = {
        "apiKey":      ODDS_API_KEY,
        "regions":     "us",
        "markets":     "h2h,spreads,totals",
        "oddsFormat":  "american",
        "bookmakers":  "draftkings,fanduel,betmgm,caesars",
    }

    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        remaining = r.headers.get("x-requests-remaining", "?")
        log.info("Odds API: %s requests remaining", remaining)
        data = r.json()
    except Exception as e:
        log.error("Odds API fetch failed: %s", e)
        return pd.DataFrame()

    rows = []
    for game in data:
        row = {
            "home_team": normalize_team(game.get("home_team", "")),
            "away_team": normalize_team(game.get("away_team", "")),
            "commence_time": game.get("commence_time", ""),
            "spread":   None,
            "total":    None,
            "home_ml":  None,
            "away_ml":  None,
        }
        # Average across bookmakers
        spreads, totals, home_mls, away_mls = [], [], [], []

        for bk in game.get("bookmakers", []):
            for market in bk.get("markets", []):
                key = market["key"]
                outcomes = {o["name"]: o for o in market.get("outcomes", [])}
                ht = game.get("home_team", "")
                at = game.get("away_team", "")

                if key == "spreads":
                    if ht in outcomes:
                        spreads.append(outcomes[ht].get("point", 0))
                elif key == "totals":
                    over = next((o for o in market["outcomes"] if o["name"] == "Over"), None)
                    if over:
                        totals.append(over.get("point", 0))
                elif key == "h2h":
                    if ht in outcomes:
                        home_mls.append(outcomes[ht].get("price", 0))
                    if at in outcomes:
                        away_mls.append(outcomes[at].get("price", 0))

        if spreads: row["spread"] = round(np.mean(spreads), 1)
        if totals:  row["total"]  = round(np.mean(totals), 1)
        if home_mls: row["home_ml"] = round(np.mean(home_mls))
        if away_mls: row["away_ml"] = round(np.mean(away_mls))
        rows.append(row)

    df = pd.DataFrame(rows)
    log.info("Fetched odds for %d games", len(df))
    return df


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3: BUILD FEATURES FOR EACH GAME
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def load_feature_matrix() -> pd.DataFrame:
    """Load the most recent full feature matrix."""
    full_path = DATA_PROC / "feature_matrix_full.parquet"
    curr_path = DATA_PROC / "feature_matrix.parquet"
    path = full_path if full_path.exists() else curr_path
    log.info("Loading feature matrix: %s", path.name)
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def build_game_features(home_team: str, away_team: str, is_neutral: int,
                        target_date: str, feature_matrix: pd.DataFrame,
                        odds_row: dict | None = None) -> pd.Series | None:
    """
    Build a feature vector for an upcoming game from the home team's perspective.

    Strategy: Find the most recent game row for each team, then construct
    the full feature vector the same way the training data was built:
      - t_* columns  = home team's BartTorvik ratings
      - o_* columns  = away team's BartTorvik ratings
      - roll*        = home team's rolling stats
      - deltas       = home minus away for matchup features
      - situational  = rest, home court, etc.

    The feature matrix rows are team-perspective (each game = 2 rows).
    We grab the home team's most recent home-perspective row as the base,
    then patch in away team stats for opponent columns.
    """
    target_dt = pd.to_datetime(target_date)
    home_norm = normalize_team(home_team)
    away_norm = normalize_team(away_team)

    # Find team column (feature_builder uses team_id)
    team_col = next((c for c in ["team_id", "team_norm", "team"] if c in feature_matrix.columns), None)
    if team_col is None:
        log.error("Cannot find team column. Available: %s", feature_matrix.columns.tolist()[:15])
        return None

    # Filter to games before target date (no leakage)
    past = feature_matrix[feature_matrix["date"] < target_dt]
    if len(past) == 0:
        past = feature_matrix

    # â”€â”€ Get home team's recent rows (home-perspective preferred) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    home_all  = past[past[team_col] == home_norm].sort_values("date")
    away_all  = past[past[team_col] == away_norm].sort_values("date")

    if len(home_all) == 0:
        log.warning("No history found for home team: %s (%s)", home_team, home_norm)
        return None
    if len(away_all) == 0:
        log.warning("No history found for away team: %s (%s)", away_team, away_norm)
        return None

    # Use most recent row for each team.
    # For BartTorvik stats, prefer the most recent row WHERE t_adj_o is not null.
    # For rolling stats, use the most recent row (may differ from BartTorvik row).
    home_row = home_all.iloc[-1]
    away_row = away_all.iloc[-1]

    # Find best BartTorvik rows (most recent with valid t_adj_o)
    home_torvik = home_all[home_all["t_adj_o"].notna()]
    away_torvik = away_all[away_all["t_adj_o"].notna()]
    home_tv_row = home_torvik.iloc[-1] if len(home_torvik) > 0 else home_row
    away_tv_row = away_torvik.iloc[-1] if len(away_torvik) > 0 else away_row

    # â”€â”€ Build the feature vector â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    feat = {}

    # --- Situational (set directly for this specific game) ---
    home_rest = _days_since_last_game(home_all, target_dt)
    away_rest = _days_since_last_game(away_all, target_dt)

    feat["is_home"]          = 0 if is_neutral else 1
    feat["is_neutral"]       = int(is_neutral)
    feat["days_rest"]        = home_rest
    feat["rest_advantage"]   = home_rest - away_rest
    feat["is_back_to_back"]  = int(home_rest <= 1)
    feat["is_short_rest"]    = int(home_rest <= 2)
    feat["games_played"]     = int(home_row.get("games_played", 20))

    # --- BartTorvik absolute ratings ---
    # Use torvik-specific rows (most recent row with valid BartTorvik data)
    for col in ["t_adj_o", "t_adj_d", "t_adj_t", "t_barthag"]:
        feat[col] = home_tv_row.get(col, np.nan)

    # Away team's "t_" columns become our "o_" columns
    feat["o_adj_o"]   = away_tv_row.get("t_adj_o", np.nan)
    feat["o_adj_d"]   = away_tv_row.get("t_adj_d", np.nan)
    feat["o_adj_t"]   = away_tv_row.get("t_adj_t", np.nan)
    feat["o_barthag"] = away_tv_row.get("t_barthag", np.nan)

    # --- BartTorvik deltas (home perspective) ---
    h_o = home_tv_row.get("t_adj_o", np.nan)
    h_d = home_tv_row.get("t_adj_d", np.nan)
    h_t = home_tv_row.get("t_adj_t", np.nan)
    a_o = away_tv_row.get("t_adj_o", np.nan)
    a_d = away_tv_row.get("t_adj_d", np.nan)
    a_t = away_tv_row.get("t_adj_t", np.nan)

    feat["net_rating_delta"]    = (h_o - h_d) - (a_o - a_d) if not any(pd.isna([h_o,h_d,a_o,a_d])) else np.nan
    feat["off_eff_delta"]       = h_o - a_d if not any(pd.isna([h_o, a_d])) else np.nan
    feat["def_eff_delta"]       = a_o - h_d if not any(pd.isna([a_o, h_d])) else np.nan
    feat["tempo_delta"]         = h_t - a_t if not any(pd.isna([h_t, a_t])) else np.nan
    feat["def_suppression_factor"] = home_row.get("def_suppression_factor", np.nan)

    # WAB and SOS deltas (use torvik rows)
    h_wab = home_tv_row.get("t_wab", np.nan)
    a_wab = away_tv_row.get("t_wab", np.nan)
    feat["wab_delta"] = h_wab - a_wab if not any(pd.isna([h_wab, a_wab])) else np.nan

    h_sos = home_tv_row.get("t_ov_cur_sos", np.nan)
    a_sos = away_tv_row.get("t_ov_cur_sos", np.nan)
    feat["sos_delta"] = h_sos - a_sos if not any(pd.isna([h_sos, a_sos])) else np.nan

    # --- Rolling form (home team's recent performance) ---
    for col in ["roll5_pts", "roll5_margin", "roll5_win_streak",
                "roll10_pts", "roll10_margin"]:
        feat[col] = home_row.get(col, np.nan)

    # --- SOS context (home team's schedule context) ---
    feat["t_ov_cur_sos"] = home_tv_row.get("t_ov_cur_sos", np.nan)
    feat["t_nc_cur_sos"] = home_tv_row.get("t_nc_cur_sos", np.nan)
    # Opponent SOS context (away team's t_ columns become our o_ columns)
    feat["o_ov_cur_sos"] = away_tv_row.get("t_ov_cur_sos", np.nan)
    feat["o_nc_cur_sos"] = away_tv_row.get("t_nc_cur_sos", np.nan)

    # --- Head-to-head: look for recent H2H between these two teams ---
    # Find games where home played away or away played home
    h2h_home = past[
        (past[team_col] == home_norm) &
        (past.get("opp_id", pd.Series(dtype=str)) == away_norm if "opp_id" in past.columns else False)
    ].sort_values("date").tail(5)

    if len(h2h_home) > 0 and "margin" in h2h_home.columns:
        feat["h2h_avg_margin"] = h2h_home["margin"].mean()
        feat["h2h_win_rate"]   = (h2h_home["margin"] > 0).mean()
    else:
        feat["h2h_avg_margin"] = home_row.get("h2h_avg_margin", 0.0)
        feat["h2h_win_rate"]   = home_row.get("h2h_win_rate", 0.5)

    # --- Vegas line (for totals model) ---
    feat["vegas_total"] = odds_row.get("total") if odds_row else np.nan

    # Debug log key features to verify correctness
    log.debug(
        "  Features [%s @ %s]: net_delta=%.1f  h_adjO=%.1f  h_adjD=%.1f  "
        "a_adjO=%.1f  a_adjD=%.1f  roll5_margin=%.1f  roll10_margin=%.1f  "
        "tv_rows_home=%d  tv_rows_away=%d",
        away_team, home_team,
        feat.get("net_rating_delta", float("nan")),
        feat.get("t_adj_o", float("nan")),
        feat.get("t_adj_d", float("nan")),
        feat.get("o_adj_o", float("nan")),
        feat.get("o_adj_d", float("nan")),
        feat.get("roll5_margin", float("nan")),
        feat.get("roll10_margin", float("nan")),
        len(home_torvik),
        len(away_torvik),
    )

    return pd.Series(feat)


def _days_since_last_game(team_rows: pd.DataFrame, target_date: pd.Timestamp) -> int:
    """Days between target date and team's most recent game."""
    if len(team_rows) == 0:
        return 3  # assume normal rest if unknown
    last_game = team_rows["date"].max()
    delta = (target_date - last_game).days
    return max(0, min(delta, 14))  # cap at 14


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 4: LOAD MODELS AND PREDICT
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def load_models() -> dict:
    """Load all three saved models."""
    models = {}
    for name in ["spread_model", "win_prob_model", "totals_model"]:
        path = MODELS_DIR / f"{name}.pkl"
        if path.exists():
            with open(path, "rb") as f:
                models[name] = pickle.load(f)
            log.info("Loaded: %s", name)
        else:
            log.warning("Model not found: %s", path)
    return models


def predict_game(features: pd.Series, models: dict) -> dict:
    """Run all three models on a feature vector. Returns prediction dict."""
    preds = {}

    for model_name, saved in models.items():
        model   = saved["model"]
        imputer = saved["imputer"]
        feats   = saved["features"]

        # Align feature vector to model's expected columns
        X = features.reindex(feats).values.reshape(1, -1)
        X_imp = imputer.transform(X)

        try:
            if model_name == "spread_model":
                preds["predicted_margin"] = round(float(model.predict(X_imp)[0]), 1)
            elif model_name == "win_prob_model":
                prob = float(model.predict_proba(X_imp)[0][1])
                preds["home_win_prob"] = round(prob, 3)
                preds["away_win_prob"] = round(1 - prob, 3)
            elif model_name == "totals_model":
                preds["predicted_total"] = round(float(model.predict(X_imp)[0]), 1)
        except Exception as e:
            log.warning("Prediction failed for %s: %s", model_name, e)

    return preds


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 5: COMPUTE EDGE vs VEGAS
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def compute_edge(preds: dict, odds_row: dict | None) -> dict:
    """
    Compare model predictions to Vegas lines.
    Edge = model prediction minus Vegas line (from home team's perspective).
    Positive edge = model thinks home team does better than Vegas says.
    """
    edge = {}
    if not odds_row:
        return edge

    # Spread edge: model_margin - (-vegas_spread) from home team perspective
    if "predicted_margin" in preds and odds_row.get("spread") is not None:
        vegas_spread = odds_row["spread"]  # negative = home team favored
        edge["spread_edge"] = round(preds["predicted_margin"] - (-vegas_spread), 1)
        edge["vegas_spread"] = vegas_spread
        edge["model_spread"] = preds["predicted_margin"]

    # Totals edge: model_total - vegas_total
    if "predicted_total" in preds and odds_row.get("total") is not None:
        edge["total_edge"]   = round(preds["predicted_total"] - odds_row["total"], 1)
        edge["vegas_total"]  = odds_row["total"]
        edge["model_total"]  = preds["predicted_total"]
        edge["ou_lean"]      = "OVER" if edge["total_edge"] > 0 else "UNDER"

    # ML implied prob vs model prob
    if "home_win_prob" in preds and odds_row.get("home_ml") is not None:
        vegas_impl = _ml_to_implied_prob(odds_row["home_ml"])
        edge["model_win_prob"] = preds["home_win_prob"]
        edge["vegas_implied_prob"] = round(vegas_impl, 3)
        edge["win_prob_edge"] = round(preds["home_win_prob"] - vegas_impl, 3)

    return edge


def _ml_to_implied_prob(ml: float) -> float:
    """Convert American moneyline to implied probability (no-vig)."""
    if ml > 0:
        return 100 / (ml + 100)
    else:
        return abs(ml) / (abs(ml) + 100)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 6: BET RECOMMENDATION
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def recommend_bets(edge: dict, preds: dict, game_tier: str = "unknown") -> list[dict]:
    """
    Flag games where the model has meaningful edge vs the market.

    Confidence tiers (spread):
      HIGH   : edge 9-15 pts, high-major conference only
      MEDIUM : edge 7-9 pts, high or mid-major only
      LOW    : edge 7-15 pts, any conference (shown but not recommended)

    Edges > 15 pts (spread) or > 12 pts (totals) are suppressed as likely bad odds data.
    NaN edges are always suppressed.
    """
    import math
    bets = []

    # -- Spread ---------------------------------------------------------------
    raw_spread = edge.get("spread_edge", 0) or 0
    spread_abs = abs(raw_spread)
    if math.isnan(spread_abs):
        spread_abs = 0

    if 7.0 <= spread_abs <= 15.0:
        side = "HOME" if raw_spread > 0 else "AWAY"
        raw_conf = "HIGH" if spread_abs >= 9.0 else "MEDIUM"

        if game_tier == "low":
            conf = "LOW"
        elif game_tier in ("mid", "unknown") and raw_conf == "HIGH":
            conf = "MEDIUM"
        else:
            conf = raw_conf

        bets.append({
            "market":     "SPREAD",
            "lean":       side,
            "edge_pts":   spread_abs,
            "confidence": conf,
        })

    # -- Totals ---------------------------------------------------------------
    raw_total = edge.get("total_edge", 0) or 0
    total_abs = abs(raw_total)
    if math.isnan(total_abs):
        total_abs = 0

    if 7.0 <= total_abs <= 12.0:
        lean = edge.get("ou_lean", "")
        raw_conf = "HIGH" if total_abs >= 9.0 else "MEDIUM"

        if game_tier == "low":
            conf = "LOW"
        elif game_tier in ("mid", "unknown") and raw_conf == "HIGH":
            conf = "MEDIUM"
        else:
            conf = raw_conf

        bets.append({
            "market":     "TOTAL",
            "lean":       lean,
            "edge_pts":   total_abs,
            "confidence": conf,
        })

    # Moneyline: disabled -- win prob model not yet calibrated for ML betting.
    # Re-enable once we have a dedicated ML calibration layer.
    # if abs(edge.get("win_prob_edge", 0)) >= 0.05: ...

    return bets
def save_predictions_to_db(predictions: list[dict], date_str: str):
    """Store predictions in SQLite for future backtesting."""
    if not predictions:
        return

    con = sqlite3.connect(DB_PATH)

    # Create table if needed
    con.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            game_id          TEXT,
            date             TEXT,
            home_team        TEXT,
            away_team        TEXT,
            predicted_margin REAL,
            home_win_prob    REAL,
            predicted_total  REAL,
            vegas_spread     REAL,
            vegas_total      REAL,
            spread_edge      REAL,
            total_edge       REAL,
            win_prob_edge    REAL,
            actual_margin    REAL,    -- filled in after game completes
            actual_total     REAL,    -- filled in after game completes
            is_neutral       INTEGER DEFAULT 0,
            created_at       TEXT,
            game_tier        TEXT,
            PRIMARY KEY (game_id, date)
        )
    """)

    now = datetime.now().isoformat()
    rows = []
    for p in predictions:
        edge = p.get("edge", {})
        preds = p.get("predictions", {})
        rows.append((
            p["game_id"], date_str,
            p["home_team"], p["away_team"],
            preds.get("predicted_margin"),
            preds.get("home_win_prob"),
            preds.get("predicted_total"),
            edge.get("vegas_spread"),
            edge.get("vegas_total"),
            edge.get("spread_edge"),
            edge.get("total_edge"),
            edge.get("win_prob_edge"),
            None, None,  # actual results filled in later
            int(p.get("neutral", False)),
            now,
            p.get("game_tier", "unknown"),
        ))

    con.executemany("""
        INSERT OR REPLACE INTO predictions
        (game_id, date, home_team, away_team, predicted_margin, home_win_prob,
         predicted_total, vegas_spread, vegas_total, spread_edge, total_edge,
         win_prob_edge, actual_margin, actual_total, is_neutral, created_at, game_tier)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, rows)
    con.commit()
    con.close()
    log.info("Saved %d predictions to DB", len(rows))


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# MAIN PIPELINE
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def run_pipeline(date_str: str, pretty: bool = False) -> list[dict]:
    log.info("=" * 60)
    log.info("  NCAAB PREDICTION PIPELINE â€” %s", date_str)
    log.info("=" * 60)

    # â”€â”€ Load models â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    models = load_models()
    if not models:
        log.error("No models found in %s. Run model_trainer first.", MODELS_DIR)
        return []

    # â”€â”€ Load feature matrix â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    try:
        feature_matrix = load_feature_matrix()
    except Exception as e:
        log.error("Failed to load feature matrix: %s", e)
        return []

    # â”€â”€ Fetch today's games â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    games = fetch_todays_schedule(date_str)
    if not games:
        log.warning("No games found for %s", date_str)
        return []

    # Separate upcoming from completed
    upcoming = [g for g in games if not g["completed"]]
    completed = [g for g in games if g["completed"]]
    log.info("%d upcoming games, %d completed", len(upcoming), len(completed))

    # Use all games for prediction (including completed for backtest purposes)
    games_to_predict = games

    # â”€â”€ Fetch odds â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    odds_df = fetch_current_odds()

    def find_odds(home_norm: str, away_norm: str) -> dict | None:
        if odds_df.empty:
            return None
        match = odds_df[
            (odds_df["home_team"] == home_norm) |
            (odds_df["away_team"] == home_norm)
        ]
        if len(match):
            return match.iloc[0].to_dict()
        return None

    # â”€â”€ Predict each game â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    predictions = []
    log.info("-" * 60)

    for game in games_to_predict:
        home = game["home_team"]
        away = game["away_team"]
        home_norm = game["home_team_norm"]
        away_norm = game["away_team_norm"]
        log.info("Processing: %s @ %s", away, home)

        # Get odds
        odds_row = find_odds(home_norm, away_norm)

        # Build features
        features = build_game_features(
            home_team=home,
            away_team=away,
            is_neutral=game.get("neutral", 0),
            target_date=date_str,
            feature_matrix=feature_matrix,
            odds_row=odds_row,
        )

        if features is None:
            log.warning("  Skipping %s @ %s â€” could not build features", away, home)
            continue

        # Run models
        preds = predict_game(features, models)

        # Compute edge vs Vegas
        edge = compute_edge(preds, odds_row)

        # Compute game tier BEFORE bet recommendations so tier filtering works
        game_tier = get_game_tier(home, away)

        # Bet recommendations
        bets = recommend_bets(edge, preds, game_tier=game_tier)
        # Stamp game_tier onto each individual bet for backtesting
        for b in bets:
            b["game_tier"] = game_tier

        # Assemble output
        result = {
            "game_id":    game["game_id"],
            "date":       date_str,
            "home_team":  home,
            "away_team":  away,
            "tipoff":     game.get("tipoff_time", ""),
            "venue":      game.get("venue", ""),
            "neutral":    bool(game.get("neutral", 0)),
            "game_tier":  game_tier,
            "status":     game.get("status", ""),
            "predictions": preds,
            "edge":        edge,
            "bets":        bets,
        }

        # Add actual scores if completed
        if game["completed"]:
            result["actual"] = {
                "home_score":  game["home_score"],
                "away_score":  game["away_score"],
                "margin":      game["home_score"] - game["away_score"],
                "total":       game["home_score"] + game["away_score"],
            }

        predictions.append(result)

        # Pretty print to console
        margin = preds.get("predicted_margin", 0)
        wp     = preds.get("home_win_prob", 0.5)
        total  = preds.get("predicted_total", 0)
        favored = home if margin >= 0 else away
        fav_margin = abs(margin)
        vegas_spread = edge.get("vegas_spread", None) if edge else None
        vegas_str = f" (Vegas: {vegas_spread:+.1f})" if vegas_spread is not None else ""

        log.info("  %s favored by %.1f%s | Total: %.1f",
                 favored, fav_margin, vegas_str, total)

        if edge:
            s_edge = edge.get("spread_edge", 0)
            t_edge = edge.get("total_edge", 0)
            if s_edge:
                log.info("  Spread edge: %+.1f pts (vs Vegas %+.1f)", s_edge, edge.get("vegas_spread", 0))
            if t_edge:
                log.info("  Total edge:  %+.1f pts (%s)", t_edge, edge.get("ou_lean", ""))

        if bets:
            for bet in bets:
                log.info("  â˜… BET: %s %s â€” %.1f pt edge [%s]",
                         bet["market"], bet["lean"],
                         bet.get("edge_pts", bet.get("edge_pct", 0)),
                         bet["confidence"])

    log.info("-" * 60)
    log.info("Predictions complete: %d / %d games", len(predictions), len(games_to_predict))

    # â”€â”€ Save to DB â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    save_predictions_to_db(predictions, date_str)

    # â”€â”€ Save to JSON â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    json_path = PRED_DIR / f"{date_str}_predictions.json"
    with open(json_path, "w") as f:
        json.dump(predictions, f, indent=2, default=str)
    log.info("Saved: %s", json_path)

    # â”€â”€ Save to CSV â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    csv_rows = []
    for p in predictions:
        preds = p.get("predictions", {})
        edge  = p.get("edge", {})
        bets  = p.get("bets", [])
        bet_markets = ", ".join(f"{b['market']} {b['lean']}" for b in bets) if bets else ""

        csv_rows.append({
            "date":             p["date"],
            "away_team":        p["away_team"],
            "home_team":        p["home_team"],
            "predicted_margin": preds.get("predicted_margin", ""),
            "home_win_prob":    preds.get("home_win_prob", ""),
            "predicted_total":  preds.get("predicted_total", ""),
            "vegas_spread":     edge.get("vegas_spread", ""),
            "spread_edge":      edge.get("spread_edge", ""),
            "vegas_total":      edge.get("vegas_total", ""),
            "total_edge":       edge.get("total_edge", ""),
            "ou_lean":          edge.get("ou_lean", ""),
            "bets":             bet_markets,
        })

    csv_path = PRED_DIR / f"{date_str}_predictions.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    log.info("Saved: %s", csv_path)

    # â”€â”€ Print summary table â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    _print_summary(predictions, pretty)

    return predictions


def _print_summary(predictions: list[dict], pretty: bool = False):
    """Print a clean summary table to console."""
    if not predictions:
        return

    print("\n" + "=" * 80)
    print(f"  PREDICTIONS SUMMARY â€” {len(predictions)} games")
    print("=" * 80)
    print(f"  {'MATCHUP':<35} {'SPREAD':>8} {'TOTAL':>7} {'HOME WP':>8} {'EDGE':>8}  BETS")
    print("-" * 80)

    for p in predictions:
        preds = p.get("predictions", {})
        edge  = p.get("edge", {})
        bets  = p.get("bets", [])

        matchup = f"{p['away_team'][:15]} @ {p['home_team'][:15]}"
        margin  = preds.get("predicted_margin", 0)
        spread_str = f"{margin:+.1f}" if margin is not None else "N/A"
        total_str  = f"{preds.get('predicted_total', 0):.1f}" if preds.get("predicted_total") else "N/A"
        wp_str     = f"{100*preds.get('home_win_prob', 0.5):.0f}%" if preds.get("home_win_prob") else "N/A"

        s_edge = edge.get("spread_edge")
        edge_str = f"{s_edge:+.1f}" if s_edge is not None else ""

        bet_str = "â˜…" * len(bets) if bets else ""

        print(f"  {matchup:<35} {spread_str:>8} {total_str:>7} {wp_str:>8} {edge_str:>8}  {bet_str}")

    # Flagged bets
    flagged = [(p, b) for p in predictions for b in p.get("bets", [])]
    if flagged:
        print("\n  FLAGGED PLAYS:")
        print("-" * 80)
        for p, b in flagged:
            matchup = f"{p['away_team']} @ {p['home_team']}"
            edge_val = b.get("edge_pts", b.get("edge_pct", 0))
            unit = "pts" if "edge_pts" in b else "%"
            print(f"  [{b['confidence']}] {b['market']} {b['lean']:5} â€” {matchup}")
            print(f"         Edge: {edge_val:.1f} {unit}")

    print("=" * 80 + "\n")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# MAIN
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def main():
    parser = argparse.ArgumentParser(description="NCAAB Daily Prediction Pipeline")
    parser.add_argument("--date", default=None,
                        help="Date to predict (YYYY-MM-DD). Default: today")
    parser.add_argument("--pretty", action="store_true",
                        help="Print detailed per-game output")
    parser.add_argument("--tomorrow", action="store_true",
                        help="Predict tomorrow's slate instead of today")
    args = parser.parse_args()

    if args.date:
        date_str = args.date
    elif args.tomorrow:
        date_str = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        date_str = datetime.now().strftime("%Y-%m-%d")

    run_pipeline(date_str, pretty=args.pretty)


if __name__ == "__main__":
    main()


