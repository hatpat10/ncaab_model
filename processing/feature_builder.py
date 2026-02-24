"""
processing/feature_builder.py
NCAAB ML Prediction System — Phase 3 (Cleaning) + Phase 4 (Feature Engineering)

Builds the full feature matrix from all raw data sources:
  - hoopR team box scores (parquet)
  - BartTorvik advanced ratings (parquet)
  - ESPN games + odds (SQLite: games_raw, odds_raw)

Output: data/processed/feature_matrix.parquet
        data/processed/feature_matrix.csv (optional)

Run:
    python -m processing.feature_builder
"""

import json
import logging
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent  # project root
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROC = BASE_DIR / "data" / "processed"
DB_PATH = BASE_DIR / "data" / "ncaab.db"
ALIASES_PATH = BASE_DIR / "data" / "team_aliases.json"

DATA_PROC.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Rolling windows to compute
ROLL_WINDOWS = [5, 10]

# BartTorvik columns we care about
# Actual columns present in barttorvik parquets (verified from diagnose_sources.py)
# BartTorvik year=2020 means the 2019-20 season; year=2024 means 2023-24 season.
# ESPN season=2026 means the 2025-26 season. Mapping: torvik_year = espn_season - 1
TORVIK_COLS = [
    "team", "conf", "barthag", "barthag_rk",
    "adj_o", "adj_o_rk", "adj_d", "adj_d_rk",
    "adj_t", "adj_t_rk",
    "wab",
    "nc_elite_sos", "ov_elite_sos",
    "nc_cur_sos", "ov_cur_sos",
    "seed", "year",
]

# ──────────────────────────────────────────────────────────────────────────────
# TEAM NAME NORMALIZATION
# ──────────────────────────────────────────────────────────────────────────────
def _load_aliases() -> dict:
    if ALIASES_PATH.exists():
        with open(ALIASES_PATH) as f:
            return json.load(f)
    log.warning("team_aliases.json not found at %s — using identity mapping", ALIASES_PATH)
    return {}


_ALIASES: dict = {}


def normalize_team(raw_name: str) -> str:
    """Convert any known alias to a canonical team_id slug.

    Handles ESPN full display names which include mascots:
      'Pacific Tigers' -> 'pacific'
      'Murray State Racers' -> 'murray_state'
    Strategy:
      1. Exact match in aliases
      2. Case-insensitive match
      3. Strip last word(s) and retry (removes mascots)
      4. Slugify fallback
    """
    global _ALIASES
    if not _ALIASES:
        _ALIASES = _load_aliases()
    if not isinstance(raw_name, str):
        return str(raw_name)
    name = raw_name.strip()

    # 1. Exact match
    if name in _ALIASES:
        return _ALIASES[name]

    # 2. Case-insensitive match
    lower = name.lower()
    for key, val in _ALIASES.items():
        if key.lower() == lower:
            return val

    # 3. Strip trailing words (mascots) one at a time and retry
    words = name.split()
    for n_drop in range(1, min(3, len(words))):
        shorter = " ".join(words[:-n_drop])
        if shorter in _ALIASES:
            return _ALIASES[shorter]
        for key, val in _ALIASES.items():
            if key.lower() == shorter.lower():
                return val

    # 3b. Check if the raw name is already a slug that exists as an alias key
    slug_input = name.lower().replace(" ", "_").replace("-", "_").replace("&", "").replace(".", "")
    if slug_input in _ALIASES:
        return _ALIASES[slug_input]

    # 4. Slugify fallback (strip common mascot words first)
    slug = name.lower()
    for junk in [" tigers", " eagles", " bulldogs", " wildcats", " bears",
                 " hawks", " panthers", " lions", " wolves", " warriors",
                 " knights", " cardinals", " rams", " bobcats", " owls",
                 " ravens", " falcons", " ducks", " beavers", " gators",
                 " seminoles", " hurricanes", " demon deacons", " tar heels",
                 " blue devils", " golden eagles", " golden bears", " red hawks",
                 " fighting hawks", " running rebels", " aggies", " cowboys",
                 " longhorns", " horned frogs", " jayhawks", " cyclones",
                 " cornhuskers", " huskers", " sooners", " boilermakers",
                 " hoosiers", " badgers", " gophers", " wolverines", " spartans",
                 " buckeyes", " hawkeyes", " illini", " terrapins", " terps",
                 " hokies", " cavaliers", " orange", " golden hurricane",
                 " racers", " skyhawks", " ospreys", " mavericks", " mastodons",
                 " colonials", " toreros", " dons", " pioneers", " jaspers",
                 " highlanders", " jaguars", " antelopes", " lopes",
                 " delta devils", " tigers", " grizzlies", " mountaineers"]:
        slug = slug.replace(junk, "")
    slug = slug.strip().replace(" ", "_").replace("-", "_").replace("&", "")
    # Clean up double underscores
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 3 — DATA LOADING & CLEANING
# ──────────────────────────────────────────────────────────────────────────────

def load_hoopr_box() -> pd.DataFrame:
    """Load hoopR team box scores, normalize team names, deduplicate."""
    path = DATA_RAW / "hoopR_team_box_2020_2025.parquet"
    log.info("Loading hoopR box scores: %s", path)
    df = pd.read_parquet(path)

    # Identify team name column (varies by hoopR version)
    team_col = next((c for c in ["team_location", "team_name", "team_display_name"] if c in df.columns), None)
    if team_col is None:
        raise ValueError(f"Cannot find team name column in hoopR box. Columns: {df.columns.tolist()}")

    df["team_id"] = df[team_col].apply(normalize_team)

    # Parse date
    date_col = next((c for c in ["game_date", "date"] if c in df.columns), None)
    if date_col:
        df["date"] = pd.to_datetime(df[date_col], errors="coerce")

    # Season column
    if "season" not in df.columns and "date" in df.columns:
        df["season"] = df["date"].dt.year.where(df["date"].dt.month >= 10, df["date"].dt.year - 1) + 1

    # Deduplicate: keep first occurrence per (game_id, team_id)
    id_col = "game_id" if "game_id" in df.columns else None
    if id_col:
        before = len(df)
        df = df.drop_duplicates(subset=[id_col, "team_id"], keep="first")
        log.info("hoopR dedup: %d → %d rows", before, len(df))

    if "game_id" in df.columns:
        df["game_id"] = df["game_id"].astype(str)
    # Flag scoring outliers (don't drop — just tag)
    if "team_score" in df.columns:
        df["score_outlier"] = df["team_score"].between(30, 130).eq(False).astype(int)
        n_out = df["score_outlier"].sum()
        if n_out:
            log.warning("hoopR: %d rows flagged as scoring outliers (< 30 or > 130)", n_out)

    log.info("hoopR box loaded: %d rows, %d cols", len(df), df.shape[1])
    return df


def load_barttorvik() -> pd.DataFrame:
    """Load + concat both BartTorvik parquets, normalize team names."""
    frames = []
    for fname in ["barttorvik_2020_2023.parquet", "barttorvik_2024_2025.parquet"]:
        p = DATA_RAW / fname
        if p.exists():
            frames.append(pd.read_parquet(p))
        else:
            log.warning("BartTorvik file not found: %s", p)

    if not frames:
        raise FileNotFoundError("No BartTorvik parquet files found in data/raw/")

    df = pd.concat(frames, ignore_index=True)

    # Normalize team column
    team_col = next((c for c in ["team", "team_name", "Team"] if c in df.columns), None)
    if team_col:
        df["team_id"] = df[team_col].apply(normalize_team)
    else:
        raise ValueError(f"No team column found in BartTorvik. Columns: {df.columns.tolist()}")

    # Lower-case column names for consistency
    df.columns = [c.lower().strip() for c in df.columns]

    # Drop any duplicate columns produced by the rename + lowercase steps
    df = df.loc[:, ~df.columns.duplicated()]

    # BartTorvik uses 'year' column. year=2024 means the 2023-24 season.
    # ESPN uses season=2025 for the same season (the year the season ENDS).
    # So: torvik_year + 1 = espn_season
    if "year" in df.columns:
        df["season"] = (pd.to_numeric(df["year"], errors="coerce") + 1).astype("Int64")
    elif "season" not in df.columns:
        raise ValueError("BartTorvik parquet has neither 'year' nor 'season' column")

    # Keep only columns that exist (use actual parquet cols, not assumed cols)
    available = set(df.columns)
    keep = list(dict.fromkeys(
        [c for c in TORVIK_COLS if c in available] + ["team_id", "season"]
    ))
    df = df[[c for c in keep if c in df.columns]].drop_duplicates(subset=["team_id", "season"])

    # Ensure season is a plain 1-D Series
    df["season"] = pd.to_numeric(df["season"], errors="coerce")

    # Rename _rk columns to _rank for consistency with rest of codebase
    df = df.rename(columns={
        "adj_o_rk": "adj_o_rank", "adj_d_rk": "adj_d_rank",
        "adj_t_rk": "adj_t_rank", "barthag_rk": "barthag_rank",
    })

    # Compute ranks from scratch if rk cols were missing
    if "adj_d_rank" not in df.columns and "adj_d" in df.columns:
        df["adj_d_rank"] = df.groupby("season")["adj_d"].rank(ascending=True, na_option="keep")
    if "adj_o_rank" not in df.columns and "adj_o" in df.columns:
        df["adj_o_rank"] = df.groupby("season")["adj_o"].rank(ascending=False, na_option="keep")

    log.info("BartTorvik loaded: %d rows, %d cols", len(df), df.shape[1])
    return df


def load_espn_games() -> pd.DataFrame:
    """Load ESPN games from SQLite games_raw, normalize team names."""
    log.info("Loading ESPN games from SQLite: %s", DB_PATH)
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM games_raw", con)
    con.close()

    for col in ["home_team", "away_team", "home_team_name", "away_team_name"]:
        if col in df.columns:
            df[col] = df[col].apply(normalize_team)

    df["date"] = pd.to_datetime(df.get("date", df.get("game_date")), errors="coerce")
    # games_raw already has a season column (e.g. 2026 for the 2025-26 season).
    # Only compute if missing.
    if "season" not in df.columns or df["season"].isna().all():
        df["season"] = df["date"].dt.year.where(df["date"].dt.month >= 10, df["date"].dt.year - 1) + 1
    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    if "game_id" in df.columns:
        df["game_id"] = df["game_id"].astype(str)

    # Neutral site fix: some sources mark neutral as home for higher seed
    if "neutral_site" not in df.columns and "neutral" in df.columns:
        df["neutral_site"] = df["neutral"]

    log.info("ESPN games loaded: %d rows", len(df))
    return df


def load_odds() -> pd.DataFrame:
    """
    Load odds_raw from SQLite and pivot from long to wide format.

    Raw schema (long): one row per (game_id, bookmaker, market, outcome)
      markets: h2h (moneyline), spreads, totals
      outcomes: home team name, away team name, Over, Under

    Output (wide): one row per game with columns:
      vegas_spread, vegas_total, vegas_ml_home, vegas_ml_away
    """
    log.info("Loading odds from SQLite")
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM odds_raw", con)
    con.close()

    if df.empty:
        log.warning("odds_raw is empty")
        return pd.DataFrame(columns=["game_id", "vegas_spread", "vegas_total",
                                     "vegas_ml_home", "vegas_ml_away"])

    df["game_id"] = df["game_id"].astype(str)

    # Bookmaker priority: use best available line per game
    priority = {"pinnacle": 0, "fanduel": 1, "draftkings": 2, "betmgm": 3,
                "caesars": 4, "pointsbet": 5, "betonlineag": 6}
    df["book_priority"] = df["bookmaker"].str.lower().map(priority).fillna(99)
    df = df.sort_values(["game_id", "market", "book_priority"])

    # Normalize team names in odds for matching
    if "home_team" in df.columns:
        df["home_team_norm"] = df["home_team"].apply(normalize_team)
    if "away_team" in df.columns:
        df["away_team_norm"] = df["away_team"].apply(normalize_team)
    if "outcome" in df.columns:
        df["outcome_norm"] = df["outcome"].apply(normalize_team)

    results = []
    for game_id, gdf in df.groupby("game_id"):
        row = {"game_id": game_id}

        # Best bookmaker for this game (lowest priority number)
        best_book = gdf["book_priority"].min()
        best = gdf[gdf["book_priority"] == best_book]

        home_norm = best["home_team_norm"].iloc[0] if "home_team_norm" in best.columns else None
        away_norm = best["away_team_norm"].iloc[0] if "away_team_norm" in best.columns else None

        # Preserve date and team names for join key in merge_odds_features
        row["date"]      = best["date"].iloc[0] if "date" in best.columns else None
        row["home_team"] = best["home_team"].iloc[0] if "home_team" in best.columns else None
        row["away_team"] = best["away_team"].iloc[0] if "away_team" in best.columns else None

        # Moneyline (h2h)
        h2h = best[best["market"] == "h2h"]
        if not h2h.empty and "outcome_norm" in h2h.columns:
            home_ml = h2h[h2h["outcome_norm"] == home_norm]["price"]
            away_ml = h2h[h2h["outcome_norm"] == away_norm]["price"]
            if not home_ml.empty:
                row["vegas_ml_home"] = float(home_ml.iloc[0])
            if not away_ml.empty:
                row["vegas_ml_away"] = float(away_ml.iloc[0])

        # Spread (home team spread — negative means home favored)
        spreads = best[best["market"] == "spreads"]
        if not spreads.empty and "outcome_norm" in spreads.columns and "point" in spreads.columns:
            home_spread = spreads[spreads["outcome_norm"] == home_norm]["point"]
            if not home_spread.empty:
                row["vegas_spread"] = float(home_spread.iloc[0])

        # Total (Over line)
        totals = best[best["market"] == "totals"]
        if not totals.empty and "point" in totals.columns:
            over_row = totals[totals["outcome"].str.lower() == "over"] if "outcome" in totals.columns else totals
            if not over_row.empty:
                row["vegas_total"] = float(over_row["point"].iloc[0])

        results.append(row)

    wide = pd.DataFrame(results)
    log.info("Odds loaded: %d games with lines (from %d raw rows)", len(wide), len(df))
    return wide


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 3 — MASTER GAME TABLE (team-per-row format)
# ──────────────────────────────────────────────────────────────────────────────

def build_master_games(espn_games: pd.DataFrame, hoopr_box: pd.DataFrame) -> pd.DataFrame:
    """
    Explode ESPN games into team-per-row format and merge hoopR box stats.
    Each row = one team's perspective for one game.
    """
    games = espn_games.copy()

    # Identify home/away columns
    h_team = next((c for c in ["home_team", "home_team_name"] if c in games.columns), None)
    a_team = next((c for c in ["away_team", "away_team_name"] if c in games.columns), None)
    h_score = next((c for c in ["home_score", "home_points"] if c in games.columns), None)
    a_score = next((c for c in ["away_score", "away_points"] if c in games.columns), None)

    if not all([h_team, a_team]):
        raise ValueError("ESPN games missing home/away team columns")

    neutral_col = "neutral_site" if "neutral_site" in games.columns else None

    # Build home perspective
    home_cols = {"game_id": "game_id", "date": "date", "season": "season",
                 h_team: "team_id", a_team: "opp_id"}
    if h_score:
        home_cols[h_score] = "team_score"
    if a_score:
        home_cols[a_score] = "opp_score"
    home = games.rename(columns=home_cols)[[c for c in home_cols.values() if c in games.rename(columns=home_cols).columns]].copy()
    home["is_home"] = 1

    # Build away perspective
    away_cols = {"game_id": "game_id", "date": "date", "season": "season",
                 a_team: "team_id", h_team: "opp_id"}
    if a_score:
        away_cols[a_score] = "team_score"
    if h_score:
        away_cols[h_score] = "opp_score"
    away = games.rename(columns=away_cols)[[c for c in away_cols.values() if c in games.rename(columns=away_cols).columns]].copy()
    away["is_home"] = 0

    if neutral_col:
        home["is_home"] = home["is_home"].where(~games[neutral_col].astype(bool).values, 0)
        away["is_home"] = 0

    master = pd.concat([home, away], ignore_index=True)
    master = master.sort_values(["team_id", "date"]).reset_index(drop=True)

    # Drop non-D1 / NAIA opponents that slipped in via ESPN exhibition games
    NON_D1_KEYWORDS = [
        "biblical", "bethesda", "ecclesia", "champion_christian", "mid_atlantic_christian",
        "lincoln_university", "noble", "nobel", "naia", "juco", "community_college",
        "bible", "seminary", "christian_college", "east_west_university", "shenango",
        "toccoa", "fisk", "cleary", "oaklanders", "penn_state_shenango",
        "virginia_lynchburg", "fort_lauderdale", "bryan__tn_", "southwestern_christian",
        "southwestern_adventist", "trinity_college_of_jacksonville", "central_penn",
        "dallas_christian",
    ]
    def is_non_d1(team_id):
        t = str(team_id).lower()
        return any(kw in t for kw in NON_D1_KEYWORDS)

    before = len(master)
    non_d1_mask = master["team_id"].apply(is_non_d1) | master["opp_id"].apply(is_non_d1)
    master = master[~non_d1_mask].reset_index(drop=True)
    dropped = before - len(master)
    if dropped:
        log.info("Dropped %d rows involving non-D1/NAIA teams", dropped)

    # Compute margin if scores available
    if "team_score" in master.columns and "opp_score" in master.columns:
        master["margin"] = master["team_score"] - master["opp_score"]
        master["win"] = (master["margin"] > 0).astype(int)

    # Merge hoopR box stats
    if hoopr_box is not None and len(hoopr_box) > 0:
        box_merge_cols = ["game_id", "team_id"] + [c for c in hoopr_box.columns
                          if c not in ["game_id", "team_id", "date", "season"]]
        # Only keep numeric columns + identifiers
        box_num = hoopr_box.select_dtypes(include=[np.number]).columns.tolist()
        box_keep = ["game_id", "team_id"] + [c for c in box_num if c not in ["game_id"]]
        box_keep = [c for c in box_keep if c in hoopr_box.columns]
        master = master.merge(hoopr_box[box_keep], on=["game_id", "team_id"], how="left", suffixes=("", "_box"))

    log.info("Master games table: %d rows (%.0f games)", len(master),
             master["game_id"].nunique() if "game_id" in master.columns else 0)
    return master


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 4 — FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────────────────────

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling window features computed BEFORE each game (shift(1) prevents leakage).
    Covers: scoring, margin, shooting efficiency, pace, four factors.
    """
    df = df.copy().sort_values(["team_id", "date"]).reset_index(drop=True)

    # Core stats to roll — use what's available
    candidate_roll_stats = {
        "team_score": "pts",
        "opp_score": "opp_pts",
        "margin": "margin",
        "field_goals_made": "fgm",
        "field_goals_attempted": "fga",
        "three_point_field_goals_made": "fg3m",
        "three_point_field_goals_attempted": "fg3a",
        "free_throws_made": "ftm",
        "free_throws_attempted": "fta",
        "offensive_rebounds": "oreb",
        "defensive_rebounds": "dreb",
        "assists": "ast",
        "turnovers": "tov",
        "steals": "stl",
        "blocks": "blk",
        "fouls": "pf",
        # hoopR may use these alternate names
        "points": "pts",
        "field_goal_pct": "fg_pct",
        "three_point_pct": "fg3_pct",
        "free_throw_pct": "ft_pct",
        "total_rebounds": "treb",
        "offensive_rebounds_allowed": "oreb_allowed",
    }

    roll_stats = {src: dst for src, dst in candidate_roll_stats.items() if src in df.columns}

    for n in ROLL_WINDOWS:
        for src_col, short in roll_stats.items():
            col_name = f"roll{n}_{short}"
            df[col_name] = (
                df.groupby("team_id")[src_col]
                .transform(lambda x: x.shift(1).rolling(n, min_periods=max(1, n // 2)).mean())
            )

    # Derived rolling efficiency (eFG%, TS%)
    for n in ROLL_WINDOWS:
        if all(f"roll{n}_{s}" in df.columns for s in ["fgm", "fg3m", "fga"]):
            df[f"roll{n}_efg"] = (df[f"roll{n}_fgm"] + 0.5 * df[f"roll{n}_fg3m"]) / df[f"roll{n}_fga"].replace(0, np.nan)
        if all(f"roll{n}_{s}" in df.columns for s in ["pts", "fga", "fta"]):
            df[f"roll{n}_ts"] = df[f"roll{n}_pts"] / (2 * (df[f"roll{n}_fga"] + 0.44 * df[f"roll{n}_fta"]).replace(0, np.nan))
        if all(f"roll{n}_{s}" in df.columns for s in ["tov", "fga", "fta"]):
            df[f"roll{n}_tov_rate"] = df[f"roll{n}_tov"] / (df[f"roll{n}_fga"] + 0.44 * df[f"roll{n}_fta"] + df[f"roll{n}_tov"]).replace(0, np.nan)

    # Season-to-date trend (score vs. expanding mean)
    if "team_score" in df.columns:
        df["score_trend_5"] = (
            df.groupby("team_id")["team_score"]
            .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean()
                       - x.shift(1).expanding().mean())
        )

    # Win streak (consecutive wins heading into game)
    if "win" in df.columns:
        def win_streak(x):
            streaks = []
            s = 0
            for v in x.shift(1).fillna(0):
                s = s + 1 if v == 1 else 0
                streaks.append(s)
            return pd.Series(streaks, index=x.index)
        df["win_streak"] = df.groupby("team_id")["win"].transform(win_streak)

    log.info("Rolling features added.")
    return df


def merge_barttorvik(master: pd.DataFrame, torvik: pd.DataFrame) -> pd.DataFrame:
    """
    Merge BartTorvik advanced ratings onto master by (team_id, torvik_season).

    BartTorvik ratings for season N are the best proxy for team strength
    entering season N+1. For current-season games (ESPN season=2026), we use
    BartTorvik season=2025 (year=2025, the last fully completed season).
    This also avoids the problem of mid-season BartTorvik rows being partial/NaN.

    Join key: torvik_season = espn_season - 1
    """
    torvik_cols = [c for c in torvik.columns if c not in ["team", "year"]]

    # Drop rows where key metrics are all null (incomplete current-season rows)
    torvik_clean = torvik.dropna(subset=["adj_o", "adj_d"], how="all").copy()

    # Create a join key: shift torvik season forward by 1 so it lines up with ESPN
    # e.g. BartTorvik season=2025 (year=2025) → join_season=2026
    torvik_clean["join_season"] = torvik_clean["season"] + 1

    torvik_cols_clean = [c for c in torvik_cols if c != "season"] + ["join_season"]

    # Team metrics
    master = master.merge(
        torvik_clean[torvik_cols_clean].rename(
            columns={"join_season": "season", **{c: f"t_{c}" for c in torvik_cols_clean
                     if c not in ["team_id", "join_season", "season"]}}
        ),
        on=["team_id", "season"],
        how="left"
    )

    # Opponent metrics
    opp_torvik = torvik_clean[torvik_cols_clean].rename(
        columns={"team_id": "opp_id", "join_season": "season",
                 **{c: f"o_{c}" for c in torvik_cols_clean
                    if c not in ["team_id", "join_season", "season"]}}
    )
    master = master.merge(opp_torvik, on=["opp_id", "season"], how="left")

    log.info("BartTorvik merged. Columns now: %d", master.shape[1])
    return master


def add_matchup_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Matchup delta features: quantify THIS game's specific offensive vs. defensive edges.
    These are the single most predictive features in the model.
    """
    df = df.copy()

    # ── Efficiency deltas ──────────────────────────────────────────────────
    if "t_adj_o" in df.columns and "o_adj_d" in df.columns:
        df["off_eff_delta"] = df["t_adj_o"] - df["o_adj_d"]   # Our offense vs their D
    if "o_adj_o" in df.columns and "t_adj_d" in df.columns:
        df["def_eff_delta"] = df["o_adj_o"] - df["t_adj_d"]   # Their offense vs our D

    # Net rating delta (overall team quality gap)
    if all(c in df.columns for c in ["t_adj_o", "t_adj_d", "o_adj_o", "o_adj_d"]):
        df["net_rating_delta"] = (df["t_adj_o"] - df["t_adj_d"]) - (df["o_adj_o"] - df["o_adj_d"])
        df["barthag_delta"] = df.get("t_barthag", np.nan) - df.get("o_barthag", np.nan)

    # ── Pace / tempo ───────────────────────────────────────────────────────
    if "t_adj_t" in df.columns and "o_adj_t" in df.columns:
        df["tempo_delta"] = df["t_adj_t"] - df["o_adj_t"]
        df["avg_tempo"] = (df["t_adj_t"] + df["o_adj_t"]) / 2   # drives total pts

    # ── Four-factor / SOS matchups (using actual BartTorvik columns) ─────
    # BartTorvik parquet has: barthag, adj_o, adj_d, adj_t, wab,
    #   nc_elite_sos, ov_elite_sos, nc_cur_sos, ov_cur_sos
    # No efg_o/tov_o/orb — use what we have

    # Strength of schedule delta (overall SOS proxy)
    if "t_ov_cur_sos" in df.columns and "o_ov_cur_sos" in df.columns:
        df["sos_delta"] = df["t_ov_cur_sos"] - df["o_ov_cur_sos"]

    # WAB (wins above bubble) delta — measures resume quality
    if "t_wab" in df.columns and "o_wab" in df.columns:
        df["wab_delta"] = df["t_wab"] - df["o_wab"]

    # Elite SOS matchup (how many elite opponents each team has faced)
    if "t_ov_elite_sos" in df.columns and "o_ov_elite_sos" in df.columns:
        df["elite_sos_delta"] = df["t_ov_elite_sos"] - df["o_ov_elite_sos"]

    # ── Defense rank tier ─────────────────────────────────────────────────
    # Tier 1 = elite (rank 1-25), Tier 2 = good (26-75), etc.
    # Column is o_adj_d_rank (renamed from adj_d_rk in load_barttorvik)
    rank_col = next((c for c in ["o_adj_d_rank", "o_adj_d_rk"] if c in df.columns), None)
    if rank_col:
        df["opp_def_tier"] = pd.cut(
            pd.to_numeric(df[rank_col], errors="coerce"),
            bins=[0, 25, 75, 150, 400],
            labels=["elite", "good", "average", "weak"]
        ).astype(str)
        tier_map = {"elite": -10.0, "good": -5.0, "average": 0.0, "weak": 4.0, "nan": 0.0}
        df["def_suppression_factor"] = df["opp_def_tier"].map(tier_map).fillna(0.0)

    # ── Recent form delta ──────────────────────────────────────────────────
    # We need opp's rolling stats — join them in by flipping team/opp
    roll_margin_col = "roll5_margin"
    if roll_margin_col in df.columns:
        opp_roll = (df[["game_id", "team_id", roll_margin_col]]
                    .rename(columns={"team_id": "opp_id", roll_margin_col: "opp_roll5_margin"}))
        if "opp_id" in df.columns:
            df["game_id"] = df["game_id"].astype(str)
            opp_roll["game_id"] = opp_roll["game_id"].astype(str)
            df = df.merge(opp_roll, on=["game_id", "opp_id"], how="left", suffixes=("", "_dup"))
            if "opp_roll5_margin" in df.columns and "roll5_margin" in df.columns:
                df["form_delta"] = df["roll5_margin"] - df["opp_roll5_margin"]

    log.info("Matchup features added.")
    return df


def add_situational_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rest, travel, home court, and back-to-back features.
    """
    df = df.copy().sort_values(["team_id", "date"])

    # Days rest since last game
    df["days_rest"] = (
        df.groupby("team_id")["date"]
        .transform(lambda x: x.diff().dt.days)
    )
    df["days_rest"] = df["days_rest"].clip(0, 30).fillna(7)  # 7 = season opener default

    # Back-to-back flag (0 or 1 days rest)
    df["is_back_to_back"] = (df["days_rest"] <= 1).astype(int)

    # Short rest (2 days)
    df["is_short_rest"] = (df["days_rest"] == 2).astype(int)

    # Rest advantage vs. opponent
    if "opp_id" in df.columns and "game_id" in df.columns:
        rest_lookup = df[["game_id", "team_id", "days_rest"]].rename(
            columns={"team_id": "opp_id", "days_rest": "opp_days_rest"}
        )
        df = df.merge(rest_lookup, on=["game_id", "opp_id"], how="left", suffixes=("", "_dup"))
        if "opp_days_rest" in df.columns:
            df["rest_advantage"] = df["days_rest"] - df["opp_days_rest"]

    # Home court (already in is_home)
    # Neutral site home court value = 0 (already handled upstream)

    # Games played (fatigue proxy — relevant for late-season)
    df["games_played"] = df.groupby(["team_id", "season"]).cumcount()

    log.info("Situational features added.")
    return df


def merge_odds_features(df: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """
    Merge betting market signals onto master.

    Odds API uses a hash game_id; ESPN uses numeric game_id — they never match.
    Join instead on (date + normalized home team + normalized away team).
    """
    if odds is None or len(odds) == 0:
        log.warning("No odds data available — skipping odds merge.")
        return df

    val_cols = [c for c in ["vegas_spread", "vegas_total", "vegas_ml_home", "vegas_ml_away"]
                if c in odds.columns]
    if not val_cols:
        log.warning("Odds DataFrame has no usable value columns: %s", odds.columns.tolist())
        return df

    # ── Build join keys on odds side ──────────────────────────────────────
    o = odds.copy()
    if "date" not in o.columns:
        log.warning("Odds missing date column — cannot join")
        return df
    o["_date"] = pd.to_datetime(o["date"], errors="coerce").dt.normalize()
    o["_home"] = o["home_team"].apply(normalize_team) if "home_team" in o.columns else ""
    o["_away"] = o["away_team"].apply(normalize_team) if "away_team" in o.columns else ""
    o = o[["_date", "_home", "_away"] + val_cols].drop_duplicates(subset=["_date", "_home", "_away"])

    # ── Build join keys on master side ────────────────────────────────────
    if "is_home" not in df.columns or "opp_id" not in df.columns:
        log.warning("Cannot join odds: master missing is_home/opp_id")
        return df

    home_rows = df[df["is_home"] == 1][["game_id", "team_id", "opp_id", "date"]].copy()
    home_rows["_date"] = pd.to_datetime(home_rows["date"], errors="coerce").dt.normalize()
    home_rows["_home"] = home_rows["team_id"]
    home_rows["_away"] = home_rows["opp_id"]

    game_odds = home_rows[["game_id", "_date", "_home", "_away"]].merge(
        o, on=["_date", "_home", "_away"], how="left"
    )[["game_id"] + val_cols].drop_duplicates("game_id")

    df["game_id"] = df["game_id"].astype(str)
    game_odds["game_id"] = game_odds["game_id"].astype(str)
    df = df.merge(game_odds, on="game_id", how="left")

    matched = df["vegas_spread"].notna().sum() if "vegas_spread" in df.columns else 0
    log.info("Odds merged: %d team-rows with spread data (%.1f%% of rows)",
             matched, 100 * matched / len(df) if len(df) else 0)

    # Implied home win probability from moneyline
    def ml_to_prob(ml):
        ml = pd.to_numeric(ml, errors="coerce")
        return np.where(ml < 0, -ml / (-ml + 100), 100 / (ml + 100))

    if "vegas_ml_home" in df.columns and "vegas_ml_away" in df.columns:
        p_home = ml_to_prob(df["vegas_ml_home"])
        p_away = ml_to_prob(df["vegas_ml_away"])
        df["implied_prob_home"] = p_home / (p_home + p_away)

    log.info("Odds features merged.")
    return df


def add_h2h_features(df: pd.DataFrame, lookback: int = 3) -> pd.DataFrame:
    """
    Head-to-head history: avg margin and win rate in last N matchups.
    """
    df = df.copy().sort_values(["team_id", "opp_id", "date"])

    if "margin" not in df.columns or "opp_id" not in df.columns:
        log.warning("Skipping H2H features — need margin and opp_id columns.")
        return df

    def h2h_stats(group):
        margins = group["margin"].shift(1)
        wins = (group["margin"].shift(1) > 0).astype(float)
        group["h2h_avg_margin"] = margins.rolling(lookback, min_periods=1).mean()
        group["h2h_win_rate"] = wins.rolling(lookback, min_periods=1).mean()
        return group

    df = df.groupby(["team_id", "opp_id"], group_keys=False).apply(h2h_stats)
    log.info("H2H features added.")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# VALIDATION
# ──────────────────────────────────────────────────────────────────────────────

def validate_no_leakage(df: pd.DataFrame) -> None:
    """Spot-check that rolling features don't include current game stats."""
    if "roll5_pts" in df.columns and "team_score" in df.columns:
        # For any team's first game of the season, roll5_pts should be NaN
        first_games = df.groupby(["team_id", "season"]).head(1)
        leakers = first_games["roll5_pts"].notna().sum()
        if leakers > 0:
            log.warning("LEAKAGE WARNING: %d first-game rows have non-null roll5_pts!", leakers)
        else:
            log.info("Leakage check passed: no roll stats on first game of season.")


def validate_coverage(df: pd.DataFrame) -> None:
    """Log feature coverage and missing rate summary."""
    total = len(df)
    feature_cols = [c for c in df.columns if any(c.startswith(p) for p in
                    ["roll", "t_", "o_", "off_", "def_", "net_", "form_", "tempo", "efg", "h2h",
                     "days_rest", "is_home", "is_back", "vegas_", "implied_"])]
    if not feature_cols:
        log.warning("No feature columns found for coverage check.")
        return

    miss_pct = df[feature_cols].isnull().mean().sort_values(ascending=False)
    high_miss = miss_pct[miss_pct > 0.3]
    if len(high_miss):
        log.warning("High missing rate (>30%%) features:\n%s", high_miss.to_string())
    else:
        log.info("Feature coverage OK — no features >30%% missing.")
    log.info("Feature matrix: %d rows × %d feature columns", total, len(feature_cols))


# ──────────────────────────────────────────────────────────────────────────────
# ORCHESTRATOR
# ──────────────────────────────────────────────────────────────────────────────

def build_feature_matrix(save_csv: bool = False) -> pd.DataFrame:
    """
    Full pipeline: load → clean → merge → engineer → validate → save.
    Returns the complete feature matrix as a DataFrame.
    """
    log.info("=" * 60)
    log.info("NCAAB Feature Builder — Starting Phase 3 + 4")
    log.info("=" * 60)

    # ── Load all sources ───────────────────────────────────────────────────
    hoopr = load_hoopr_box()
    torvik = load_barttorvik()
    espn = load_espn_games()
    odds = load_odds()

    # ── Phase 3: Build master game table ──────────────────────────────────
    master = build_master_games(espn, hoopr)
    master = merge_barttorvik(master, torvik)

    # ── Phase 4: Feature engineering ──────────────────────────────────────
    master = add_rolling_features(master)
    master = add_situational_features(master)
    master = add_matchup_features(master)
    master = add_h2h_features(master)
    master = merge_odds_features(master, odds)

    # ── Validate ───────────────────────────────────────────────────────────
    validate_no_leakage(master)
    validate_coverage(master)

    # ── Save ───────────────────────────────────────────────────────────────
    out_parquet = DATA_PROC / "feature_matrix.parquet"
    master.to_parquet(out_parquet, index=False)
    log.info("Saved: %s  (%d rows × %d cols)", out_parquet, *master.shape)

    if save_csv:
        out_csv = DATA_PROC / "feature_matrix.csv"
        master.to_csv(out_csv, index=False)
        log.info("Saved CSV: %s", out_csv)

    log.info("=" * 60)
    log.info("Feature matrix complete: %d rows, %d columns", *master.shape)
    log.info("Games covered: %d", master["game_id"].nunique() if "game_id" in master.columns else 0)
    log.info("Seasons: %s", sorted(master["season"].dropna().unique().tolist()) if "season" in master.columns else "?")
    log.info("=" * 60)

    return master


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build NCAAB feature matrix")
    parser.add_argument("--csv", action="store_true", help="Also save as CSV")
    args = parser.parse_args()

    df = build_feature_matrix(save_csv=args.csv)

    # Quick summary
    print("\n── Feature Matrix Summary ──────────────────────────────")
    print(f"Shape: {df.shape}")
    print(f"Date range: {df['date'].min().date()} → {df['date'].max().date()}" if "date" in df.columns else "")
    feature_groups = {
        "Rolling (5/10 game)": [c for c in df.columns if c.startswith("roll")],
        "BartTorvik (team)":   [c for c in df.columns if c.startswith("t_")],
        "BartTorvik (opp)":    [c for c in df.columns if c.startswith("o_")],
        "Matchup deltas":      [c for c in df.columns if any(c.startswith(p) for p in
                                ["off_eff", "def_eff", "net_", "efg_", "oreb_", "tov_", "tempo", "3p_", "sos_", "form_"])],
        "Situational":         [c for c in df.columns if any(c.startswith(p) for p in
                                ["days_rest", "is_home", "is_back", "is_short", "rest_adv", "games_played"])],
        "Odds/market":         [c for c in df.columns if c.startswith("vegas_") or c.startswith("implied_")],
        "H2H":                 [c for c in df.columns if c.startswith("h2h_")],
    }
    for group, cols in feature_groups.items():
        print(f"  {group}: {len(cols)} features")
    print("────────────────────────────────────────────────────────")