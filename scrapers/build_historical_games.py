"""
scrapers/build_historical_games.py
====================================
Builds historical games_raw entries (2020-2025 seasons) from the
hoopR_schedule_2020_2025.parquet file already on disk.

hoopR schedule has all the fields we need:
  game_id, game_date, home_team_name, away_team_name,
  home_score, away_score, neutral_site, season

This script:
  1. Loads hoopR schedule parquet
  2. Normalizes to games_raw schema
  3. Filters to completed D1 games only
  4. Inserts into SQLite games_raw (skips existing game_ids)
  5. Reports coverage

Usage:
    python -m scrapers.build_historical_games
    python -m scrapers.build_historical_games --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT     = Path(__file__).resolve().parent.parent
RAW      = ROOT / "data" / "raw"
DB_PATH  = ROOT / "data" / "ncaab.db"
SCHEDULE = RAW / "hoopR_schedule_2020_2025.parquet"

# Non-D1 keywords to filter out
NON_D1_KEYWORDS = [
    "biblical", "bethesda", "ecclesia", "champion_christian",
    "mid_atlantic_christian", "lincoln_university", "noble", "nobel",
    "naia", "juco", "community_college", "bible", "seminary",
    "christian_college", "east_west_university", "shenango", "toccoa",
    "fisk", "cleary", "oaklanders", "penn_state_shenango",
    "fort_lauderdale", "southwestern_christian", "southwestern_adventist",
    "trinity_college_of_jacksonville", "central_penn", "dallas_christian",
    "bryan__tn_", "virginia_lynchburg",
]


def is_non_d1(name: str) -> bool:
    n = str(name).lower()
    return any(k in n for k in NON_D1_KEYWORDS)


def load_hoopr_schedule() -> pd.DataFrame:
    """Load hoopR schedule and normalize to games_raw schema."""
    log.info("Loading hoopR schedule: %s", SCHEDULE)
    df = pd.read_parquet(SCHEDULE)
    log.info("Raw schedule: %d rows, cols: %s", len(df), df.columns.tolist()[:20])

    # Identify columns (hoopR versions differ slightly)
    col_map = {}

    # game_id
    for c in ["game_id", "id"]:
        if c in df.columns:
            col_map["game_id"] = c
            break

    # date
    for c in ["game_date", "date", "start_date"]:
        if c in df.columns:
            col_map["date"] = c
            break

    # home team
    for c in ["home_display_name", "home_team_location", "home_team_name",
              "home_team", "home_name"]:
        if c in df.columns:
            col_map["home_team"] = c
            break

    # away team
    for c in ["away_display_name", "away_team_location", "away_team_name",
              "away_team", "away_name"]:
        if c in df.columns:
            col_map["away_team"] = c
            break

    # scores
    for c in ["home_score", "home_points", "home_team_score"]:
        if c in df.columns:
            col_map["home_score"] = c
            break

    for c in ["away_score", "away_points", "away_team_score"]:
        if c in df.columns:
            col_map["away_score"] = c
            break

    # neutral
    for c in ["neutral_site", "neutral"]:
        if c in df.columns:
            col_map["neutral"] = c
            break

    # season
    for c in ["season", "season_type", "year"]:
        if c in df.columns and df[c].dtype in ["int64", "float64", "int32"]:
            col_map["season"] = c
            break

    # completed flag
    for c in ["status_type_completed", "game_played", "completed", "status_completed"]:
        if c in df.columns:
            col_map["completed"] = c
            break

    log.info("Column mapping: %s", col_map)
    missing = [k for k in ["game_id", "date", "home_team", "away_team",
                            "home_score", "away_score"] if k not in col_map]
    if missing:
        log.error("Missing required columns: %s", missing)
        log.error("Available cols: %s", df.columns.tolist())
        raise ValueError(f"Cannot map required columns: {missing}")

    # Build normalized frame
    out = pd.DataFrame()
    out["game_id"]   = df[col_map["game_id"]].astype(str)
    out["date"]      = pd.to_datetime(df[col_map["date"]], errors="coerce").dt.date.astype(str)
    out["home_team"] = df[col_map["home_team"]].astype(str)
    out["away_team"] = df[col_map["away_team"]].astype(str)
    out["home_score"]= pd.to_numeric(df[col_map["home_score"]], errors="coerce")
    out["away_score"]= pd.to_numeric(df[col_map["away_score"]], errors="coerce")
    out["neutral"]   = pd.to_numeric(df[col_map.get("neutral", "")], errors="coerce").fillna(0).astype(int) \
                       if "neutral" in col_map else 0
    out["completed"] = 1  # will filter below

    # Season: hoopR uses the ending year (2021 = 2020-21 season)
    if "season" in col_map:
        out["season"] = pd.to_numeric(df[col_map["season"]], errors="coerce").astype("Int64")
    else:
        # Derive from date
        dates = pd.to_datetime(df[col_map["date"]], errors="coerce")
        out["season"] = dates.dt.year.where(dates.dt.month >= 10, dates.dt.year - 1) + 1

    # Conference game flag (optional)
    for c in ["conference_game", "season_type"]:
        if c in df.columns:
            out["conference_game"] = df[c].astype(str)
            break
    else:
        out["conference_game"] = None

    # Venue (optional)
    for c in ["venue_name", "venue", "arena"]:
        if c in df.columns:
            out["venue"] = df[c].astype(str)
            break
    else:
        out["venue"] = None

    # Home/away team IDs (same as names for now)
    out["home_team_id"] = out["home_team"]
    out["away_team_id"] = out["away_team"]

    return out


def filter_games(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only completed D1 games with valid scores."""
    before = len(df)

    # Must have valid scores (completed games)
    df = df.dropna(subset=["home_score", "away_score"])
    df = df[(df["home_score"] > 0) | (df["away_score"] > 0)]

    # Filter non-D1
    mask = df["home_team"].apply(is_non_d1) | df["away_team"].apply(is_non_d1)
    df = df[~mask]

    # Must have valid date
    df = df.dropna(subset=["date"])
    df = df[df["date"] != "NaT"]

    # Deduplicate
    df = df.drop_duplicates(subset=["game_id"])

    log.info("Filtered: %d → %d games (removed %d)", before, len(df), before - len(df))
    return df.reset_index(drop=True)


def insert_to_sqlite(df: pd.DataFrame, dry_run: bool = False) -> int:
    """Insert new games into games_raw, skip existing game_ids."""
    con = sqlite3.connect(DB_PATH)

    # Get existing game_ids
    existing = set(pd.read_sql("SELECT game_id FROM games_raw", con)["game_id"].astype(str))
    log.info("Existing games in DB: %d", len(existing))

    # Only insert new games
    new = df[~df["game_id"].isin(existing)].copy()
    log.info("New games to insert: %d", len(new))

    if dry_run:
        log.info("[DRY RUN] Would insert %d games", len(new))
        log.info("Season breakdown:\n%s", new["season"].value_counts().sort_index().to_string())
        con.close()
        return len(new)

    if len(new) == 0:
        log.info("Nothing to insert — all games already in DB")
        con.close()
        return 0

    # Ensure schema matches games_raw
    cols = ["game_id", "date", "season", "home_team_id", "away_team_id",
            "home_team", "away_team", "home_score", "away_score",
            "neutral", "completed", "venue", "conference_game"]

    for c in cols:
        if c not in new.columns:
            new[c] = None

    new[cols].to_sql("games_raw", con, if_exists="append", index=False)
    con.commit()

    # Verify
    total = pd.read_sql("SELECT COUNT(*) as n FROM games_raw", con).iloc[0]["n"]
    log.info("Insert complete. Total games in DB: %d", total)

    # Season breakdown
    seasons = pd.read_sql("SELECT season, COUNT(*) as n FROM games_raw GROUP BY season ORDER BY season", con)
    log.info("Games by season:\n%s", seasons.to_string(index=False))

    con.close()
    return len(new)


def main():
    parser = argparse.ArgumentParser(description="Build historical games from hoopR schedule")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be inserted without writing to DB")
    args = parser.parse_args()

    # Load and normalize
    df = load_hoopr_schedule()

    # Filter
    df = filter_games(df)

    log.info("Season breakdown in hoopR schedule:")
    log.info("\n%s", df["season"].value_counts().sort_index().to_string())
    log.info("Date range: %s → %s", df["date"].min(), df["date"].max())
    log.info("Sample rows:")
    log.info("\n%s", df[["game_id","date","season","home_team","away_team",
                          "home_score","away_score"]].head(5).to_string())

    # Insert
    n = insert_to_sqlite(df, dry_run=args.dry_run)

    if not args.dry_run:
        log.info("Done. Inserted %d new historical games.", n)
        log.info("Next steps:")
        log.info("  1. python -m processing.feature_builder   (rebuild feature matrix)")
        log.info("  2. python -m models.model_trainer          (retrain with more data)")


if __name__ == "__main__":
    main()