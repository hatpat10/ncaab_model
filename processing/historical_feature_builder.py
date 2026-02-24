"""
processing/historical_feature_builder.py
==========================================
NCAAB ML — Historical Feature Builder
2025-2026 Season • February 23, 2026

Builds a feature matrix from hoopR historical data (2020-2025 seasons)
using the same schema as feature_builder.py, then stacks them together
for a full 6-season training dataset (~70k team-game rows).

Data sources used:
  - data/raw/hoopR_schedule_2020_2025.parquet  → game results (scores, venues)
  - data/raw/hoopR_team_box_2020_2025.parquet  → team box scores per game
  - data/raw/barttorvik_*.parquet              → advanced ratings by season

Output:
  - data/processed/feature_matrix_historical.parquet  (2020-2025 seasons)
  - data/processed/feature_matrix_full.parquet        (all seasons stacked)

Usage:
    python -m processing.historical_feature_builder
    python -m processing.historical_feature_builder --audit   # check columns first
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT         = Path(__file__).resolve().parent.parent
DATA_RAW     = ROOT / "data" / "raw"
DATA_PROC    = ROOT / "data" / "processed"
ALIASES_PATH = ROOT / "data" / "team_aliases.json"

DATA_PROC.mkdir(parents=True, exist_ok=True)

# ── Load aliases (reuse from feature_builder) ─────────────────────────────────
_ALIASES: dict = {}

def _load_aliases() -> dict:
    if ALIASES_PATH.exists():
        with open(ALIASES_PATH) as f:
            return json.load(f)
    log.warning("team_aliases.json not found")
    return {}

def normalize_team(raw_name: str) -> str:
    global _ALIASES
    if not _ALIASES:
        _ALIASES = _load_aliases()
    if not isinstance(raw_name, str):
        return str(raw_name)
    name = raw_name.strip()
    if name in _ALIASES: return _ALIASES[name]
    lower = name.lower()
    for key, val in _ALIASES.items():
        if key.lower() == lower: return val
    words = name.split()
    for n_drop in range(1, min(3, len(words))):
        shorter = " ".join(words[:-n_drop])
        if shorter in _ALIASES: return _ALIASES[shorter]
        for key, val in _ALIASES.items():
            if key.lower() == shorter.lower(): return val
    slug_input = name.lower().replace(" ", "_").replace("-", "_").replace("&", "").replace(".", "")
    if slug_input in _ALIASES: return _ALIASES[slug_input]
    slug = name.lower()
    for junk in [
        " tigers", " eagles", " bulldogs", " wildcats", " bears", " hawks",
        " panthers", " lions", " wolves", " warriors", " knights", " cardinals",
        " rams", " bobcats", " owls", " ravens", " falcons", " ducks", " beavers",
        " gators", " seminoles", " hurricanes", " tar heels", " blue devils",
        " golden eagles", " golden bears", " red hawks", " fighting hawks",
        " running rebels", " aggies", " cowboys", " longhorns", " horned frogs",
        " jayhawks", " cyclones", " cornhuskers", " huskers", " sooners",
        " boilermakers", " hoosiers", " badgers", " gophers", " wolverines",
        " spartans", " buckeyes", " hawkeyes", " illini", " terrapins", " terps",
        " hokies", " cavaliers", " orange", " golden hurricane", " racers",
        " skyhawks", " ospreys", " mavericks", " mastodons", " colonials",
        " toreros", " dons", " pioneers", " jaspers", " highlanders", " jaguars",
        " antelopes", " lopes", " delta devils", " grizzlies", " mountaineers",
        " screaming eagles", " trailblazers", " wolfpack", " redhawks", " gamecocks",
    ]:
        slug = slug.replace(junk, "")
    slug = slug.strip().replace(" ", "_").replace("-", "_").replace("&", "")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug


# ──────────────────────────────────────────────────────────────────────────────
# AUDIT — Inspect column names before building
# ──────────────────────────────────────────────────────────────────────────────

def audit_sources():
    """Print column names and sample rows for all hoopR sources."""
    print("\n" + "="*60)
    print("AUDIT: hoopR source columns")
    print("="*60)

    schedule_path = DATA_RAW / "hoopR_schedule_2020_2025.parquet"
    box_path      = DATA_RAW / "hoopR_team_box_2020_2025.parquet"

    if schedule_path.exists():
        sched = pd.read_parquet(schedule_path)
        print(f"\nSCHEDULE  →  {sched.shape}")
        print(f"Columns: {sched.columns.tolist()}")
        print(f"Sample:\n{sched.head(2).to_string()}")
        # Key columns we need
        for col in ["game_id", "date", "season", "home_team_id", "away_team_id",
                    "home_score", "away_score", "home_team_location", "away_team_location",
                    "neutral_site", "completed"]:
            present = col in sched.columns
            print(f"  {'✓' if present else '✗'}  {col}")
    else:
        print("SCHEDULE parquet not found!")

    if box_path.exists():
        box = pd.read_parquet(box_path)
        print(f"\nTEAM BOX  →  {box.shape}")
        print(f"Columns: {box.columns.tolist()}")
        print(f"Sample:\n{box.head(2).to_string()}")
        # Key columns we need
        for col in ["game_id", "team_id", "team_score", "opponent_team_score",
                    "field_goals_made", "field_goals_attempted",
                    "three_point_field_goals_made", "three_point_field_goals_attempted",
                    "free_throws_made", "free_throws_attempted",
                    "offensive_rebounds", "defensive_rebounds", "total_rebounds",
                    "assists", "turnovers", "steals", "blocks",
                    "team_location", "team_home_away"]:
            present = col in box.columns
            print(f"  {'✓' if present else '✗'}  {col}")


# ──────────────────────────────────────────────────────────────────────────────
# LOAD SOURCES
# ──────────────────────────────────────────────────────────────────────────────

def load_schedule() -> pd.DataFrame:
    path = DATA_RAW / "hoopR_schedule_2020_2025.parquet"
    log.info("Loading hoopR schedule: %s", path)
    df = pd.read_parquet(path)
    df.columns = [c.lower().strip() for c in df.columns]

    # Normalize date
    date_col = next((c for c in ["game_date", "date"] if c in df.columns), None)
    if date_col:
        df["date"] = pd.to_datetime(df[date_col], errors="coerce")

    # Season: use existing or compute from date
    if "season" not in df.columns and "date" in df.columns:
        df["season"] = df["date"].dt.year.where(
            df["date"].dt.month >= 10, df["date"].dt.year - 1) + 1
    df["season"] = pd.to_numeric(df["season"], errors="coerce")

    # Keep only completed games
    if "completed" in df.columns:
        df = df[df["completed"].astype(str).str.lower().isin(["1", "true", "yes"])]
        log.info("Schedule after filtering completed: %d games", len(df))

    # Identify score columns
    home_score = next((c for c in ["home_score", "home_team_score", "home_points"] if c in df.columns), None)
    away_score = next((c for c in ["away_score", "away_team_score", "away_points"] if c in df.columns), None)

    if home_score:
        df["home_score"] = pd.to_numeric(df[home_score], errors="coerce")
    if away_score:
        df["away_score"] = pd.to_numeric(df[away_score], errors="coerce")

    # Drop games without scores
    df = df.dropna(subset=["home_score", "away_score"])
    df = df[(df["home_score"] > 0) & (df["away_score"] > 0)]

    # game_id
    if "game_id" in df.columns:
        df["game_id"] = df["game_id"].astype(str)

    log.info("Schedule loaded: %d completed games", len(df))
    return df


def load_box() -> pd.DataFrame:
    path = DATA_RAW / "hoopR_team_box_2020_2025.parquet"
    log.info("Loading hoopR team box: %s", path)
    df = pd.read_parquet(path)
    df.columns = [c.lower().strip() for c in df.columns]

    if "game_id" in df.columns:
        df["game_id"] = df["game_id"].astype(str)

    # Normalize date + season
    date_col = next((c for c in ["game_date", "date"] if c in df.columns), None)
    if date_col:
        df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    if "season" not in df.columns and "date" in df.columns:
        df["season"] = df["date"].dt.year.where(
            df["date"].dt.month >= 10, df["date"].dt.year - 1) + 1
    df["season"] = pd.to_numeric(df["season"], errors="coerce")

    # team_id: find the best available name column
    name_col = next((c for c in ["team_location", "team_display_name", "team_name",
                                  "team_short_display_name"] if c in df.columns), None)
    if name_col:
        df["team_id"] = df[name_col].apply(normalize_team)
    elif "team_id" in df.columns:
        df["team_id"] = df["team_id"].apply(normalize_team)

    # Standardize score columns
    score_col = next((c for c in ["team_score", "score", "points",
                                   "team_points"] if c in df.columns), None)
    if score_col and score_col != "team_score":
        df["team_score"] = pd.to_numeric(df[score_col], errors="coerce")
    elif score_col:
        df["team_score"] = pd.to_numeric(df["team_score"], errors="coerce")

    opp_score_col = next((c for c in ["opponent_team_score", "opp_score",
                                       "opp_points"] if c in df.columns), None)
    if opp_score_col and opp_score_col != "opp_score":
        df["opp_score"] = pd.to_numeric(df[opp_score_col], errors="coerce")
    elif opp_score_col:
        df["opp_score"] = pd.to_numeric(df["opp_score"], errors="coerce")

    # home/away flag
    ha_col = next((c for c in ["team_home_away", "home_away", "location"] if c in df.columns), None)
    if ha_col:
        df["is_home"] = (df[ha_col].str.lower() == "home").astype(int)
    elif "team_location" in df.columns:
        df["is_home"] = (df["team_location"].str.lower() == "home").astype(int)

    log.info("Box scores loaded: %d rows, %d cols", len(df), df.shape[1])
    return df


def load_barttorvik() -> pd.DataFrame:
    frames = []
    for fname in ["barttorvik_2020_2023.parquet", "barttorvik_2024_2025.parquet"]:
        p = DATA_RAW / fname
        if p.exists():
            frames.append(pd.read_parquet(p))
    if not frames:
        raise FileNotFoundError("No BartTorvik parquets found")
    df = pd.concat(frames, ignore_index=True)
    df.columns = [c.lower().strip() for c in df.columns]
    df["team_id"] = df["team"].apply(normalize_team)
    # year+1 = the season it's used for (same logic as feature_builder.py)
    df["join_season"] = pd.to_numeric(df["year"], errors="coerce") + 1
    df = df.rename(columns={
        "adj_o_rk": "adj_o_rank", "adj_d_rk": "adj_d_rank",
        "adj_t_rk": "adj_t_rank", "barthag_rk": "barthag_rank",
    })
    # Drop rows where key metrics are all null
    df = df.dropna(subset=["adj_o", "adj_d"], how="all")
    log.info("BartTorvik loaded: %d rows, seasons %s",
             len(df), sorted(df["join_season"].dropna().unique().tolist()))
    return df


# ──────────────────────────────────────────────────────────────────────────────
# BUILD MASTER GAME TABLE (team-per-row) from hoopR
# ──────────────────────────────────────────────────────────────────────────────

NON_D1_KEYWORDS = [
    "biblical", "bethesda", "ecclesia", "champion_christian", "mid_atlantic_christian",
    "lincoln_university", "noble", "nobel", "naia", "juco", "community_college",
    "bible", "seminary", "christian_college", "east_west_university", "shenango",
    "toccoa", "fisk", "cleary", "oaklanders", "penn_state_shenango",
    "virginia_lynchburg", "fort_lauderdale", "bryan__tn_", "southwestern_christian",
    "southwestern_adventist", "trinity_college_of_jacksonville", "central_penn",
    "dallas_christian",
]

def is_non_d1(team_id: str) -> bool:
    t = str(team_id).lower()
    return any(kw in t for kw in NON_D1_KEYWORDS)


def build_master_from_box(box: pd.DataFrame) -> pd.DataFrame:
    """
    The hoopR team box is already in team-per-row format.
    Each row = one team's stats for one game.
    We just need to add opp_id and opp_score.
    """
    log.info("Building master game table from hoopR box scores...")

    df = box.copy()

    # We need to find each team's opponent — pair rows by game_id
    if "game_id" not in df.columns:
        raise ValueError("game_id not in box scores")

    # Each game has 2 rows — home team and away team
    # Match them up by game_id
    g1 = df[["game_id", "team_id", "team_score", "is_home", "date", "season"]].copy()
    g1.columns = ["game_id", "team_id", "team_score", "is_home", "date", "season"]

    # Get opponent by merging with itself on game_id, different team
    g2 = g1.rename(columns={"team_id": "opp_id", "team_score": "opp_score",
                              "is_home": "opp_is_home"})

    master = g1.merge(g2[["game_id", "opp_id", "opp_score"]], on="game_id", how="inner")
    # Remove self-joins
    master = master[master["team_id"] != master["opp_id"]]
    # Each game now has 2 rows (one per team perspective) — deduplicate
    master = master.drop_duplicates(subset=["game_id", "team_id"])

    # Add derived columns
    master["margin"]      = master["team_score"] - master["opp_score"]
    master["win"]         = (master["margin"] > 0).astype(int)
    master["total_score"] = master["team_score"] + master["opp_score"]
    master["is_neutral"]  = 0  # hoopR schedule has neutral info but we'll add later

    # Non-D1 filter
    before = len(master)
    mask = master["team_id"].apply(is_non_d1) | master["opp_id"].apply(is_non_d1)
    master = master[~mask].reset_index(drop=True)
    if before - len(master):
        log.info("Dropped %d non-D1 rows", before - len(master))

    log.info("Master from box: %d rows (%d games)", len(master),
             master["game_id"].nunique())
    return master


def enrich_from_schedule(master: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """Add neutral site flag and venue from schedule."""
    sched_cols = ["game_id"]
    if "neutral_site" in schedule.columns:
        sched_cols.append("neutral_site")
    if "venue" in schedule.columns:
        sched_cols.append("venue")

    schedule["game_id"] = schedule["game_id"].astype(str) if "game_id" in schedule.columns else None
    if "game_id" in schedule.columns:
        master = master.merge(schedule[sched_cols].drop_duplicates("game_id"),
                              on="game_id", how="left")
        if "neutral_site" in master.columns:
            master["is_neutral"] = pd.to_numeric(master["neutral_site"], errors="coerce").fillna(0).astype(int)

    return master


# ──────────────────────────────────────────────────────────────────────────────
# ROLLING FEATURES (same logic as feature_builder.py)
# ──────────────────────────────────────────────────────────────────────────────

def add_rolling_features(df: pd.DataFrame, box: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling stats from box scores, shift(1) to prevent leakage."""
    log.info("Adding rolling features...")

    # Map box score stats to team_id+date
    stat_cols = []
    col_map = {
        "team_score": "pts", "margin": "margin",
    }
    # Find available box score columns
    fg_made = next((c for c in ["field_goals_made", "fgm", "fg_made"] if c in box.columns), None)
    fg_att  = next((c for c in ["field_goals_attempted", "fga", "fg_att"] if c in box.columns), None)
    fg3_made = next((c for c in ["three_point_field_goals_made", "fg3m", "three_made"] if c in box.columns), None)
    fg3_att  = next((c for c in ["three_point_field_goals_attempted", "fg3a", "three_att"] if c in box.columns), None)
    ft_made  = next((c for c in ["free_throws_made", "ftm", "ft_made"] if c in box.columns), None)
    ft_att   = next((c for c in ["free_throws_attempted", "fta", "ft_att"] if c in box.columns), None)
    oreb = next((c for c in ["offensive_rebounds", "oreb", "off_reb"] if c in box.columns), None)
    dreb = next((c for c in ["defensive_rebounds", "dreb", "def_reb"] if c in box.columns), None)
    treb = next((c for c in ["total_rebounds", "treb", "rebounds"] if c in box.columns), None)
    ast  = next((c for c in ["assists", "ast"] if c in box.columns), None)
    tov  = next((c for c in ["turnovers", "tov", "to"] if c in box.columns), None)
    stl  = next((c for c in ["steals", "stl"] if c in box.columns), None)
    blk  = next((c for c in ["blocks", "blk"] if c in box.columns), None)
    pf   = next((c for c in ["fouls", "personal_fouls", "pf"] if c in box.columns), None)

    # Build a lean stat table
    stat_df = box[["game_id", "team_id", "date", "season"]].copy()
    if "team_score" in box.columns:
        stat_df["pts"] = pd.to_numeric(box["team_score"], errors="coerce")
    if "opp_score" in box.columns and "team_score" in box.columns:
        stat_df["margin"] = pd.to_numeric(box["team_score"], errors="coerce") - \
                            pd.to_numeric(box["opp_score"], errors="coerce")
    for src, dst in [(fg_made,"fgm"),(fg_att,"fga"),(fg3_made,"fg3m"),(fg3_att,"fg3a"),
                     (ft_made,"ftm"),(ft_att,"fta"),(oreb,"oreb"),(dreb,"dreb"),
                     (treb,"treb"),(ast,"ast"),(tov,"tov"),(stl,"stl"),(blk,"blk"),(pf,"pf")]:
        if src:
            stat_df[dst] = pd.to_numeric(box[src], errors="coerce")

    # Compute derived efficiency metrics
    if all(c in stat_df.columns for c in ["fgm","fg3m","fta"]):
        stat_df["ts"] = stat_df["pts"] / (2 * (stat_df["fga"] + 0.44 * stat_df["fta"]))
    if all(c in stat_df.columns for c in ["fgm","fg3m","fga"]):
        stat_df["efg"] = (stat_df["fgm"] + 0.5 * stat_df["fg3m"]) / stat_df["fga"]
    if all(c in stat_df.columns for c in ["tov","fga","fta"]):
        stat_df["tov_rate"] = stat_df["tov"] / (stat_df["fga"] + 0.44 * stat_df["fta"] + stat_df["tov"])
    if "ftm" in stat_df.columns and "fta" in stat_df.columns:
        stat_df["ft_pct"] = stat_df["ftm"] / stat_df["fta"].replace(0, np.nan)
    if "fgm" in stat_df.columns and "fga" in stat_df.columns:
        stat_df["fg_pct"] = stat_df["fgm"] / stat_df["fga"].replace(0, np.nan)

    stat_df = stat_df.sort_values(["team_id", "date"])

    roll_cols = [c for c in ["pts","margin","oreb","dreb","treb","ast","tov","stl","blk","pf",
                              "fgm","fga","fg3m","fg3a","ftm","fta","ts","efg","tov_rate",
                              "ft_pct","fg_pct"] if c in stat_df.columns]

    for window in [5, 10]:
        for col in roll_cols:
            feat_name = f"roll{window}_{col}"
            stat_df[feat_name] = (
                stat_df.groupby("team_id")[col]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=2).mean())
            )

    # Win streak
    if "margin" in stat_df.columns:
        stat_df["win_result"] = (stat_df["margin"] > 0).astype(float)
        stat_df["roll5_win_streak"] = (
            stat_df.groupby("team_id")["win_result"]
            .transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum())
        )

    # games_played
    stat_df["games_played"] = (
        stat_df.groupby("team_id").cumcount()
    )

    # Merge rolling stats back onto master
    roll_feat_cols = [c for c in stat_df.columns if c.startswith("roll") or c == "games_played"]
    merge_cols = ["game_id", "team_id"] + roll_feat_cols

    df = df.merge(stat_df[merge_cols].drop_duplicates(["game_id","team_id"]),
                  on=["game_id","team_id"], how="left")

    log.info("Rolling features added: %d cols", len(roll_feat_cols))
    return df


# ──────────────────────────────────────────────────────────────────────────────
# BARTTORVIK MERGE (same logic as feature_builder.py)
# ──────────────────────────────────────────────────────────────────────────────

TORVIK_METRICS = ["adj_o", "adj_o_rank", "adj_d", "adj_d_rank", "adj_t", "adj_t_rank",
                   "barthag", "barthag_rank", "wab", "nc_elite_sos", "ov_elite_sos",
                   "nc_cur_sos", "ov_cur_sos", "conf", "seed"]

def merge_barttorvik(master: pd.DataFrame, torvik: pd.DataFrame) -> pd.DataFrame:
    torvik_cols = ["team_id", "join_season"] + [c for c in TORVIK_METRICS if c in torvik.columns]
    tv = torvik[torvik_cols].drop_duplicates(["team_id","join_season"])

    # Team metrics
    master = master.merge(
        tv.rename(columns={"join_season": "season",
                            **{c: f"t_{c}" for c in TORVIK_METRICS if c in tv.columns}}),
        on=["team_id","season"], how="left"
    )
    # Opponent metrics
    opp_tv = tv.rename(columns={"team_id": "opp_id", "join_season": "season",
                                  **{c: f"o_{c}" for c in TORVIK_METRICS if c in tv.columns}})
    master = master.merge(opp_tv, on=["opp_id","season"], how="left")

    log.info("BartTorvik merged. Columns: %d", master.shape[1])
    return master


# ──────────────────────────────────────────────────────────────────────────────
# SITUATIONAL FEATURES
# ──────────────────────────────────────────────────────────────────────────────

def add_situational_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["team_id","date"])

    df["days_rest"] = (
        df.groupby("team_id")["date"]
        .transform(lambda x: x.diff().dt.days)
    ).fillna(3.0)

    df["is_back_to_back"] = (df["days_rest"] == 1).astype(int)
    df["is_short_rest"]   = (df["days_rest"] <= 2).astype(int)

    if "days_rest" in df.columns:
        opp_rest = df[["game_id","team_id","days_rest"]].rename(
            columns={"team_id":"opp_id","days_rest":"opp_days_rest"})
        df = df.merge(opp_rest, on=["game_id","opp_id"], how="left")
        df["rest_advantage"] = df["days_rest"] - df["opp_days_rest"].fillna(3.0)

    return df


# ──────────────────────────────────────────────────────────────────────────────
# MATCHUP DELTA FEATURES
# ──────────────────────────────────────────────────────────────────────────────

def add_matchup_features(df: pd.DataFrame) -> pd.DataFrame:
    if "t_adj_o" in df.columns and "o_adj_d" in df.columns:
        df["off_eff_delta"] = df["t_adj_o"] - df["o_adj_d"]
    if "t_adj_d" in df.columns and "o_adj_o" in df.columns:
        df["def_eff_delta"] = df["o_adj_o"] - df["t_adj_d"]
    if "t_adj_o" in df.columns and "t_adj_d" in df.columns and \
       "o_adj_o" in df.columns and "o_adj_d" in df.columns:
        df["net_rating_delta"] = (df["t_adj_o"] - df["t_adj_d"]) - \
                                  (df["o_adj_o"] - df["o_adj_d"])
    if "t_adj_t" in df.columns and "o_adj_t" in df.columns:
        df["tempo_delta"] = df["t_adj_t"] - df["o_adj_t"]
    if "t_wab" in df.columns and "o_wab" in df.columns:
        df["wab_delta"] = df["t_wab"] - df["o_wab"]
    if "t_ov_cur_sos" in df.columns and "o_ov_cur_sos" in df.columns:
        df["sos_delta"] = df["t_ov_cur_sos"] - df["o_ov_cur_sos"]

    # Defense tier
    rank_col = next((c for c in ["o_adj_d_rank","o_adj_d_rk"] if c in df.columns), None)
    if rank_col:
        df["opp_def_tier"] = pd.cut(
            pd.to_numeric(df[rank_col], errors="coerce"),
            bins=[0,25,75,150,400], labels=["elite","good","average","weak"]
        ).astype(str)
        tier_map = {"elite":-10.0,"good":-5.0,"average":0.0,"weak":4.0,"nan":0.0}
        df["def_suppression_factor"] = df["opp_def_tier"].map(tier_map).fillna(0.0)

    return df


# ──────────────────────────────────────────────────────────────────────────────
# H2H FEATURES
# ──────────────────────────────────────────────────────────────────────────────

def add_h2h_features(df: pd.DataFrame, lookback: int = 3) -> pd.DataFrame:
    df = df.sort_values("date")
    h2h_margins, h2h_wins = [], []

    for idx, row in df.iterrows():
        past = df[
            (df["team_id"] == row["team_id"]) &
            (df["opp_id"]  == row["opp_id"]) &
            (df["date"]    <  row["date"])
        ].tail(lookback)
        h2h_margins.append(past["margin"].mean() if len(past) else np.nan)
        h2h_wins.append(   past["win"].mean()    if len(past) else np.nan)

    df["h2h_avg_margin"] = h2h_margins
    df["h2h_win_rate"]   = h2h_wins
    return df


# ──────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def build_historical_features(seasons: list[int] | None = None) -> pd.DataFrame:
    """
    Build feature matrix from hoopR historical data.

    Args:
        seasons: list of seasons to include (e.g. [2021,2022,2023,2024,2025])
                 If None, uses all available seasons.
    """
    schedule = load_schedule()
    box      = load_box()
    torvik   = load_barttorvik()

    # Filter to requested seasons
    if seasons:
        box      = box[box["season"].isin(seasons)]
        schedule = schedule[schedule["season"].isin(seasons)] if "season" in schedule.columns else schedule

    log.info("Building master game table...")
    master = build_master_from_box(box)
    master = enrich_from_schedule(master, schedule)

    log.info("Adding rolling features...")
    master = add_rolling_features(master, box)

    log.info("Merging BartTorvik...")
    master = merge_barttorvik(master, torvik)

    log.info("Adding situational features...")
    master = add_situational_features(master)

    log.info("Adding matchup features...")
    master = add_matchup_features(master)

    log.info("Adding H2H features (this may take a few minutes)...")
    master = add_h2h_features(master)

    # Final cleanup
    master = master.dropna(subset=["margin","win","total_score"])

    seasons_present = sorted(master["season"].dropna().unique().tolist())
    log.info("Historical feature matrix: %d rows × %d cols | seasons %s",
             len(master), master.shape[1], seasons_present)

    return master


def main():
    parser = argparse.ArgumentParser(description="Historical Feature Builder")
    parser.add_argument("--audit",   action="store_true", help="Audit source columns only")
    parser.add_argument("--seasons", nargs="+", type=int,
                        help="Seasons to process (e.g. --seasons 2021 2022 2023)")
    parser.add_argument("--no-stack", action="store_true",
                        help="Don't stack with current season matrix")
    args = parser.parse_args()

    if args.audit:
        audit_sources()
        return

    # Build historical matrix
    hist = build_historical_features(seasons=args.seasons)
    hist_path = DATA_PROC / "feature_matrix_historical.parquet"
    hist.to_parquet(hist_path, index=False)
    log.info("Saved historical matrix: %s", hist_path)

    if not args.no_stack:
        # Stack with current season
        curr_path = DATA_PROC / "feature_matrix.parquet"
        if curr_path.exists():
            curr = pd.read_parquet(curr_path)
            log.info("Current season matrix: %d rows", len(curr))

            # Align columns — use intersection + fill missing with NaN
            all_cols = sorted(set(hist.columns) | set(curr.columns))
            hist_aligned = hist.reindex(columns=all_cols)
            curr_aligned = curr.reindex(columns=all_cols)

            full = pd.concat([hist_aligned, curr_aligned], ignore_index=True)
            full = full.sort_values("date").reset_index(drop=True)

            full_path = DATA_PROC / "feature_matrix_full.parquet"
            full.to_parquet(full_path, index=False)
            log.info("Stacked full matrix: %d rows × %d cols → %s",
                     len(full), full.shape[1], full_path)
            log.info("Season breakdown:\n%s",
                     full.groupby("season").size().to_string())
        else:
            log.warning("Current season matrix not found at %s", curr_path)

    log.info("Done. Run model trainer with full matrix:")
    log.info("  python -m models.model_trainer --feature-path data/processed/feature_matrix_full.parquet")


if __name__ == "__main__":
    main()