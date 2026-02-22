# processing/data_loader.py
import pandas as pd
import sqlite3
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DB_PATH

RAW = "data/raw"

def load_team_box() -> pd.DataFrame:
    """Load hoopR team box scores (2020-2025)."""
    return pd.read_parquet(f"{RAW}/hoopR_team_box_2020_2025.parquet")

def load_player_box() -> pd.DataFrame:
    """Load hoopR player box scores (2020-2025)."""
    return pd.read_parquet(f"{RAW}/hoopR_player_box_2020_2025.parquet")

def load_schedule() -> pd.DataFrame:
    """Load hoopR schedule (2020-2025)."""
    return pd.read_parquet(f"{RAW}/hoopR_schedule_2020_2025.parquet")

def load_barttorvik() -> pd.DataFrame:
    """Load BartTorvik ratings (2020-2025), combined from both sources."""
    bt_old = pd.read_parquet(f"{RAW}/barttorvik_2020_2023.parquet")
    bt_new = pd.read_parquet(f"{RAW}/barttorvik_2024_2025.parquet")
    
    # Align columns — toRvik and cbbdata use slightly different names
    common = list(set(bt_old.columns) & set(bt_new.columns))
    combined = pd.concat([bt_old[common], bt_new[common]], ignore_index=True)
    return combined.sort_values(["year", "team"]).reset_index(drop=True)

def load_current_season_games() -> pd.DataFrame:
    """Load 2025-26 ESPN scraped games from SQLite."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM games_raw ORDER BY date", conn)
    conn.close()
    return df

def load_odds() -> pd.DataFrame:
    """Load odds from SQLite."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM odds_raw ORDER BY date", conn)
    conn.close()
    return df

if __name__ == "__main__":
    print("Loading all data sources...\n")

    team_box = load_team_box()
    print(f"Team box:        {len(team_box):>8,} rows | cols: {team_box.shape[1]}")

    player_box = load_player_box()
    print(f"Player box:      {len(player_box):>8,} rows | cols: {player_box.shape[1]}")

    schedule = load_schedule()
    print(f"Schedule:        {len(schedule):>8,} rows | cols: {schedule.shape[1]}")

    bt = load_barttorvik()
    print(f"BartTorvik:      {len(bt):>8,} rows | years: {sorted(bt['year'].unique())}")

    games = load_current_season_games()
    print(f"Current season:  {len(games):>8,} rows | {games['date'].min()} to {games['date'].max()}")

    odds = load_odds()
    print(f"Odds:            {len(odds):>8,} rows | books: {odds['bookmaker'].nunique()}")

    print("\nAll sources loaded successfully.")