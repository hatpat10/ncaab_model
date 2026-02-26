# processing/database.py
import sqlite3
import os
import sys

# Add project root to path so config.py is findable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DB_PATH

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Master team table
    c.execute('''CREATE TABLE IF NOT EXISTS teams (
        team_id     TEXT PRIMARY KEY,
        full_name   TEXT,
        conference  TEXT,
        espn_id     TEXT,
        barttorvik_id TEXT
    )''')

    # Every game result
    c.execute('''CREATE TABLE IF NOT EXISTS games (
        game_id     TEXT PRIMARY KEY,
        date        TEXT,
        season      INTEGER,
        home_team_id TEXT,
        away_team_id TEXT,
        home_score  INTEGER,
        away_score  INTEGER,
        neutral     INTEGER,
        completed   INTEGER,
        venue       TEXT,
        conference_game INTEGER
    )''')

    # Team-level box score per game (one row per team per game)
    c.execute('''CREATE TABLE IF NOT EXISTS team_box (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        game_id     TEXT,
        team_id     TEXT,
        season      INTEGER,
        game_date   TEXT,
        home_away   TEXT,
        score       INTEGER,
        winner      INTEGER,
        fgm         INTEGER,
        fga         INTEGER,
        fg3m        INTEGER,
        fg3a        INTEGER,
        ftm         INTEGER,
        fta         INTEGER,
        oreb        INTEGER,
        dreb        INTEGER,
        ast         INTEGER,
        tov         INTEGER,
        stl         INTEGER,
        blk         INTEGER,
        pf          INTEGER,
        UNIQUE(game_id, team_id)
    )''')

    # Efficiency ratings (KenPom / BartTorvik) — date-stamped
    c.execute('''CREATE TABLE IF NOT EXISTS ratings (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        team_id     TEXT,
        date        TEXT,
        source      TEXT,
        adj_o       REAL,
        adj_d       REAL,
        adj_t       REAL,
        barthag     REAL,
        rank        INTEGER,
        UNIQUE(team_id, date, source)
    )''')

    # Betting lines
    c.execute('''CREATE TABLE IF NOT EXISTS odds (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        game_id     TEXT,
        bookmaker   TEXT,
        market      TEXT,
        home_line   REAL,
        away_line   REAL,
        total       REAL,
        fetched_at  TEXT,
        UNIQUE(game_id, bookmaker, market)
    )''')

    conn.commit()
    conn.close()
    print(f"Database initialized at {DB_PATH}")
    print("Tables created: teams, games, team_box, ratings, odds")

if __name__ == "__main__":
    init_db() 