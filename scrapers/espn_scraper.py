# scrapers/espn_scraper.py
import requests
import time
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DB_PATH

BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"
HEADERS = {"User-Agent": "Mozilla/5.0"}

def get_scoreboard(date_str: str) -> dict:
    """Get all games for a date. date_str = YYYYMMDD format."""
    url = f"{BASE}/scoreboard"
    params = {"limit": 200, "dates": date_str, "groups": 50}
    r = requests.get(url, params=params, headers=HEADERS, timeout=15)
    r.raise_for_status()
    return r.json()

def parse_game(event: dict) -> dict:
    """Parse a single game event into a flat dict."""
    comp = event["competitions"][0]
    home = next(t for t in comp["competitors"] if t["homeAway"] == "home")
    away = next(t for t in comp["competitors"] if t["homeAway"] == "away")
    status = event["status"]["type"]

    return {
        "game_id":        int(event["id"]),
        "date":           event["date"][:10],
        "season":         2026,
        "home_team_id":   int(home["id"]),
        "away_team_id":   int(away["id"]),
        "home_team":      home["team"]["displayName"],
        "away_team":      away["team"]["displayName"],
        "home_score":     int(home.get("score", 0) or 0),
        "away_score":     int(away.get("score", 0) or 0),
        "neutral":        int(comp.get("neutralSite", False)),
        "completed":      int(status["completed"]),
        "venue":          comp.get("venue", {}).get("fullName", "Unknown"),
        "conference_game": int(comp.get("conferenceCompetition", False)),
    }

def scrape_date_range(start: str, end: str):
    """
    Scrape all games from start to end date.
    start/end: YYYY-MM-DD strings.
    """
    conn = sqlite3.connect(DB_PATH)
    
    # Create table if not exists
    conn.execute('''CREATE TABLE IF NOT EXISTS games_raw (
        game_id         INTEGER PRIMARY KEY,
        date            TEXT,
        season          INTEGER,
        home_team_id    INTEGER,
        away_team_id    INTEGER,
        home_team       TEXT,
        away_team       TEXT,
        home_score      INTEGER,
        away_score      INTEGER,
        neutral         INTEGER,
        completed       INTEGER,
        venue           TEXT,
        conference_game INTEGER
    )''')
    conn.commit()

    d = datetime.strptime(start, "%Y-%m-%d")
    end_d = datetime.strptime(end, "%Y-%m-%d")
    total_games = 0

    while d <= end_d:
        date_str = d.strftime("%Y%m%d")
        try:
            data = get_scoreboard(date_str)
            events = data.get("events", [])
            if events:
                parsed = [parse_game(g) for g in events if g.get("competitions")]
                df = pd.DataFrame(parsed)
                df.to_sql("games_raw", conn, if_exists="append", index=False)
                # Remove duplicates
                conn.execute('''DELETE FROM games_raw WHERE rowid NOT IN (
                    SELECT MIN(rowid) FROM games_raw GROUP BY game_id)''')
                conn.commit()
                total_games += len(parsed)
                print(f"{d.date()}: {len(parsed)} games")
            else:
                print(f"{d.date()}: no games")
        except Exception as e:
            print(f"{d.date()}: ERROR - {e}")
        
        time.sleep(1.2)
        d += timedelta(days=1)

    conn.close()
    print(f"\nDone. Total games scraped: {total_games}")

if __name__ == "__main__":
    # 2025-26 season started November 4, 2025
    # Scrape from opening day through yesterday
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"Scraping 2025-26 season through {yesterday}...")
    scrape_date_range("2025-11-04", yesterday)