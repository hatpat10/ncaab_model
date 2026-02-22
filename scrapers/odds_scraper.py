# scrapers/odds_scraper.py
import requests
import sqlite3
import pandas as pd
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ODDS_API_KEY, DB_PATH

BASE = "https://api.the-odds-api.com/v4"

def get_current_odds():
    """Get live odds for upcoming NCAAB games."""
    url = f"{BASE}/sports/basketball_ncaab/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": "h2h,spreads,totals",
        "oddsFormat": "american",
    }
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    remaining = r.headers.get("x-requests-remaining", "?")
    print(f"API requests remaining: {remaining}")
    return r.json()

def parse_odds(data: list) -> list:
    """Flatten odds response into rows."""
    rows = []
    for game in data:
        game_id = game["id"]
        home    = game["home_team"]
        away    = game["away_team"]
        date    = game["commence_time"][:10]

        for book in game.get("bookmakers", []):
            book_key = book["key"]
            for market in book.get("markets", []):
                market_key = market["key"]
                for outcome in market.get("outcomes", []):
                    rows.append({
                        "game_id":    game_id,
                        "date":       date,
                        "home_team":  home,
                        "away_team":  away,
                        "bookmaker":  book_key,
                        "market":     market_key,
                        "outcome":    outcome["name"],
                        "price":      outcome["price"],
                        "point":      outcome.get("point", None),
                        "scraped_at": pd.Timestamp.now().isoformat(),
                    })
    return rows

def get_historical_odds(date_str: str):
    """
    Get historical odds for a specific date.
    date_str: YYYY-MM-DDTHH:MM:SSZ format
    Note: uses more API credits, use sparingly.
    """
    url = f"{BASE}/sports/basketball_ncaab/odds-history"
    params = {
        "apiKey":     ODDS_API_KEY,
        "regions":    "us",
        "markets":    "h2h,spreads,totals",
        "oddsFormat": "american",
        "date":       date_str,
    }
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    return r.json()

def save_odds(rows: list):
    """Save odds rows to SQLite."""
    if not rows:
        print("No odds data to save.")
        return
    
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''CREATE TABLE IF NOT EXISTS odds_raw (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        game_id     TEXT,
        date        TEXT,
        home_team   TEXT,
        away_team   TEXT,
        bookmaker   TEXT,
        market      TEXT,
        outcome     TEXT,
        price       REAL,
        point       REAL,
        scraped_at  TEXT
    )''')
    conn.commit()

    df = pd.DataFrame(rows)
    df.to_sql("odds_raw", conn, if_exists="append", index=False)
    conn.close()
    print(f"Saved {len(rows)} odds rows to DB")

if __name__ == "__main__":
    print("Fetching current NCAAB odds...")
    data = get_current_odds()
    print(f"Games with odds: {len(data)}")
    
    rows = parse_odds(data)
    save_odds(rows)
    
    # Show sample
    if rows:
        df = pd.DataFrame(rows)
        print("\nSample - today's games:")
        print(df[["home_team","away_team","bookmaker","market","outcome","price"]].head(12).to_string(index=False))