# scrapers/barttorvik_scraper.py
import pandas as pd
import requests
import time

def scrape_barttorvik(year: int) -> pd.DataFrame:
    """Scrape team ratings from BartTorvik for a given season year."""
    url = f"https://barttorvik.com/trank.php?year={year}&json=1"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=15)
    r.raise_for_status()
    data = r.json()
    
    rows = []
    for team in data:
        rows.append({
            "team":    team[0],
            "conf":    team[1],
            "adj_o":   team[4],
            "adj_d":   team[5],
            "adj_t":   team[13],
            "barthag": team[10],
            "rank":    team[3],
            "year":    year,
        })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    dfs = []
    for year in [2024, 2025]:
        print(f"Scraping {year}...")
        df = scrape_barttorvik(year)
        dfs.append(df)
        print(f"  {year}: {len(df)} teams")
        time.sleep(2)
    
    combined = pd.concat(dfs, ignore_index=True)
    combined.to_parquet("data/raw/barttorvik_2024_2025.parquet", index=False)
    print(f"Saved {len(combined)} rows to data/raw/barttorvik_2024_2025.parquet")