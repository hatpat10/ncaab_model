# update_barttorvik.py
import pandas as pd
import requests
import time

def scrape_barttorvik(year: int) -> pd.DataFrame:
    url = f"https://barttorvik.com/trank.php?year={year}&json=1"
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    r.raise_for_status()
    rows = []
    for team in r.json():
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

dfs = []
for year in [2024, 2025, 2026]:
    print(f"Scraping BartTorvik {year}...")
    try:
        df = scrape_barttorvik(year)
        dfs.append(df)
        print(f"  {year}: {len(df)} teams")
    except Exception as e:
        print(f"  {year}: FAILED — {e}")
    time.sleep(2)

combined = pd.concat(dfs, ignore_index=True)
combined.to_parquet("data/raw/barttorvik_2024_2025.parquet", index=False)
print(f"\nSaved {len(combined)} total rows to data/raw/barttorvik_2024_2025.parquet")
print(combined.groupby("year").size())
