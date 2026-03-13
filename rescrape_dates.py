"""
rescrape_dates.py
Re-scrapes ESPN for specific dates where games_raw is missing games.
Run this to fill in gaps before running backtest_ats.py.
"""
import subprocess, sys

# Dates with known gaps in clean period
DATES_TO_RESCRAPE = [
    "2026-03-05",
    "2026-03-06", 
    "2026-03-07",
    "2026-03-08",
    "2026-03-09",
    "2026-03-10",
]

def main():
    print("Re-scraping ESPN for dates with missing games...")
    print("(This will INSERT new games but skip duplicates)\n")
    
    for date in DATES_TO_RESCRAPE:
        print(f"  Scraping {date}...", end=" ", flush=True)
        result = subprocess.run(
            [sys.executable, "scrapers/espn_scraper.py", "--date", date],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            # Count new games from output
            lines = result.stdout.strip().splitlines()
            summary = next((l for l in reversed(lines) if l.strip()), "done")
            print(f"✓  {summary}")
        else:
            print(f"✗  Error: {result.stderr.strip()[:80]}")
    
    print("\nDone. Now run: python backtest_ats.py")

if __name__ == "__main__":
    main()