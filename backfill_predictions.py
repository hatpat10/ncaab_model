# backfill_predictions.py
# Re-runs the pipeline for every date that has a prediction JSON file
# This fills in 'actual' scores for completed games so backtest_ats.py can grade them

import subprocess
import glob
import os

files = sorted(glob.glob("predictions/2026-*.json"))
dates = [os.path.basename(f)[:10] for f in files]

# Only backfill dates up to yesterday (today's games aren't complete yet)
from datetime import datetime, timedelta
yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
dates_to_backfill = [d for d in dates if d <= yesterday]

print(f"Found {len(dates_to_backfill)} dates to backfill: {dates_to_backfill[0]} → {dates_to_backfill[-1]}")
print("This will re-run the pipeline for each date to fill in actual scores...\n")

success, failed = [], []

for date in dates_to_backfill:
    print(f"  Processing {date}...", end=" ", flush=True)
    result = subprocess.run(
        ["python", "predictions/daily_pipeline.py", "--date", date],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        # Check if it saved predictions
        if "Saved" in result.stdout or "predictions" in result.stdout.lower():
            print("✓")
            success.append(date)
        else:
            print("? (no output)")
            success.append(date)
    else:
        print(f"✗ ERROR")
        print(f"    {result.stderr[-200:] if result.stderr else 'no stderr'}")
        failed.append(date)

print(f"\n{'='*50}")
print(f"Done: {len(success)} succeeded, {len(failed)} failed")
if failed:
    print(f"Failed dates: {failed}")
print(f"\nNow run: python backtest_ats.py")