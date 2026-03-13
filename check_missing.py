"""
check_missing.py
Check why March 5 and 7 show 0 completed games.
Run: python check_missing.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PRED_DIR = ROOT / "predictions"

for date_str in ["2026-03-05", "2026-03-07"]:
    pred_file = PRED_DIR / f"{date_str}_predictions.json"
    if not pred_file.exists():
        print(f"{date_str}: No file found")
        continue

    with open(pred_file) as f:
        preds = json.load(f)

    completed = [g for g in preds if g.get("actual")]
    statuses = {}
    for g in preds:
        s = g.get("status", "unknown")
        statuses[s] = statuses.get(s, 0) + 1

    print(f"\n{date_str}: {len(completed)}/{len(preds)} have actuals")
    print(f"  Status breakdown: {statuses}")

    # Show a few sample games
    print(f"  Sample games:")
    for g in preds[:3]:
        print(f"    {g.get('away_team','?')} @ {g.get('home_team','?')} — status={g.get('status','?')} — actual={g.get('actual')}")