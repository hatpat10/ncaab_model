"""
run_daily.py
One-command daily routine. Run this every evening before games tip.

Usage:
    python run_daily.py                    # grades yesterday, predicts tomorrow
    python run_daily.py --date 2026-02-26  # predict specific date
    python run_daily.py --grade-only       # just grade yesterday
    python run_daily.py --predict-only     # just run pipeline for tomorrow
"""

import subprocess
import sys
import argparse
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def run(cmd: list[str]) -> int:
    log.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        log.warning("Command exited with code %d", result.returncode)
    return result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", help="Predict date YYYY-MM-DD (default: tomorrow)")
    parser.add_argument("--grade-only", action="store_true")
    parser.add_argument("--predict-only", action="store_true")
    args = parser.parse_args()

    today = datetime.now()
    yesterday = (today - timedelta(days=1)).strftime("%Y-%m-%d")
    tomorrow = (today + timedelta(days=1)).strftime("%Y-%m-%d")
    predict_date = args.date or tomorrow

    print("\n" + "=" * 60)
    print("  NCAAB DAILY ROUTINE — {}".format(today.strftime("%Y-%m-%d %H:%M")))
    print("=" * 60)

    # ── Step 1: Grade yesterday ──────────────────────────────────────
    if not args.predict_only:
        print("\n[1/3] Grading yesterday's games ({})...".format(yesterday))
        run([sys.executable, "-m", "validation.results_fetcher", "--date", yesterday])

        print("\n[2/3] Coverage report:")
        run([sys.executable, "-m", "validation.results_fetcher", "--coverage"])

        print("\n[3/3] Running backtest with latest data...")
        run([sys.executable, "-m", "validation.backtester",
             "--output", "reports/backtest_latest.json"])

    # ── Step 2: Predict tomorrow ─────────────────────────────────────
    if not args.grade_only:
        print("\n[4/4] Generating predictions for {}...".format(predict_date))
        run([sys.executable, "-m", "predictions.daily_pipeline", "--date", predict_date])

    print("\n" + "=" * 60)
    print("  Done. Check reports/backtest_latest.json for updated stats.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()