"""
validation/daily_grader.py
Automated daily job: fetch yesterday's results, update DB, append to running backtest log.
Run this every morning after games complete.

Usage:
    python -m validation.daily_grader
    python -m validation.daily_grader --date 2025-02-24
"""

import argparse
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

from validation.results_fetcher import fill_results
from validation.backtester import run_backtest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

HISTORY_PATH = Path("reports/backtest_history.jsonl")


def grade_date(date_str: str, run_full_backtest: bool = True) -> dict:
    """Fetch results for a date, then recompute rolling backtest stats."""
    log.info(f"Grading games for {date_str}...")

    fill_summary = fill_results(date_str)
    log.info(f"Filled {fill_summary['filled']}/{fill_summary['total_espn']} games")

    if run_full_backtest:
        log.info("Running updated backtest...")
        report = run_backtest()

        # Append to history log
        HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(HISTORY_PATH, "a") as f:
            entry = {"date": date_str, "fill_summary": fill_summary, "backtest": report}
            f.write(json.dumps(entry) + "\n")

        # Save latest report
        out_path = Path("reports/backtest_latest.json")
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        log.info(f"Latest backtest saved to {out_path}")

        return report

    return {"fill_summary": fill_summary}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", help="Date to grade (default: yesterday)")
    parser.add_argument("--no-backtest", action="store_true",
                        help="Skip full backtest recompute")
    args = parser.parse_args()

    date = args.date or (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    grade_date(date, run_full_backtest=not args.no_backtest)