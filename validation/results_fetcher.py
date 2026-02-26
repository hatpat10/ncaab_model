"""
validation/results_fetcher.py
Fetches actual game results from ESPN and fills in predictions database.

Usage:
    python -m validation.results_fetcher --date 2026-02-24
    python -m validation.results_fetcher --start 2026-02-01 --end 2026-02-23
    python -m validation.results_fetcher --start 2026-02-01 --end 2026-02-23 --dry-run
    python -m validation.results_fetcher --coverage
"""

import requests
import sqlite3
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DB_PATH = Path("data/ncaab.db")
ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"


def fetch_espn_results(date_str: str) -> dict:
    """
    Fetch completed game results from ESPN for a given date.
    Returns dict keyed by ESPN game_id (string) -> result dict.
    date_str: YYYY-MM-DD
    """
    espn_date = date_str.replace("-", "")
    url = f"{ESPN_SCOREBOARD}?dates={espn_date}&limit=200"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log.error(f"ESPN fetch failed for {date_str}: {e}")
        return {}

    results = {}
    total = 0
    completed = 0
    for event in data.get("events", []):
        total += 1
        status = event.get("status", {}).get("type", {})
        if not status.get("completed"):
            continue
        completed += 1

        game_id = str(event.get("id", ""))
        comps = event.get("competitions", [{}])[0]
        competitors = comps.get("competitors", [])
        if len(competitors) != 2:
            continue

        home = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away = next((c for c in competitors if c.get("homeAway") == "away"), None)
        if not home or not away:
            continue

        home_score = int(home.get("score", 0) or 0)
        away_score = int(away.get("score", 0) or 0)

        results[game_id] = {
            "game_id": game_id,
            "home_team": home.get("team", {}).get("displayName", ""),
            "away_team": away.get("team", {}).get("displayName", ""),
            "home_score": home_score,
            "away_score": away_score,
            "actual_margin": home_score - away_score,
            "actual_total": home_score + away_score,
        }

    log.info(f"{date_str}: {completed}/{total} games completed on ESPN")
    return results


def fill_results(date_str: str, dry_run: bool = False) -> dict:
    """Fetch ESPN results for date and update predictions table in ncaab.db."""
    if not DB_PATH.exists():
        log.error(f"Database not found: {DB_PATH.resolve()}")
        log.error("Make sure you're running from D:\\ncaab_model")
        return {"filled": 0, "not_found": 0, "total_espn": 0}

    espn_results = fetch_espn_results(date_str)
    if not espn_results:
        return {"filled": 0, "not_found": 0, "total_espn": 0}

    filled = 0
    not_found = 0
    already_filled = 0

    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()

        # Get all predictions for this date
        cur.execute("""
            SELECT game_id, home_team, away_team, actual_margin
            FROM predictions
            WHERE date = ?
        """, (date_str,))
        pred_rows = cur.fetchall()

        if not pred_rows:
            log.warning(f"No predictions found for {date_str} in DB")
            cur.execute("SELECT DISTINCT date FROM predictions ORDER BY date DESC LIMIT 10")
            dates = [r[0] for r in cur.fetchall()]
            log.info(f"Prediction dates available in DB: {dates}")
            return {"filled": 0, "not_found": 0, "total_espn": len(espn_results)}

        for game_id, home_team, away_team, existing_margin in pred_rows:
            if existing_margin is not None:
                already_filled += 1
                continue

            if game_id in espn_results:
                result = espn_results[game_id]
                if not dry_run:
                    cur.execute("""
                        UPDATE predictions
                        SET actual_margin = ?, actual_total = ?
                        WHERE game_id = ? AND date = ?
                    """, (result["actual_margin"], result["actual_total"],
                          game_id, date_str))
                filled += 1
                log.info(f"{'[DRY] ' if dry_run else ''}Filled {away_team} @ {home_team}: "
                         f"margin={result['actual_margin']:+.0f}, total={result['actual_total']}")
            else:
                log.debug(f"No ESPN result for game_id={game_id} ({away_team} @ {home_team})")
                not_found += 1

        if not dry_run:
            con.commit()

    summary = {
        "date": date_str,
        "filled": filled,
        "not_found": not_found,
        "already_filled": already_filled,
        "total_espn": len(espn_results),
        "total_predictions": len(pred_rows),
    }
    log.info(f"Results for {date_str}: {summary}")
    return summary


def fill_date_range(start_date: str, end_date: str, dry_run: bool = False) -> dict:
    """Fill results for a range of dates."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    totals = {"filled": 0, "not_found": 0, "already_filled": 0, "total_espn": 0}
    current = start
    while current <= end:
        result = fill_results(current.strftime("%Y-%m-%d"), dry_run=dry_run)
        for k in totals:
            totals[k] += result.get(k, 0)
        current += timedelta(days=1)
    log.info(f"Range {start_date} -> {end_date} complete: {totals}")
    return totals


def check_coverage() -> None:
    """Print a coverage report: how many predictions have actuals filled."""
    if not DB_PATH.exists():
        log.error(f"Database not found: {DB_PATH.resolve()}")
        return
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute("""
            SELECT
                date,
                COUNT(*) as total,
                SUM(CASE WHEN actual_margin IS NOT NULL THEN 1 ELSE 0 END) as graded,
                SUM(CASE WHEN actual_margin IS NULL THEN 1 ELSE 0 END) as pending
            FROM predictions
            GROUP BY date
            ORDER BY date DESC
            LIMIT 30
        """)
        rows = cur.fetchall()
        print(f"\n{'Date':<14} {'Total':>6} {'Graded':>8} {'Pending':>9}")
        print("-" * 40)
        for date, total, graded, pending in rows:
            print(f"{date:<14} {total:>6} {graded:>8} {pending:>9}")

        # Overall summary
        cur.execute("""
            SELECT COUNT(*), 
                   SUM(CASE WHEN actual_margin IS NOT NULL THEN 1 ELSE 0 END)
            FROM predictions
        """)
        total, graded = cur.fetchone()
        print(f"\nTotal: {graded}/{total} predictions graded ({graded/total*100:.1f}%)\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", help="Single date YYYY-MM-DD")
    parser.add_argument("--start", help="Start date for range YYYY-MM-DD")
    parser.add_argument("--end", help="End date for range YYYY-MM-DD")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing to DB")
    parser.add_argument("--coverage", action="store_true", help="Print coverage report and exit")
    args = parser.parse_args()

    if args.coverage:
        check_coverage()
    elif args.start and args.end:
        fill_date_range(args.start, args.end, dry_run=args.dry_run)
    elif args.date:
        fill_results(args.date, dry_run=args.dry_run)
    else:
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        log.info(f"No date specified, using yesterday: {yesterday}")
        fill_results(yesterday, dry_run=args.dry_run)