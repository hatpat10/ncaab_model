"""
validation/results_fetcher.py
Fetches actual game results from ESPN and fills in predictions database.

Uses the ESPN game summary endpoint (per game_id) instead of the scoreboard,
so results are available indefinitely regardless of how old the game is.

Usage:
    python -m validation.results_fetcher --date 2026-02-24
    python -m validation.results_fetcher --start 2026-02-01 --end 2026-02-23
    python -m validation.results_fetcher --coverage
    python -m validation.results_fetcher --backfill        # grade all pending
"""
import requests
import sqlite3
import argparse
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DB_PATH      = Path("data/ncaab.db")
ESPN_SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/summary"
ESPN_BOARD   = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"


# ── Per-game fetch (never expires) ───────────────────────────────────────────

def fetch_game_result(game_id: str) -> dict | None:
    """
    Fetch result for a single game by ESPN game_id.
    Returns result dict or None if not completed / not found.
    """
    try:
        r = requests.get(ESPN_SUMMARY, params={"event": game_id}, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log.debug("ESPN summary failed for %s: %s", game_id, e)
        return None

    # Pull from header > competitions
    header = data.get("header", {})
    comps  = header.get("competitions", [{}])
    if not comps:
        return None
    comp = comps[0]

    status = comp.get("status", {}).get("type", {})
    if not status.get("completed", False):
        return None   # game not finished yet

    competitors = comp.get("competitors", [])
    home = next((c for c in competitors if c.get("homeAway") == "home"), None)
    away = next((c for c in competitors if c.get("homeAway") == "away"), None)
    if not home or not away:
        return None

    try:
        home_score = int(home.get("score", 0) or 0)
        away_score = int(away.get("score", 0) or 0)
    except (ValueError, TypeError):
        return None

    if home_score == 0 and away_score == 0:
        return None  # scores not populated yet

    return {
        "game_id":       game_id,
        "home_score":    home_score,
        "away_score":    away_score,
        "actual_margin": home_score - away_score,
        "actual_total":  home_score + away_score,
    }


# ── Scoreboard fetch (fast, but limited window) ───────────────────────────────

def fetch_scoreboard_results(date_str: str) -> dict:
    """
    Fast path: fetch all completed games from ESPN scoreboard for a date.
    Only works for recent dates (ESPN drops old games from the feed).
    Returns dict keyed by game_id.
    """
    espn_date = date_str.replace("-", "")
    try:
        r = requests.get(ESPN_BOARD, params={"dates": espn_date, "limit": 300}, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log.warning("Scoreboard fetch failed for %s: %s", date_str, e)
        return {}

    results = {}
    for event in data.get("events", []):
        status = event.get("status", {}).get("type", {})
        if not status.get("completed"):
            continue
        game_id = str(event.get("id", ""))
        comps   = event.get("competitions", [{}])[0]
        comps_list = comps.get("competitors", [])
        home = next((c for c in comps_list if c.get("homeAway") == "home"), None)
        away = next((c for c in comps_list if c.get("homeAway") == "away"), None)
        if not home or not away:
            continue
        try:
            home_score = int(home.get("score", 0) or 0)
            away_score = int(away.get("score", 0) or 0)
        except (ValueError, TypeError):
            continue
        if home_score == 0 and away_score == 0:
            continue
        results[game_id] = {
            "game_id":       game_id,
            "home_score":    home_score,
            "away_score":    away_score,
            "actual_margin": home_score - away_score,
            "actual_total":  home_score + away_score,
        }

    log.info("Scoreboard: %d completed games for %s", len(results), date_str)
    return results


# ── Main fill function ────────────────────────────────────────────────────────

def fill_results(date_str: str, dry_run: bool = False, delay: float = 0.15) -> dict:
    """
    Grade all ungraded predictions for a given date.

    Strategy:
    1. Try scoreboard first (fast, covers recent games)
    2. For any remaining ungraded, fetch per game_id (always works)
    """
    if not DB_PATH.exists():
        log.error("Database not found: %s", DB_PATH.resolve())
        return {"filled": 0, "not_found": 0, "already_filled": 0, "total_predictions": 0}

    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute("""
            SELECT game_id, actual_margin
            FROM predictions
            WHERE date = ?
        """, (date_str,))
        rows = cur.fetchall()

    if not rows:
        log.warning("No predictions found for %s", date_str)
        return {"filled": 0, "not_found": 0, "already_filled": 0, "total_predictions": 0}

    already_filled = sum(1 for _, m in rows if m is not None)
    pending = [(gid,) for gid, m in rows if m is None]

    log.info("%s: %d predictions, %d already graded, %d pending",
             date_str, len(rows), already_filled, len(pending))

    if not pending:
        return {
            "filled": 0, "not_found": 0,
            "already_filled": already_filled,
            "total_predictions": len(rows),
        }

    # Step 1: scoreboard fast-path
    board_results = fetch_scoreboard_results(date_str)
    pending_ids   = {gid for (gid,) in pending}
    board_hits    = {gid: r for gid, r in board_results.items() if gid in pending_ids}

    # Step 2: per-game fetch for remaining
    remaining_ids = pending_ids - set(board_hits.keys())
    per_game      = {}
    if remaining_ids:
        log.info("Fetching %d games individually via ESPN summary...", len(remaining_ids))
        for gid in sorted(remaining_ids):
            result = fetch_game_result(gid)
            if result:
                per_game[gid] = result
            time.sleep(delay)  # be polite to ESPN API

    all_results = {**board_hits, **per_game}

    # Write to DB
    filled    = 0
    not_found = 0
    with sqlite3.connect(DB_PATH) as con:
        for (gid,) in pending:
            if gid in all_results:
                r = all_results[gid]
                if not dry_run:
                    con.execute("""
                        UPDATE predictions
                        SET actual_margin = ?, actual_total = ?
                        WHERE game_id = ?
                    """, (r["actual_margin"], r["actual_total"], gid))
                filled += 1
            else:
                not_found += 1
        if not dry_run:
            con.commit()

    log.info("Results for %s: filled=%d  not_found=%d  already=%d  total=%d",
             date_str, filled, not_found, already_filled, len(rows))

    return {
        "date":             date_str,
        "filled":           filled,
        "not_found":        not_found,
        "already_filled":   already_filled,
        "total_predictions": len(rows),
    }


def fill_date_range(start: str, end: str, dry_run: bool = False):
    """Grade all dates from start to end inclusive."""
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt   = datetime.strptime(end,   "%Y-%m-%d")
    current  = start_dt
    total_filled = 0
    while current <= end_dt:
        date_str = current.strftime("%Y-%m-%d")
        result   = fill_results(date_str, dry_run=dry_run)
        total_filled += result.get("filled", 0)
        current += timedelta(days=1)
    log.info("Total filled across range: %d", total_filled)


def backfill_all_pending(dry_run: bool = False):
    """Grade every ungraded prediction in the DB regardless of date."""
    if not DB_PATH.exists():
        log.error("Database not found")
        return
    with sqlite3.connect(DB_PATH) as con:
        rows = con.execute("""
            SELECT DISTINCT date FROM predictions
            WHERE actual_margin IS NULL
            ORDER BY date
        """).fetchall()
    dates = [r[0] for r in rows]
    log.info("Backfilling %d dates with pending predictions...", len(dates))
    total = 0
    for date_str in dates:
        result = fill_results(date_str, dry_run=dry_run)
        total += result.get("filled", 0)
    log.info("Backfill complete. Total filled: %d", total)


def print_coverage():
    """Print grading coverage report."""
    if not DB_PATH.exists():
        log.error("Database not found")
        return
    with sqlite3.connect(DB_PATH) as con:
        rows = con.execute("""
            SELECT date,
                   COUNT(*) as total,
                   SUM(CASE WHEN actual_margin IS NOT NULL THEN 1 ELSE 0 END) as graded
            FROM predictions
            GROUP BY date
            ORDER BY date DESC
        """).fetchall()

    total_all  = sum(r[1] for r in rows)
    graded_all = sum(r[2] for r in rows)

    print(f"\n{'Date':<16} {'Total':>7} {'Graded':>8} {'Pending':>9}")
    print("-" * 44)
    for date, total, graded in rows:
        pending = total - graded
        print(f"{date:<16} {total:>7} {graded:>8} {pending:>9}")
    print(f"\nTotal: {graded_all}/{total_all} predictions graded "
          f"({100*graded_all/total_all:.1f}%)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date",     help="Grade a specific date (YYYY-MM-DD)")
    parser.add_argument("--start",    help="Start of date range")
    parser.add_argument("--end",      help="End of date range")
    parser.add_argument("--backfill", action="store_true",
                        help="Grade all pending predictions in DB")
    parser.add_argument("--coverage", action="store_true",
                        help="Print coverage report")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Don't write to DB")
    args = parser.parse_args()

    if args.coverage:
        print_coverage()
    elif args.backfill:
        backfill_all_pending(dry_run=args.dry_run)
    elif args.date:
        fill_results(args.date, dry_run=args.dry_run)
    elif args.start and args.end:
        fill_date_range(args.start, args.end, dry_run=args.dry_run)
    else:
        # Default: grade yesterday
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        fill_results(yesterday, dry_run=args.dry_run)


if __name__ == "__main__":
    main()