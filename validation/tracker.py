"""
validation/tracker.py
=====================
NCAAB ML - Phase 7: Live Performance Tracker

Tracks cumulative blind-prediction performance with:
  - Confidence intervals (Wilson score) so you know when to trust results
  - Blind integrity check (predicted_at must be before game tip)
  - Daily summary report
  - Running ROI curve by edge bucket

Usage:
    python -m validation.tracker                    # full report
    python -m validation.tracker --daily            # yesterday's summary
    python -m validation.tracker --since 2026-02-25 # from date forward
    python -m validation.tracker --integrity        # check blind validity
"""

import sqlite3
import argparse
import json
import logging
import math
from datetime import datetime, timedelta
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DB_PATH      = Path("data/ncaab.db")
REPORTS_DIR  = Path("reports")
BREAKEVEN    = 100 / 210          # 47.6% — minimum ATS% to profit at -110
MIN_SAMPLE   = 30                 # minimum games before trusting a bucket
Z_95         = 1.96               # 95% confidence interval


# ── Wilson score confidence interval for a proportion ────────────────────────

def wilson_ci(wins, n, z=Z_95):
    """Return (lower, upper) 95% CI for a win rate."""
    if n == 0:
        return 0.0, 1.0
    p = wins / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def roi(wins, n):
    """Standard -110 ROI formula."""
    if n == 0:
        return 0.0
    return ((wins * (100 / 1.1)) - ((n - wins) * 100)) / (n * 100) * 100


# ── DB helpers ────────────────────────────────────────────────────────────────

def load_predictions(since=None, blind_only=True):
    """
    Load graded predictions.
    blind_only=True: only include rows where predicted_at < game tip time.
    Falls back gracefully if predicted_at column doesn't exist.
    """
    if not DB_PATH.exists():
        raise FileNotFoundError(f"DB not found: {DB_PATH.resolve()}")

    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row

        # Check if predicted_at column exists
        cols = [r[1] for r in con.execute("PRAGMA table_info(predictions)").fetchall()]
        has_predicted_at = "predicted_at" in cols or "created_at" in cols
        has_tip_time     = "tip_time" in cols

        query = "SELECT * FROM predictions WHERE actual_margin IS NOT NULL"
        params = []

        if since:
            query += " AND date >= ?"
            params.append(since)

        if blind_only and has_predicted_at and has_tip_time:
            query += " AND created_at < tip_time"
        elif blind_only and has_predicted_at:
            query += " AND DATE(created_at) < date"

        query += " ORDER BY date"
        rows = [dict(r) for r in con.execute(query, params).fetchall()]

    return rows


def check_blind_integrity():
    """Report on prediction timing vs game time."""
    if not DB_PATH.exists():
        print("DB not found.")
        return

    with sqlite3.connect(DB_PATH) as con:
        cols = [r[1] for r in con.execute("PRAGMA table_info(predictions)").fetchall()]

    print("\n── Blind Prediction Integrity ──────────────────────────────")
    if "predicted_at" not in cols:
        print("  WARNING: 'predicted_at' column not in predictions table.")
        print("  Add it to your daily pipeline to enable integrity checks.")
        print("  For now, all graded predictions are included regardless of timing.")
    else:
        with sqlite3.connect(DB_PATH) as con:
            row = con.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN predicted_at < tip_time THEN 1 ELSE 0 END) as blind,
                    SUM(CASE WHEN predicted_at >= tip_time THEN 1 ELSE 0 END) as leaked,
                    MIN(date) as first_date,
                    MAX(date) as last_date
                FROM predictions
                WHERE actual_margin IS NOT NULL AND predicted_at IS NOT NULL
            """).fetchone()
            total, blind, leaked, first, last = row
            print(f"  Date range : {first} to {last}")
            print(f"  Total      : {total}")
            print(f"  Blind      : {blind}  OK")
            print(f"  Leaked     : {leaked}  {'EXCLUDED from analysis' if leaked else 'none - good'}")
    print()


# ── Core analysis ─────────────────────────────────────────────────────────────

def spread_win(row):
    edge   = row.get("spread_edge") or 0
    actual = row.get("actual_margin") or 0
    vegas  = row.get("vegas_spread") or 0
    home_covered = actual > -vegas
    return home_covered if edge > 0 else not home_covered


def total_win(row):
    edge   = row.get("total_edge") or 0
    actual = row.get("actual_total") or 0
    vegas  = row.get("vegas_total") or 0
    return actual < vegas if edge < 0 else actual > vegas


SPREAD_BUCKETS = [0, 2, 4, 6, 8, 10, 15, 999]
TOTAL_BUCKETS  = [0, 3, 6, 9, 12, 999]


def bucket_stats(rows, edge_key, win_fn, buckets, max_edge=None):
    results = []
    for lo, hi in zip(buckets, buckets[1:]):
        subset = [
            r for r in rows
            if r.get(edge_key) is not None
            and lo <= abs(r[edge_key]) < hi
            and (max_edge is None or abs(r[edge_key]) <= max_edge)
        ]
        n = len(subset)
        if n == 0:
            continue
        wins = sum(1 for r in subset if win_fn(r))
        lo_ci, hi_ci = wilson_ci(wins, n)
        results.append({
            "edge_min":   lo,
            "edge_max":   hi if hi < 999 else "+",
            "n":          n,
            "wins":       wins,
            "win_pct":    round(wins / n, 4),
            "roi":        round(roi(wins, n), 2),
            "ci_low":     round(lo_ci, 4),
            "ci_high":    round(hi_ci, 4),
            "reliable":   n >= MIN_SAMPLE,
            "profitable": lo_ci > BREAKEVEN,
        })
    return results


def cumulative_stats(rows, edge_key, win_fn, min_edge=0):
    subset = [r for r in rows if r.get(edge_key) is not None and abs(r[edge_key]) >= min_edge]
    n = len(subset)
    if n == 0:
        return None
    wins = sum(1 for r in subset if win_fn(r))
    lo_ci, hi_ci = wilson_ci(wins, n)
    return {
        "n": n, "wins": wins,
        "win_pct": round(wins / n, 4),
        "roi": round(roi(wins, n), 2),
        "ci_low": round(lo_ci, 4),
        "ci_high": round(hi_ci, 4),
        "reliable": n >= MIN_SAMPLE,
        "profitable": lo_ci > BREAKEVEN,
    }


def games_needed_to_confirm(current_wins, current_n, target=BREAKEVEN, z=Z_95):
    """Estimate additional games needed for lower CI to cross breakeven."""
    if current_n == 0:
        return None
    p = current_wins / current_n
    if p <= target:
        return None  # win rate too low to ever confirm
    for n in range(current_n, current_n + 3000):
        w = round(p * n)
        lo, _ = wilson_ci(w, n, z)
        if lo > target:
            return n - current_n
    return None


# ── Printing ──────────────────────────────────────────────────────────────────

def print_bucket_table(stats, label):
    print(f"\n  {label}")
    print(f"  {'Edge':>8}  {'N':>5}  {'Wins':>5}  {'Win%':>6}  {'ROI%':>7}  {'95% CI':>17}")
    print("  " + "-" * 62)
    for s in stats:
        hi_str   = "+" if s['edge_max'] == "+" else f"{s['edge_max']:.0f}"
        edge_str = f"{s['edge_min']:.0f}-{hi_str}"
        ci_str   = f"[{s['ci_low']:.1%}, {s['ci_high']:.1%}]"
        flag = ""
        if not s["reliable"]:
            flag = "  (small sample)"
        elif s["profitable"]:
            flag = "  CONFIRMED EDGE"
        print(f"  {edge_str:>8}  {s['n']:>5}  {s['wins']:>5}  {s['win_pct']:>5.1%}  "
              f"{s['roi']:>+7.1f}%  {ci_str:>17}{flag}")


def print_cumulative_line(stats, label):
    if not stats or stats["n"] == 0:
        return
    ci_str = f"[{stats['ci_low']:.1%}, {stats['ci_high']:.1%}]"
    if stats["profitable"]:
        verdict = "CONFIRMED EDGE"
    elif not stats["reliable"]:
        verdict = f"need {MIN_SAMPLE - stats['n']} more games"
    else:
        needed = games_needed_to_confirm(stats["wins"], stats["n"])
        verdict = f"need ~{needed} more games to confirm" if needed else "win rate too low"
    print(f"  {label:<12} {stats['n']:>5} games  {stats['win_pct']:.1%} ATS  "
          f"ROI {stats['roi']:>+6.1f}%  CI {ci_str}  -> {verdict}")


# ── Full report ───────────────────────────────────────────────────────────────

def full_report(since=None, save=True):
    rows = load_predictions(since=since, blind_only=False)

    spread_rows = [r for r in rows if r.get("spread_edge") is not None
                   and r.get("vegas_spread") is not None]
    total_rows  = [r for r in rows if r.get("total_edge") is not None
                   and r.get("vegas_total") is not None
                   and abs(r.get("total_edge") or 0) <= 12]

    now       = datetime.now().strftime("%Y-%m-%d %H:%M")
    since_str = since or "all time"

    print("\n" + "=" * 70)
    print(f"  NCAAB BLIND PREDICTION TRACKER  --  {now}")
    print(f"  Period: {since_str}")
    if since is None:
        print("  NOTE: Run with --since 2026-02-25 for truly blind predictions only")
    print("=" * 70)

    # ── Spread ────────────────────────────────────────────────────────────────
    if spread_rows:
        dates = sorted(set(r["date"] for r in spread_rows))
        print(f"\nSPREAD  ({len(spread_rows)} games with edges, {dates[0]} to {dates[-1]})")
        sp_stats = bucket_stats(spread_rows, "spread_edge", spread_win, SPREAD_BUCKETS)
        print_bucket_table(sp_stats, "By Edge Bucket")
        print("\n  Cumulative (edge >= threshold):")
        for min_edge in [0, 4, 6, 8, 10]:
            s = cumulative_stats(spread_rows, "spread_edge", spread_win, min_edge)
            if s and s["n"] > 0:
                print_cumulative_line(s, f"edge >= {min_edge}")
    else:
        print("\nSPREAD: no graded predictions with spread edges.")

    # ── Totals ────────────────────────────────────────────────────────────────
    if total_rows:
        dates = sorted(set(r["date"] for r in total_rows))
        print(f"\nTOTALS  ({len(total_rows)} games with edges, {dates[0]} to {dates[-1]})")
        tot_stats = bucket_stats(total_rows, "total_edge", total_win, TOTAL_BUCKETS, max_edge=12)
        print_bucket_table(tot_stats, "By Edge Bucket")
        print("\n  Cumulative (edge >= threshold):")
        for min_edge in [0, 3, 6]:
            s = cumulative_stats(total_rows, "total_edge", total_win, min_edge)
            if s and s["n"] > 0:
                print_cumulative_line(s, f"edge >= {min_edge}")
    else:
        print("\nTOTALS: no graded predictions with total edges.")

    print(f"\n  Breakeven: {BREAKEVEN:.1%} at -110 juice")
    print(f"  'CONFIRMED EDGE' = lower 95% CI bound > breakeven (statistically profitable)")
    print(f"  Reliable bucket = n >= {MIN_SAMPLE} games")
    print("=" * 70 + "\n")

    if save:
        report = {
            "generated_at":   now,
            "since":          since_str,
            "spread_games":   len(spread_rows),
            "total_games":    len(total_rows),
            "spread_buckets": bucket_stats(spread_rows, "spread_edge", spread_win, SPREAD_BUCKETS) if spread_rows else [],
            "total_buckets":  bucket_stats(total_rows, "total_edge", total_win, TOTAL_BUCKETS, max_edge=12) if total_rows else [],
        }
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        out = REPORTS_DIR / "tracker_latest.json"
        with open(out, "w") as f:
            json.dump(report, f, indent=2, default=str)
        log.info("Saved to %s", out)


# ── Daily summary ─────────────────────────────────────────────────────────────

def daily_summary(date_str=None):
    if not date_str:
        date_str = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        rows = [dict(r) for r in con.execute(
            "SELECT * FROM predictions WHERE date = ? AND actual_margin IS NOT NULL",
            (date_str,)
        ).fetchall()]

    if not rows:
        print(f"\nNo graded predictions for {date_str}.")
        return

    spread_rows = [r for r in rows if r.get("spread_edge") is not None
                   and r.get("vegas_spread") is not None]
    total_rows  = [r for r in rows if r.get("total_edge") is not None
                   and r.get("vegas_total") is not None]

    print(f"\n── Daily Summary: {date_str} " + "─" * 40)
    print(f"  Total predictions : {len(rows)}")
    print(f"  With spread edge  : {len(spread_rows)}")
    print(f"  With total edge   : {len(total_rows)}")

    if spread_rows:
        wins = sum(1 for r in spread_rows if spread_win(r))
        mae  = sum(abs((r.get("actual_margin") or 0) - (r.get("predicted_margin") or 0))
                   for r in spread_rows) / len(spread_rows)
        print(f"\n  Spread ATS  : {wins}/{len(spread_rows)} ({wins/len(spread_rows):.1%})  "
              f"ROI: {roi(wins, len(spread_rows)):+.1f}%  MAE: {mae:.1f} pts")

        high_edge = sorted(
            [r for r in spread_rows if abs(r.get("spread_edge") or 0) >= 6],
            key=lambda x: abs(x.get("spread_edge") or 0), reverse=True
        )
        if high_edge:
            print(f"\n  High-edge calls (spread >= 6):")
            for r in high_edge:
                result = "WIN" if spread_win(r) else "LOSS"
                home = r.get("home_team", "?")
                away = r.get("away_team", "?")
                print(f"    {home} vs {away}  edge={r['spread_edge']:+.1f}  "
                      f"actual={r.get('actual_margin', 0):+.0f}  {result}")

    if total_rows:
        wins = sum(1 for r in total_rows if total_win(r))
        print(f"\n  Totals O/U  : {wins}/{len(total_rows)} ({wins/len(total_rows):.1%})  "
              f"ROI: {roi(wins, len(total_rows)):+.1f}%")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NCAAB Blind Prediction Tracker")
    parser.add_argument("--since",     help="Only predictions from this date forward (YYYY-MM-DD)")
    parser.add_argument("--daily",     action="store_true", help="Show yesterday's daily summary")
    parser.add_argument("--date",      help="Show daily summary for a specific date")
    parser.add_argument("--integrity", action="store_true", help="Check blind prediction integrity")
    parser.add_argument("--no-save",   action="store_true", help="Don't save JSON report")
    args = parser.parse_args()

    if args.integrity:
        check_blind_integrity()
        return

    if args.daily or args.date:
        daily_summary(args.date)
        return

    full_report(since=args.since, save=not args.no_save)


if __name__ == "__main__":
    main()