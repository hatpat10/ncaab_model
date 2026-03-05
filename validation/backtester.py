import sqlite3, argparse, json, logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
DB_PATH = Path("data/ncaab.db")
BREAKEVEN = 100 / 210

def load_graded_predictions(min_date=None):
    if not DB_PATH.exists():
        raise FileNotFoundError("DB not found: " + str(DB_PATH.resolve()))
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        query = """SELECT * FROM predictions 
           WHERE actual_margin IS NOT NULL 
           AND actual_total IS NOT NULL 
           AND spread_edge IS NOT NULL 
           AND total_edge IS NOT NULL
           """
        params = []
        if min_date:
            query += " AND date >= ?"
            params.append(min_date)
        cur.execute(query + " ORDER BY date", params)
        return [dict(r) for r in cur.fetchall()]

def bucket_analysis(rows, edge_key, win_fn, buckets):
    results = []
    for i in range(len(buckets) - 1):
        lo, hi = buckets[i], buckets[i + 1]
        subset = [r for r in rows if lo <= abs(r.get(edge_key, 0) or 0) < hi]
        n = len(subset)
        if n == 0:
            continue
        wins = sum(1 for r in subset if win_fn(r))
        win_rate = wins / n
        roi = ((wins * (100 / 1.1)) - ((n - wins) * 100)) / (n * 100) * 100
        results.append({"edge_min": lo, "edge_max": "+" if hi >= 999 else hi,
                        "n_games": n, "wins": wins, "win_rate": round(win_rate, 4),
                        "roi_pct": round(roi, 2), "profitable": win_rate > BREAKEVEN})
    return results

def spread_win(row):
    edge = row.get("spread_edge", 0) or 0
    actual = row.get("actual_margin", 0) or 0
    vegas = row.get("vegas_spread", 0) or 0
    home_covered = actual > -vegas
    return home_covered if edge > 0 else not home_covered

def total_win(row):
    edge = row.get("total_edge", 0) or 0
    actual = row.get("actual_total", 0) or 0
    vegas = row.get("vegas_total", 0) or 0
    return actual < vegas if edge < 0 else actual > vegas

def print_bucket_table(analysis):
    print("  " + "-" * 58)
    print("  {:>8}  {:>8}  {:>5}  {:>5}  {:>6}  {:>7}  {}".format(
        "Edge Min", "Edge Max", "N", "Wins", "Win%", "ROI%", "Profitable"))
    print("  " + "-" * 58)
    for b in analysis:
        flag = "YES ***" if b["profitable"] else ""
        print("  {:>8.1f}  {:>8}  {:>5}  {:>5}  {:>5.1f}%  {:>+7.1f}%  {}".format(
            b["edge_min"], b["edge_max"], b["n_games"], b["wins"],
            b["win_rate"] * 100, b["roi_pct"], flag))

def run_backtest(min_date=None, output_path=None):
    rows = load_graded_predictions(min_date)
    if not rows:
        print("\nNo graded predictions yet.")
        return {}
    spread_rows = [r for r in rows if r.get("vegas_spread") is not None]
    spread_errors = [abs((r.get("actual_margin") or 0) - (r.get("predicted_margin") or 0)) for r in spread_rows]
    spread_mae = sum(spread_errors) / len(spread_errors) if spread_errors else 0
    overall_ats = sum(1 for r in spread_rows if spread_win(r)) / len(spread_rows) if spread_rows else 0
    spread_analysis = bucket_analysis(spread_rows, "spread_edge", spread_win, [0, 2, 4, 6, 8, 10, 15, 999])
    total_rows = [r for r in rows if r.get("vegas_total") is not None and abs(r.get("total_edge", 0) or 0) <= 12]
    total_errors = [abs((r.get("actual_total") or 0) - (r.get("predicted_total") or 0)) for r in total_rows]
    total_mae = sum(total_errors) / len(total_errors) if total_errors else 0
    total_analysis = bucket_analysis(total_rows, "total_edge", total_win, [0, 3, 6, 9, 12, 999])
    profitable = [b for b in spread_analysis if b["profitable"] and b["n_games"] >= 5]
    recommended = profitable[0]["edge_min"] if profitable else 6.0
    dates = sorted(set(r["date"] for r in rows))
    print("\n" + "=" * 62)
    print("PHASE 7 BACKTEST REPORT")
    print("=" * 62)
    print("Games graded : {}".format(len(rows)))
    print("Date range   : {} -> {}".format(dates[0], dates[-1]))
    print("\nSPREAD MODEL")
    print("  MAE         : {:.2f} pts".format(spread_mae))
    print("  Overall ATS : {:.1f}%  (need >52.4% to profit)".format(overall_ats * 100))
    print_bucket_table(spread_analysis)
    print("\nTOTALS MODEL  (edges >12 suppressed)")
    print("  MAE         : {:.2f} pts".format(total_mae))
    print_bucket_table(total_analysis)
    print("\nRECOMMENDED: Spread HIGH flag at edge >= {:.1f} pts".format(recommended))
    print("=" * 62 + "\n")
    report = {"generated_at": datetime.now().isoformat(), "games_graded": len(rows),
              "date_range": {"min": dates[0], "max": dates[-1]},
              "spread": {"mae": round(spread_mae, 3), "overall_ats": round(overall_ats, 4),
                         "by_bucket": spread_analysis, "recommended_threshold": recommended},
              "totals": {"mae": round(total_mae, 3), "by_bucket": total_analysis}}
    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(report, f, indent=2)
        log.info("Report saved to %s", p)
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-date")
    parser.add_argument("--output")
    args = parser.parse_args()
    run_backtest(min_date=args.min_date, output_path=args.output)

