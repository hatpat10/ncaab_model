"""
check_results.py
Quick script to review today's predictions vs actuals.
Usage: python check_results.py
       python check_results.py --date 2026-03-10
"""
import json
import os
import argparse
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PRED_DIR = ROOT / "predictions"

parser = argparse.ArgumentParser()
parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"))
args = parser.parse_args()

date_str = args.date
pred_file = PRED_DIR / f"{date_str}_predictions.json"

if not pred_file.exists():
    print(f"No prediction file found for {date_str}")
    print(f"Looked in: {pred_file}")
    exit()

with open(pred_file) as f:
    preds = json.load(f)

print(f"\n{'='*70}")
print(f"  RESULTS CHECK — {date_str}  ({len(preds)} games)")
print(f"{'='*70}")

spread_results = []
total_results  = []
bet_results    = []

for p in preds:
    actual    = p.get("actual", {})
    edge      = p.get("edge", {})
    pred_vals = p.get("predictions", {})
    bets      = p.get("bets", [])

    pred_margin  = pred_vals.get("predicted_margin")
    pred_total   = pred_vals.get("predicted_total")
    vegas_spread = edge.get("vegas_spread")
    vegas_total  = edge.get("vegas_total")
    spread_edge  = edge.get("spread_edge")
    total_edge   = edge.get("total_edge")

    actual_margin = actual.get("margin")
    actual_total  = actual.get("total")

    # Compute margin/total from raw scores if not pre-computed
    if actual_margin is None and actual.get("home_score") is not None and actual.get("away_score") is not None:
        actual_margin = actual["home_score"] - actual["away_score"]
        actual_total  = actual["home_score"] + actual["away_score"]

    completed = actual_margin is not None

    away = p["away_team"]
    home = p["home_team"]
    tier = p.get("game_tier", "?")
    status = "FINAL" if completed else "scheduled"

    print(f"\n  {away} @ {home}  [{status}] [{tier}]")

    # Spread
    if pred_margin is not None:
        vs = f"Vegas: {vegas_spread:+.1f}" if vegas_spread is not None else "no line"
        print(f"    Spread  — Model: {pred_margin:+.1f}  |  {vs}", end="")
        if completed:
            spread_err = abs(pred_margin - actual_margin)
            print(f"  |  Actual: {actual_margin:+.1f}  |  Error: {spread_err:.1f} pts", end="")
            spread_results.append(spread_err)

            # ATS result
            if vegas_spread is not None:
                # Home covers if actual_margin > -vegas_spread
                home_covered = actual_margin > -vegas_spread
                model_said_home = pred_margin > -vegas_spread
                ats_correct = (home_covered == model_said_home)
                print(f"  |  ATS: {'✓' if ats_correct else '✗'}", end="")
        print()

    # Total
    if pred_total is not None:
        vt = f"Vegas: {vegas_total:.1f}" if vegas_total is not None else "no line"
        print(f"    Total   — Model: {pred_total:.1f}  |  {vt}", end="")
        if completed and actual_total is not None:
            total_err = abs(pred_total - actual_total)
            ou_correct = None
            if vegas_total is not None:
                went_over = actual_total > vegas_total
                model_said_over = total_edge > 0 if total_edge else False
                ou_correct = (went_over == model_said_over)
            print(f"  |  Actual: {actual_total}  |  Error: {total_err:.1f}", end="")
            if ou_correct is not None:
                print(f"  |  O/U: {'✓' if ou_correct else '✗'}", end="")
            total_results.append(total_err)
        print()

    # Flagged bets
    if bets:
        for b in bets:
            market = b["market"]
            lean   = b["lean"]
            conf   = b["confidence"]
            epts   = b.get("edge_pts", 0)
            result_str = ""

            if completed:
                if market == "SPREAD" and vegas_spread is not None:
                    home_covered = actual_margin > -vegas_spread
                    bet_home     = (lean == "HOME")
                    won = (bet_home == home_covered)
                    result_str = "  WON ✓" if won else "  LOST ✗"
                    bet_results.append({"game": f"{away} @ {home}", "market": market,
                                        "lean": lean, "conf": conf, "won": won,
                                        "edge": epts})
                elif market == "TOTAL" and vegas_total is not None:
                    went_over = actual_total > vegas_total
                    bet_over  = (lean == "OVER")
                    won = (bet_over == went_over)
                    result_str = "  WON ✓" if won else "  LOST ✗"
                    bet_results.append({"game": f"{away} @ {home}", "market": market,
                                        "lean": lean, "conf": conf, "won": won,
                                        "edge": epts})

            print(f"    ★ BET: {market} {lean} [{conf}] — {epts:.1f} pt edge{result_str}")

# ── Summary ──────────────────────────────────────────────────────────────────
completed_count = sum(1 for p in preds if p.get("actual", {}).get("margin") is not None)
print(f"\n{'='*70}")
print(f"  SUMMARY — {completed_count}/{len(preds)} games completed")

if spread_results:
    print(f"  Spread MAE today:  {sum(spread_results)/len(spread_results):.1f} pts  ({len(spread_results)} games)")
if total_results:
    print(f"  Totals MAE today:  {sum(total_results)/len(total_results):.1f} pts  ({len(total_results)} games)")

if bet_results:
    wins = sum(1 for b in bet_results if b["won"])
    print(f"\n  Flagged bets: {wins}/{len(bet_results)} won")
    for b in bet_results:
        print(f"    [{b['conf']}] {b['market']} {b['lean']:5} — {b['game']}  {'✓' if b['won'] else '✗'}")
elif completed_count > 0:
    print(f"\n  No flagged bets today")

print(f"{'='*70}\n")