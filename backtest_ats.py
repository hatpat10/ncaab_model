"""
backtest_ats.py — NCAAB ATS Backtester
Grades prediction JSON files vs actual scores.
Breaks down by confidence level, conference tier, and market type.
"""
import json, glob, os
from collections import defaultdict

CLEAN_DATE = "2026-03-04"  # Predictions before this used broken tier logic


def grade_bet(bet, edge, actual):
    if not actual:
        return None
    market = bet.get("market")
    lean   = bet.get("lean")
    margin = actual.get("margin", 0)
    total  = actual.get("total", 0)
    vs     = edge.get("vegas_spread")
    vt     = edge.get("vegas_total")

    if market == "SPREAD":
        if vs is None:
            return None
        cover = -vs  # how much home needs to win by
        if lean == "HOME":
            if margin > cover:    return "WIN"
            elif margin == cover: return "PUSH"
            else:                 return "LOSS"
        elif lean == "AWAY":
            if margin < cover:    return "WIN"
            elif margin == cover: return "PUSH"
            else:                 return "LOSS"

    elif market == "TOTAL":
        if vt is None:
            return None
        if lean == "OVER":
            if total > vt:    return "WIN"
            elif total == vt: return "PUSH"
            else:             return "LOSS"
        elif lean == "UNDER":
            if total < vt:    return "WIN"
            elif total == vt: return "PUSH"
            else:             return "LOSS"
    return None


def wp(w, l):
    return f"{w/(w+l)*100:.1f}%" if (w+l) > 0 else "N/A"


def summarize(rows, label=""):
    w = sum(1 for r in rows if r["result"] == "WIN")
    l = sum(1 for r in rows if r["result"] == "LOSS")
    p = sum(1 for r in rows if r["result"] == "PUSH")
    t = w + l + p
    tag = f"[{label}]" if label else ""
    print(f"  {tag:<18} W:{w:>4}  L:{l:>4}  P:{p:>3}  n={t:<5}  {wp(w,l):>8}")


def print_group(title, rows, key, order=None):
    groups = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
    keys = order if order else sorted(groups)
    print(f"\n  {title}")
    print(f"  {'-'*55}")
    for k in keys:
        if k in groups:
            summarize(groups[k], str(k))


def run_backtest():
    files = sorted(glob.glob("predictions/2026-*.json"))
    all_bets, clean_bets = [], []

    for f in files:
        date = os.path.basename(f)[:10]
        data = json.load(open(f, encoding="utf-8"))
        is_clean = date >= CLEAN_DATE

        for game in data:
            actual    = game.get("actual")
            edge      = game.get("edge", {})
            bets      = game.get("bets", [])
            game_tier = game.get("game_tier") or "unknown"
            if not actual or not bets:
                continue

            for bet in bets:
                result = grade_bet(bet, edge, actual)
                if result is None:
                    continue
                row = {
                    "date":       date,
                    "home":       game.get("home_team", ""),
                    "away":       game.get("away_team", ""),
                    "market":     bet.get("market"),
                    "lean":       bet.get("lean"),
                    "edge_pts":   bet.get("edge_pts"),
                    "confidence": bet.get("confidence", "UNKNOWN"),
                    "game_tier":  bet.get("game_tier") or game_tier,
                    "vegas_spread": edge.get("vegas_spread"),
                    "vegas_total":  edge.get("vegas_total"),
                    "actual_margin": actual.get("margin"),
                    "actual_total":  actual.get("total"),
                    "result":     result,
                }
                all_bets.append(row)
                if is_clean:
                    clean_bets.append(row)

    cb = clean_bets
    n  = len(cb)
    excl = len(all_bets) - n

    print(f"\n{'='*65}")
    print(f"  NCAAB ML MODEL — ATS BACKTEST")
    print(f"  Clean from: {CLEAN_DATE}  |  Clean bets: {n}  |  Excluded (pre-fix): {excl}")
    print(f"{'='*65}")

    # Overall
    print(f"\n{'='*65}")
    print(f"  OVERALL")
    print(f"{'='*65}")
    summarize(cb, "All")
    summarize([b for b in cb if b["market"] == "SPREAD"], "Spread")
    summarize([b for b in cb if b["market"] == "TOTAL"],  "Total")

    # ROI estimate
    w = sum(1 for b in cb if b["result"] == "WIN")
    l = sum(1 for b in cb if b["result"] == "LOSS")
    if w + l > 0:
        roi = (w * 0.909 - l) / (w + l) * 100
        print(f"\n  Est. ROI @ -110 vig:  {roi:+.1f}%  (breakeven = 52.4%)")

    # By confidence
    print(f"\n{'='*65}")
    print(f"  BY CONFIDENCE LEVEL")
    print(f"{'='*65}")
    print_group("", cb, "confidence", order=["HIGH", "MEDIUM", "LOW"])

    # By conference tier
    print(f"\n{'='*65}")
    print(f"  BY CONFERENCE TIER")
    print(f"{'='*65}")
    print_group("", cb, "game_tier", order=["high", "mid", "low", "unknown"])

    # Spread vs Total by tier
    print(f"\n{'='*65}")
    print(f"  SPREAD vs TOTAL BY TIER")
    print(f"{'='*65}")
    for tier in ["high", "mid", "low", "unknown"]:
        tier_bets = [b for b in cb if b["game_tier"] == tier]
        if not tier_bets:
            continue
        sp = [b for b in tier_bets if b["market"] == "SPREAD"]
        to = [b for b in tier_bets if b["market"] == "TOTAL"]
        print(f"\n  [{tier.upper()} conference]")
        if sp: summarize(sp, "Spread")
        if to: summarize(to, "Total")

    # High-major breakdown
    hm = [b for b in cb if b["game_tier"] == "high"]
    if hm:
        print(f"\n{'='*65}")
        print(f"  HIGH-MAJOR ONLY (most reliable)")
        print(f"{'='*65}")
        print_group("By confidence", hm, "confidence", order=["HIGH", "MEDIUM", "LOW"])
        print_group("By market", hm, "market")

    # Day by day
    print(f"\n{'='*65}")
    print(f"  DAY-BY-DAY")
    print(f"{'='*65}")
    by_date = defaultdict(list)
    for b in cb:
        by_date[b["date"]].append(b)
    running_w, running_l = 0, 0
    print(f"  {'Date':<14} {'W':>4} {'L':>4} {'P':>4} {'Win%':>8}  {'Running W%':>12}")
    print(f"  {'-'*52}")
    for date in sorted(by_date):
        d = by_date[date]
        dw = sum(1 for b in d if b["result"] == "WIN")
        dl = sum(1 for b in d if b["result"] == "LOSS")
        dp = sum(1 for b in d if b["result"] == "PUSH")
        running_w += dw
        running_l += dl
        print(f"  {date:<14} {dw:>4} {dl:>4} {dp:>4} {wp(dw,dl):>8}  {wp(running_w,running_l):>12}")

    # Detail table
    print(f"\n{'='*65}")
    print(f"  ALL GRADED BETS")
    print(f"{'='*65}")
    print(f"  {'Date':<12} {'Matchup':<30} {'Mkt':<7} {'Lean':<6} {'Edge':>5} {'Conf':<8} {'Tier':<8} Result")
    print(f"  {'-'*92}")
    for b in cb:
        matchup = f"{b['away'][:13]} @ {b['home'][:13]}"
        ep = f"{b['edge_pts']:.1f}" if b['edge_pts'] else "?"
        result_sym = "✓" if b["result"] == "WIN" else ("~" if b["result"] == "PUSH" else "✗")
        print(f"  {b['date']:<12} {matchup:<30} {b['market']:<7} {b['lean']:<6} {ep:>5} {b['confidence']:<8} {b['game_tier']:<8} {result_sym} {b['result']}")

    print(f"\n  ⚠  {n} clean bets graded. Need 200+ for statistical significance.")


if __name__ == "__main__":
    run_backtest()