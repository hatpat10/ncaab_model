"""
backtest_ats.py — NCAAB ATS Backtester v5
- Reads actual scores from games_raw DB
- ±1 day date tolerance for matching (handles day-of-game vs predicted date)
- Aggressive team name normalization
- Falls back to JSON 'actual' field if DB lookup fails
"""
import json, glob, os, sqlite3, re
from datetime import datetime, timedelta
from collections import defaultdict

CLEAN_DATE = "2026-03-04"
DB_PATH    = "data/ncaab.db"

STRIP_WORDS = {
    "university","college","state","the","of","at","and",
    "golden","blue","red","green","black","white","big","little",
    "a&m","st","jr","am",
}

def normalize_name(name):
    n = name.lower().strip()
    n = re.sub(r"['.&()]", "", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n

def key_token(name):
    n = normalize_name(name)
    words = [w for w in n.split() if w not in STRIP_WORDS and len(w) > 2]
    return " ".join(words[:2]) if words else n

def build_db_index(scores):
    idx = {}
    for (date, home, away), v in scores.items():
        k = (date, key_token(home), key_token(away))
        idx[k] = v
    return idx

def load_db_scores():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur  = conn.cursor()
    cur.execute("""
        SELECT date, home_team, away_team, home_score, away_score
        FROM games_raw
        WHERE completed = 1 AND home_score IS NOT NULL
    """)
    scores = {}
    for r in cur.fetchall():
        key = (r["date"], r["home_team"].strip().lower(), r["away_team"].strip().lower())
        scores[key] = {
            "home_score": r["home_score"],
            "away_score":  r["away_score"],
            "margin":      r["home_score"] - r["away_score"],
            "total":       r["home_score"] + r["away_score"],
        }
    conn.close()
    return scores

def find_score(scores, db_idx, date, home, away):
    hl = home.strip().lower()
    al = away.strip().lower()
    hn = normalize_name(home)
    an = normalize_name(away)
    hk = key_token(home)
    ak = key_token(away)
    hw = hl.split()[0] if hl else ""
    aw = al.split()[0] if al else ""

    # Search ±1 day window (handles games played day before/after predicted date)
    base = datetime.strptime(date, "%Y-%m-%d")
    for delta in [0, 1, -1]:
        d = (base + timedelta(days=delta)).strftime("%Y-%m-%d")

        # 1. Exact lowercase
        if (d, hl, al) in scores:
            return scores[(d, hl, al)]

        # 2. Normalized exact
        for (dd, h, a), v in scores.items():
            if dd == d and normalize_name(h) == hn and normalize_name(a) == an:
                return v

        # 3. Key-token
        if (d, hk, ak) in db_idx:
            return db_idx[(d, hk, ak)]

        # 4. First-word prefix (min 4 chars to avoid false matches)
        if len(hw) >= 4 and len(aw) >= 4:
            for (dd, h, a), v in scores.items():
                if dd == d and h.startswith(hw) and a.startswith(aw):
                    return v

    return None

def grade(bet, edge, score):
    if not score:
        return None
    mkt  = bet.get("market")
    lean = bet.get("lean")
    mg   = score["margin"]
    tot  = score["total"]
    vs   = edge.get("vegas_spread")
    vt   = edge.get("vegas_total")

    if mkt == "SPREAD":
        if vs is None: return None
        cover = -vs
        if lean == "HOME":
            return "WIN" if mg > cover else ("PUSH" if mg == cover else "LOSS")
        elif lean == "AWAY":
            return "WIN" if mg < cover else ("PUSH" if mg == cover else "LOSS")
    elif mkt == "TOTAL":
        if vt is None: return None
        if lean == "OVER":
            return "WIN" if tot > vt else ("PUSH" if tot == vt else "LOSS")
        elif lean == "UNDER":
            return "WIN" if tot < vt else ("PUSH" if tot == vt else "LOSS")
    return None

def wp(w, l):
    return f"{w/(w+l)*100:.1f}%" if (w + l) > 0 else "N/A"

def summarize(rows, label=""):
    w = sum(1 for r in rows if r["result"] == "WIN")
    l = sum(1 for r in rows if r["result"] == "LOSS")
    p = sum(1 for r in rows if r["result"] == "PUSH")
    t = w + l + p
    print(f"  [{label}]{' '*(16-len(label))} W:{w:>4}  L:{l:>4}  P:{p:>3}  n={t:<5}  {wp(w,l):>8}")

def group_print(rows, key, order=None):
    g = defaultdict(list)
    for r in rows: g[r[key]].append(r)
    keys = order if order else sorted(g)
    print(f"  {'-'*57}")
    for k in keys:
        if k in g:
            summarize(g[k], str(k))

def main():
    scores = load_db_scores()
    db_idx = build_db_index(scores)
    files  = sorted(glob.glob("predictions/2026-*.json"))

    all_bets, clean_bets = [], []
    unmatched = 0
    matched   = 0

    for f in files:
        date     = os.path.basename(f)[:10]
        is_clean = date >= CLEAN_DATE
        data     = json.load(open(f, encoding="utf-8"))

        for game in data:
            bets = game.get("bets", [])
            edge = game.get("edge", {})
            if not bets or not edge:
                continue

            home      = game.get("home_team", "")
            away      = game.get("away_team", "")
            game_tier = game.get("game_tier") or "unknown"

            score = find_score(scores, db_idx, date, home, away)
            if not score:
                ja = game.get("actual")
                if ja and ja.get("home_score") is not None:
                    hs, as_ = ja["home_score"], ja["away_score"]
                    score = {"home_score": hs, "away_score": as_,
                             "margin": hs - as_, "total": hs + as_}
            if not score:
                unmatched += 1
                continue

            matched += 1
            for bet in bets:
                result = grade(bet, edge, score)
                if result is None:
                    continue
                row = {
                    "date":          date,
                    "home":          home,
                    "away":          away,
                    "market":        bet.get("market"),
                    "lean":          bet.get("lean"),
                    "edge_pts":      bet.get("edge_pts"),
                    "confidence":    bet.get("confidence", "UNKNOWN"),
                    "game_tier":     bet.get("game_tier") or game_tier,
                    "vegas_spread":  edge.get("vegas_spread"),
                    "vegas_total":   edge.get("vegas_total"),
                    "actual_margin": score.get("margin"),
                    "actual_total":  score.get("total"),
                    "result":        result,
                }
                all_bets.append(row)
                if is_clean:
                    clean_bets.append(row)

    cb   = clean_bets
    n    = len(cb)
    excl = len(all_bets) - n

    print(f"\n{'='*67}")
    print(f"  NCAAB ML MODEL — ATS BACKTEST  (v5 — ±1 day matching)")
    print(f"  Clean from: {CLEAN_DATE}  |  Clean bets: {n}  |  Pre-fix excluded: {excl}")
    print(f"  Games matched: {matched}  |  Unmatched: {unmatched}")
    print(f"{'='*67}")

    if n == 0:
        ab = len(all_bets)
        print(f"\n  ⚠  0 clean bets graded. Pre-fix pool: {ab} bets.")
        if ab:
            summarize(all_bets, "All pre-fix")
        print(f"\n  Try: python rescrape_dates.py  (fills gaps in games_raw)")
        return

    # Overall
    print(f"\n{'='*67}\n  OVERALL\n{'='*67}")
    summarize(cb, "All")
    summarize([b for b in cb if b["market"] == "SPREAD"], "Spread")
    summarize([b for b in cb if b["market"] == "TOTAL"],  "Total")
    w = sum(1 for b in cb if b["result"] == "WIN")
    l = sum(1 for b in cb if b["result"] == "LOSS")
    if w + l > 0:
        roi = (w * 0.909 - l) / (w + l) * 100
        print(f"\n  Est. ROI @ -110 vig:  {roi:+.1f}%  (breakeven = 52.4%)")

    # By confidence
    print(f"\n{'='*67}\n  BY CONFIDENCE LEVEL\n{'='*67}")
    group_print(cb, "confidence", ["HIGH", "MEDIUM", "LOW"])

    # By tier
    print(f"\n{'='*67}\n  BY CONFERENCE TIER\n{'='*67}")
    group_print(cb, "game_tier", ["high", "mid", "low", "unknown"])

    # Spread vs Total by tier
    print(f"\n{'='*67}\n  SPREAD vs TOTAL BY TIER\n{'='*67}")
    for tier in ["high", "mid", "low", "unknown"]:
        tb = [b for b in cb if b["game_tier"] == tier]
        if not tb: continue
        sp = [b for b in tb if b["market"] == "SPREAD"]
        to = [b for b in tb if b["market"] == "TOTAL"]
        print(f"\n  [{tier.upper()} conference]")
        if sp: summarize(sp, "Spread")
        if to: summarize(to, "Total")

    # High-major only
    hm = [b for b in cb if b["game_tier"] == "high"]
    if hm:
        print(f"\n{'='*67}\n  HIGH-MAJOR ONLY\n{'='*67}")
        group_print(hm, "confidence", ["HIGH", "MEDIUM", "LOW"])
        print()
        group_print(hm, "market")

    # Pre-fix breakdown
    if excl > 0:
        pre = [b for b in all_bets if b["date"] < CLEAN_DATE]
        print(f"\n{'='*67}\n  PRE-FIX SAMPLE  ⚠  (unreliable — broken tier logic)\n{'='*67}")
        summarize(pre, "All pre-fix")
        group_print(pre, "confidence", ["HIGH", "MEDIUM", "LOW"])

    # Day by day
    print(f"\n{'='*67}\n  DAY-BY-DAY  (clean bets only)\n{'='*67}")
    by_date = defaultdict(list)
    for b in cb: by_date[b["date"]].append(b)
    rw = rl = 0
    print(f"  {'Date':<14} {'W':>4} {'L':>4} {'P':>4}   {'Win%':>8}   {'Running':>10}")
    print(f"  {'-'*54}")
    for date in sorted(by_date):
        d  = by_date[date]
        dw = sum(1 for b in d if b["result"] == "WIN")
        dl = sum(1 for b in d if b["result"] == "LOSS")
        dp = sum(1 for b in d if b["result"] == "PUSH")
        rw += dw; rl += dl
        print(f"  {date:<14} {dw:>4} {dl:>4} {dp:>4}   {wp(dw,dl):>8}   {wp(rw,rl):>10}")

    # Detail
    print(f"\n{'='*67}\n  ALL GRADED CLEAN BETS\n{'='*67}")
    print(f"  {'Date':<12} {'Matchup':<30} {'Mkt':<7} {'Lean':<6} {'Edge':>5} {'Conf':<8} {'Tier':<8} Result")
    print(f"  {'-'*94}")
    for b in sorted(cb, key=lambda x: x["date"]):
        matchup = f"{b['away'][:13]} @ {b['home'][:13]}"
        ep  = f"{b['edge_pts']:.1f}" if b["edge_pts"] else "?"
        sym = "✓" if b["result"] == "WIN" else ("~" if b["result"] == "PUSH" else "✗")
        print(f"  {b['date']:<12} {matchup:<30} {b['market']:<7} {b['lean']:<6} {ep:>5} "
              f"{b['confidence']:<8} {b['game_tier']:<8} {sym} {b['result']}")

    sig = "✅ Good sample" if n >= 200 else f"⚠  Need {200-n} more bets for significance (200+ target)"
    print(f"\n  {sig}  |  Breakeven ATS win rate = 52.4%")

if __name__ == "__main__":
    main()