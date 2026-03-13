"""
diagnose_unmatched.py
Shows which games in prediction files can't be matched to games_raw in the DB.
Helps identify team name normalization issues.
"""
import json, glob, os, sqlite3

DB_PATH = "data/ncaab.db"
CLEAN_DATE = "2026-03-04"

def load_db_scores():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("""
        SELECT date, home_team, away_team, home_score, away_score
        FROM games_raw
        WHERE completed = 1 AND home_score IS NOT NULL
    """)
    scores = {}
    for r in cur.fetchall():
        key = (r["date"], r["home_team"].strip().lower(), r["away_team"].strip().lower())
        scores[key] = (r["home_score"], r["away_score"])
    conn.close()
    return scores

def main():
    db = load_db_scores()
    files = sorted(glob.glob("predictions/2026-*.json"))

    unmatched_by_date = {}

    for f in files:
        date = os.path.basename(f)[:10]
        data = json.load(open(f, encoding="utf-8"))

        for game in data:
            bets = game.get("bets", [])
            if not bets:
                continue

            home = game.get("home_team", "").strip()
            away = game.get("away_team", "").strip()
            hl   = home.lower()
            al   = away.lower()

            # Try exact
            if (date, hl, al) in db:
                continue

            # Try partial
            hw = hl.split()[0] if hl else ""
            aw = al.split()[0] if al else ""
            found = False
            candidates = []
            for (d, h, a) in db:
                if d == date:
                    if h.startswith(hw) and a.startswith(aw):
                        found = True
                        break
                    # Show close candidates
                    if h.startswith(hw) or a.startswith(aw):
                        candidates.append((h, a))

            if not found:
                if date not in unmatched_by_date:
                    unmatched_by_date[date] = []
                unmatched_by_date[date].append({
                    "pred": f"{away} @ {home}",
                    "candidates": candidates[:3]
                })

    total = sum(len(v) for v in unmatched_by_date.values())
    print(f"Total unmatched games with bets: {total}\n")

    for date in sorted(unmatched_by_date):
        games = unmatched_by_date[date]
        marker = " ← CLEAN" if date >= CLEAN_DATE else ""
        print(f"[{date}]{marker}  ({len(games)} unmatched)")
        for g in games:
            print(f"  PRED: {g['pred']}")
            if g['candidates']:
                print(f"    DB candidates: {g['candidates']}")
        print()

if __name__ == "__main__":
    main()