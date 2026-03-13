"""
find_missing_games.py
For each unmatched game (bet game with no DB score on that date),
searches games_raw across ALL dates to see if it exists elsewhere.
This identifies date mismatches vs truly missing games.
"""
import json, glob, os, sqlite3, re
from collections import defaultdict

DB_PATH    = "data/ncaab.db"
CLEAN_DATE = "2026-03-04"

STRIP_WORDS = {
    "university","college","state","the","of","at","and",
    "golden","blue","red","green","black","white","big","little",
    "a&m","st","jr","am",
}

def normalize(name):
    n = name.lower().strip()
    n = re.sub(r"['.&()]", "", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n

def key_token(name):
    n = normalize(name)
    words = [w for w in n.split() if w not in STRIP_WORDS and len(w) > 2]
    return " ".join(words[:2]) if words else n

def load_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("""
        SELECT date, home_team, away_team, home_score, away_score, completed
        FROM games_raw
    """)
    rows = cur.fetchall()
    conn.close()

    # Index by exact key and by token key (across all dates)
    exact = {}
    token_to_dates = defaultdict(list)  # (home_token, away_token) -> [(date, home, away, scores)]
    for r in rows:
        hl = r["home_team"].strip().lower()
        al = r["away_team"].strip().lower()
        exact[(r["date"], hl, al)] = r
        tk = (key_token(r["home_team"]), key_token(r["away_team"]))
        token_to_dates[tk].append({
            "date": r["date"],
            "home": r["home_team"],
            "away": r["away_team"],
            "home_score": r["home_score"],
            "away_score": r["away_score"],
            "completed": r["completed"],
        })
    return exact, token_to_dates

def main():
    exact, token_index = load_db()
    files = sorted(glob.glob("predictions/2026-*.json"))

    date_mismatches = []
    truly_missing   = []
    clean_missing   = []

    for f in files:
        date = os.path.basename(f)[:10]
        data = json.load(open(f, encoding="utf-8"))

        for game in data:
            bets = game.get("bets", [])
            if not bets:
                continue

            home = game.get("home_team", "")
            away = game.get("away_team", "")
            hl   = home.strip().lower()
            al   = away.strip().lower()

            # Check if exact match exists on this date
            if (date, hl, al) in exact:
                continue
            # Check normalized
            hn, an = normalize(home), normalize(away)
            found_exact = False
            for (d, h, a) in exact:
                if d == date and normalize(h) == hn and normalize(a) == an:
                    found_exact = True
                    break
            if found_exact:
                continue

            # Not found on this date — search across all dates
            hk = key_token(home)
            ak = key_token(away)
            tk = (hk, ak)

            matches_other_date = token_index.get(tk, [])
            # Also try partial token
            if not matches_other_date:
                for (h_tok, a_tok), entries in token_index.items():
                    if (hk.split()[0] if hk else "") in h_tok and (ak.split()[0] if ak else "") in a_tok:
                        matches_other_date = entries
                        break

            entry = {
                "pred_date": date,
                "home": home,
                "away": away,
                "is_clean": date >= CLEAN_DATE,
                "matches": matches_other_date[:3],
            }

            if matches_other_date:
                date_mismatches.append(entry)
            else:
                truly_missing.append(entry)
                if date >= CLEAN_DATE:
                    clean_missing.append(entry)

    print(f"{'='*65}")
    print(f"  UNMATCHED GAMES ANALYSIS")
    print(f"  Date mismatches (game in DB, wrong date): {len(date_mismatches)}")
    print(f"  Truly missing (not in DB at all):         {len(truly_missing)}")
    print(f"  Clean-period truly missing:               {len(clean_missing)}")
    print(f"{'='*65}")

    # Show clean-period date mismatches
    clean_dm = [e for e in date_mismatches if e["is_clean"]]
    if clean_dm:
        print(f"\n  DATE MISMATCHES — CLEAN PERIOD ({len(clean_dm)} games)")
        print(f"  (These could be graded if we match by team name across dates)")
        print(f"  {'-'*60}")
        for e in clean_dm:
            print(f"  [{e['pred_date']}] {e['away']} @ {e['home']}")
            for m in e["matches"]:
                sc = f"{m['away_score']}-{m['home_score']}" if m["home_score"] is not None else "no score"
                print(f"    → DB: {m['date']}  {m['away']} @ {m['home']}  ({sc})")

    if clean_missing:
        print(f"\n  TRULY MISSING — CLEAN PERIOD ({len(clean_missing)} games)")
        print(f"  (These games are not in games_raw at all)")
        print(f"  {'-'*60}")
        for e in clean_missing:
            print(f"  [{e['pred_date']}] {e['away']} @ {e['home']}")

    # Summary by date for pre-fix period
    print(f"\n  PRE-FIX SUMMARY BY DATE")
    pre_dm    = [e for e in date_mismatches if not e["is_clean"]]
    pre_miss  = [e for e in truly_missing   if not e["is_clean"]]
    by_date   = defaultdict(lambda: {"dm": 0, "miss": 0})
    for e in pre_dm:   by_date[e["pred_date"]]["dm"]   += 1
    for e in pre_miss: by_date[e["pred_date"]]["miss"] += 1
    print(f"  {'Date':<14} {'Date-mismatch':>14} {'Missing':>10}")
    print(f"  {'-'*40}")
    for d in sorted(by_date):
        print(f"  {d:<14} {by_date[d]['dm']:>14} {by_date[d]['miss']:>10}")

if __name__ == "__main__":
    main()