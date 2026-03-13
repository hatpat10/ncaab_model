"""
restore_predictions.py
Restores original prediction JSON files (with bets/edge data) from git history,
then re-populates 'actual' scores from games_raw in the DB.

Run from D:\ncaab_model with venv active.
"""
import subprocess, json, os, sqlite3, glob

DB_PATH = "data/ncaab.db"

# ── Step 1: find all prediction dates in git history ────────────────────────
def get_git_prediction_files():
    """List all prediction JSON files that ever existed in git."""
    result = subprocess.run(
        ["git", "log", "--name-only", "--pretty=format:", "--diff-filter=A",
         "--", "predictions/2026-*.json"],
        capture_output=True, text=True
    )
    files = sorted(set(
        line.strip() for line in result.stdout.splitlines()
        if line.strip().endswith(".json")
    ))
    return files

# ── Step 2: get the FIRST commit that introduced each file ───────────────────
def get_first_commit(filepath):
    result = subprocess.run(
        ["git", "log", "--follow", "--diff-filter=A",
         "--pretty=format:%H", "--", filepath],
        capture_output=True, text=True
    )
    commits = [c.strip() for c in result.stdout.splitlines() if c.strip()]
    return commits[-1] if commits else None  # oldest = last in log

# ── Step 3: restore file content at a given commit ──────────────────────────
def get_file_at_commit(filepath, commit):
    result = subprocess.run(
        ["git", "show", f"{commit}:{filepath}"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        return None
    try:
        return json.loads(result.stdout)
    except:
        return None

# ── Step 4: load actual scores from DB ──────────────────────────────────────
def load_actual_scores():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("""
        SELECT date, home_team, away_team, home_score, away_score
        FROM games_raw
        WHERE completed = 1 AND home_score IS NOT NULL
    """)
    scores = {}
    for row in cur.fetchall():
        key = (row["date"], row["home_team"].strip().lower(), row["away_team"].strip().lower())
        scores[key] = {
            "home_score": row["home_score"],
            "away_score":  row["away_score"]
        }
    conn.close()
    return scores

# ── Step 5: match and inject actual scores ──────────────────────────────────
def inject_actual(data, scores, date):
    matched = 0
    for game in data:
        home = game.get("home_team", "").strip().lower()
        away = game.get("away_team", "").strip().lower()
        key  = (date, home, away)
        if key in scores:
            game["actual"] = scores[key]
            matched += 1
        else:
            # Try partial first-word match
            hw = home.split()[0] if home else ""
            aw = away.split()[0] if away else ""
            for (d, h, a), v in scores.items():
                if d == date and h.startswith(hw) and a.startswith(aw):
                    game["actual"] = v
                    matched += 1
                    break
    return matched

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("Loading actual scores from DB...")
    scores = load_actual_scores()
    print(f"  {len(scores)} completed games found\n")

    print("Scanning git history for original prediction files...")
    pred_files = get_git_prediction_files()
    if not pred_files:
        # Fallback: just use files on disk
        pred_files = [f.replace("\\","/") for f in glob.glob("predictions/2026-*.json")]
    print(f"  Found {len(pred_files)} prediction files in history\n")

    restored = 0
    skipped  = 0

    for filepath in sorted(pred_files):
        date = os.path.basename(filepath)[:10]

        # Get original file with bets from first commit
        commit = get_first_commit(filepath)
        if not commit:
            print(f"  {date}: no git history found, skipping")
            skipped += 1
            continue

        data = get_file_at_commit(filepath, commit)
        if not data:
            print(f"  {date}: could not parse file at commit {commit[:7]}")
            skipped += 1
            continue

        with_bets = sum(1 for g in data if g.get("bets"))
        if with_bets == 0:
            print(f"  {date}: no bets in original file (commit {commit[:7]}), skipping")
            skipped += 1
            continue

        # Inject actual scores
        matched = inject_actual(data, scores, date)

        # Save restored file
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"  {date}: ✓  {len(data)} games | {with_bets} with bets | {matched} actual scores injected  [commit {commit[:7]}]")
        restored += 1

    print(f"\n{'='*60}")
    print(f"Done: {restored} files restored, {skipped} skipped")
    print(f"\nNow run:  python backtest_ats.py")

if __name__ == "__main__":
    main()