"""Run from project root: python diagnose_odds_join.py"""
import sqlite3, pandas as pd, json
from pathlib import Path

DB = Path("data/ncaab.db")
RAW = Path("data/raw")

with open("data/team_aliases.json") as f:
    aliases = json.load(f)

def normalize_team(raw_name):
    if not isinstance(raw_name, str): return str(raw_name)
    name = raw_name.strip()
    if name in aliases: return aliases[name]
    lower = name.lower()
    for key, val in aliases.items():
        if key.lower() == lower: return val
    words = name.split()
    for n_drop in range(1, min(3, len(words))):
        shorter = " ".join(words[:-n_drop])
        if shorter in aliases: return aliases[shorter]
        for key, val in aliases.items():
            if key.lower() == shorter.lower(): return val
    slug_input = name.lower().replace(" ", "_").replace("-", "_").replace("&", "").replace(".", "")
    if slug_input in aliases: return aliases[slug_input]
    return name.lower().replace(" ", "_").replace("-", "_")

# ── Odds side ─────────────────────────────────────────────────────────
con = sqlite3.connect(DB)
odds_raw = pd.read_sql("SELECT DISTINCT game_id, date, home_team, away_team FROM odds_raw LIMIT 5", con)
con.close()
print("=== Odds sample ===")
print(odds_raw.to_string())
print(f"date dtype: {odds_raw['date'].dtype}, sample: {odds_raw['date'].iloc[0]!r}")
odds_raw["_date"] = pd.to_datetime(odds_raw["date"], errors="coerce").dt.normalize()
odds_raw["_home"] = odds_raw["home_team"].apply(normalize_team)
print(f"_date after normalize: {odds_raw['_date'].iloc[0]!r} (type: {type(odds_raw['_date'].iloc[0])})")

# ── Master/ESPN side ───────────────────────────────────────────────────
con = sqlite3.connect(DB)
games = pd.read_sql("SELECT game_id, date, home_team, away_team FROM games_raw LIMIT 5", con)
con.close()
print("\n=== ESPN games sample ===")
print(games.to_string())
print(f"date dtype: {games['date'].dtype}, sample: {games['date'].iloc[0]!r}")
games["_date"] = pd.to_datetime(games["date"], errors="coerce").dt.normalize()
games["_home"] = games["home_team"].apply(normalize_team)
print(f"_date after normalize: {games['_date'].iloc[0]!r} (type: {type(games['_date'].iloc[0])})")

# ── Overlap check ──────────────────────────────────────────────────────
con = sqlite3.connect(DB)
odds_all = pd.read_sql("SELECT DISTINCT game_id, date, home_team, away_team FROM odds_raw", con)
games_all = pd.read_sql("SELECT game_id, date, home_team, away_team FROM games_raw", con)
con.close()

odds_all["_date"] = pd.to_datetime(odds_all["date"], errors="coerce").dt.normalize()
odds_all["_home"] = odds_all["home_team"].apply(normalize_team)
odds_all["_away"] = odds_all["away_team"].apply(normalize_team)

games_all["_date"] = pd.to_datetime(games_all["date"], errors="coerce").dt.normalize()
games_all["_home"] = games_all["home_team"].apply(normalize_team)
games_all["_away"] = games_all["away_team"].apply(normalize_team)

merged = odds_all.merge(games_all, on=["_date","_home","_away"], how="inner")
print(f"\n=== Join test ===")
print(f"Odds games: {len(odds_all)}, ESPN games: {len(games_all)}")
print(f"Matched on (date+home+away): {len(merged)}")
if len(merged) == 0:
    print("\nSample odds keys:")
    print(odds_all[["_date","_home","_away"]].head(5).to_string())
    print("\nSample ESPN keys (same date range):")
    min_d = odds_all["_date"].min()
    max_d = odds_all["_date"].max()
    mask = (games_all["_date"] >= min_d) & (games_all["_date"] <= max_d)
    print(games_all[mask][["_date","_home","_away"]].head(10).to_string())

# ── Missing BartTorvik teams ───────────────────────────────────────────
print("\n=== Missing BartTorvik teams — in parquet? ===")
frames = []
for fname in ["barttorvik_2020_2023.parquet", "barttorvik_2024_2025.parquet"]:
    p = RAW / fname
    if p.exists():
        frames.append(pd.read_parquet(p))
torvik = pd.concat(frames, ignore_index=True)
torvik["team_id"] = torvik["team"].apply(normalize_team)
torvik25 = torvik[torvik["year"] == 2025]

missing = ["iu_indianapolis", "east_texas_am", "omaha", "east_tennessee_state",
           "chicago_state", "alabama_state", "georgia_state", "mercyhurst",
           "st_thomas_minnesota", "new_haven", "southeast_missouri_state",
           "arkansas_state", "west_georgia", "mississippi_valley_state", "maryland_eastern_shore"]
for t in missing:
    row = torvik25[torvik25["team_id"] == t]
    if len(row):
        print(f"  ✓ {t} → adj_o={row['adj_o'].values[0]:.1f}")
    else:
        # Search raw names
        raw_match = torvik[torvik["team"].str.contains(t.replace("_"," ").split()[0], case=False, na=False)]["team"].unique()
        print(f"  ✗ {t} NOT FOUND  (raw search: {raw_match[:3].tolist()})")