"""Run from project root: python diagnose_season2.py"""
import sqlite3
import pandas as pd
from pathlib import Path
import json

RAW = Path("data/raw")
DB  = Path("data/ncaab.db")

# ── What season values does BartTorvik produce after the +1 fix? ──────────
frames = []
for fname in ["barttorvik_2020_2023.parquet", "barttorvik_2024_2025.parquet"]:
    p = RAW / fname
    if p.exists():
        df = pd.read_parquet(p)
        frames.append(df)
torvik = pd.concat(frames, ignore_index=True)
torvik.columns = [c.lower().strip() for c in torvik.columns]

print("BartTorvik raw 'year' values:", sorted(torvik["year"].dropna().unique().tolist()))

# Simulate what load_barttorvik does
torvik["season"] = pd.to_numeric(torvik["year"], errors="coerce") + 1
print("BartTorvik season values after +1:", sorted(torvik["season"].dropna().unique().tolist()))
print("BartTorvik season dtype:", torvik["season"].dtype)
print()

# ── What season values does ESPN produce? ─────────────────────────────────
con = sqlite3.connect(DB)
games = pd.read_sql("SELECT season, date FROM games_raw LIMIT 10", con)
con.close()
print("ESPN games_raw season values:", sorted(games["season"].dropna().unique().tolist()))
print("ESPN season dtype:", games["season"].dtype)
print()

# ── Simulate the actual merge ─────────────────────────────────────────────
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
    return name.lower().replace(" ", "_").replace("-", "_")

torvik["team_id"] = torvik["team"].apply(normalize_team)

con = sqlite3.connect(DB)
espn = pd.read_sql("SELECT game_id, home_team, away_team, season FROM games_raw", con)
con.close()
espn["home_id"] = espn["home_team"].apply(normalize_team)
espn["season"] = pd.to_numeric(espn["season"], errors="coerce")

print("Trying test join for 'kentucky', season 2026:")
kt = torvik[torvik["team_id"] == "kentucky"][["team_id","season","adj_o"]]
print("  BartTorvik rows:", kt.to_string())

espn_kt = espn[espn["home_id"] == "kentucky"][["game_id","home_id","season"]].head(3)
print("  ESPN rows:", espn_kt.to_string())

# Direct merge test
test = espn_kt.rename(columns={"home_id":"team_id"}).merge(
    kt, on=["team_id","season"], how="left"
)
print("  After merge:", test[["team_id","season","adj_o"]].to_string())
print()
print("Season value types:")
print(f"  BartTorvik season sample: {torvik['season'].iloc[0]} (type: {type(torvik['season'].iloc[0])})")
print(f"  ESPN season sample: {espn['season'].iloc[0]} (type: {type(espn['season'].iloc[0])})")