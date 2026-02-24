"""Run from project root: python diagnose_join.py"""
import sqlite3
import pandas as pd
from pathlib import Path
import json

RAW = Path("data/raw")
DB  = Path("data/ncaab.db")

# Load aliases
with open("data/team_aliases.json") as f:
    ALIASES = json.load(f)

def normalize_team(raw_name):
    if not isinstance(raw_name, str):
        return str(raw_name)
    name = raw_name.strip()
    if name in ALIASES:
        return ALIASES[name]
    lower = name.lower()
    for key, val in ALIASES.items():
        if key.lower() == lower:
            return val
    words = name.split()
    for n_drop in range(1, min(3, len(words))):
        shorter = " ".join(words[:-n_drop])
        if shorter in ALIASES:
            return ALIASES[shorter]
        for key, val in ALIASES.items():
            if key.lower() == shorter.lower():
                return val
    return name.lower().replace(" ", "_").replace("-", "_")

# ── BartTorvik team_ids ───────────────────────────────────────────────
frames = []
for fname in ["barttorvik_2020_2023.parquet", "barttorvik_2024_2025.parquet"]:
    p = RAW / fname
    if p.exists():
        frames.append(pd.read_parquet(p))
torvik = pd.concat(frames, ignore_index=True)
torvik["team_id"] = torvik["team"].apply(normalize_team)
torvik["season"]  = pd.to_numeric(torvik["year"], errors="coerce") + 1

torvik_2026 = torvik[torvik["season"] == 2026]
print(f"BartTorvik rows with season=2026: {len(torvik_2026)}")
print(f"Sample team_ids from BartTorvik (season=2026):")
print(sorted(torvik_2026["team_id"].unique().tolist())[:40])

# ── ESPN team_ids ─────────────────────────────────────────────────────
con = sqlite3.connect(DB)
games = pd.read_sql("SELECT home_team, away_team, season FROM games_raw LIMIT 200", con)
con.close()
games["home_id"] = games["home_team"].apply(normalize_team)
games["away_id"] = games["away_team"].apply(normalize_team)
espn_teams = set(games["home_id"].tolist() + games["away_id"].tolist())
print(f"\nESPN unique team_ids (sample 200 games): {len(espn_teams)}")
print(f"Sample ESPN team_ids:")
print(sorted(list(espn_teams))[:40])

# ── Find mismatches ───────────────────────────────────────────────────
torvik_ids = set(torvik_2026["team_id"].unique())
in_espn_not_torvik = espn_teams - torvik_ids
in_torvik_not_espn = torvik_ids - espn_teams

print(f"\nIn ESPN but NOT in BartTorvik ({len(in_espn_not_torvik)}):")
for t in sorted(in_espn_not_torvik)[:30]:
    # Show what BartTorvik raw name is closest
    print(f"  '{t}'")

print(f"\nIn BartTorvik but NOT in ESPN ({len(in_torvik_not_espn)}):")
for t in sorted(in_torvik_not_espn)[:30]:
    raw = torvik_2026[torvik_2026["team_id"] == t]["team"].iloc[0] if t in torvik_ids else "?"
    print(f"  '{t}'  (raw: '{raw}')")