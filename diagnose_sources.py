"""
Run this from your project root to inspect actual column names in all sources.
    python diagnose_sources.py
"""
import sqlite3
import pandas as pd
from pathlib import Path

BASE = Path(".")
RAW  = BASE / "data" / "raw"
DB   = BASE / "data" / "ncaab.db"

SEP = "─" * 60

# ── BartTorvik ────────────────────────────────────────────────
print(f"\n{'='*60}")
print("BARTTORVIK COLUMNS")
print(f"{'='*60}")
for fname in ["barttorvik_2020_2023.parquet", "barttorvik_2024_2025.parquet"]:
    p = RAW / fname
    if p.exists():
        df = pd.read_parquet(p)
        print(f"\n{fname}  →  {df.shape}")
        print(f"  Columns ({len(df.columns)}): {df.columns.tolist()}")
        print(f"  Sample row:\n{df.iloc[0].to_dict()}")
    else:
        print(f"  NOT FOUND: {p}")

# ── odds_raw ──────────────────────────────────────────────────
print(f"\n{'='*60}")
print("ODDS_RAW COLUMNS")
print(f"{'='*60}")
try:
    con = sqlite3.connect(DB)
    odds = pd.read_sql("SELECT * FROM odds_raw LIMIT 5", con)
    con.close()
    print(f"  Shape (first 5): {odds.shape}")
    print(f"  Columns: {odds.columns.tolist()}")
    print(f"  Sample:\n{odds.to_string()}")
except Exception as e:
    print(f"  ERROR: {e}")

# ── games_raw ─────────────────────────────────────────────────
print(f"\n{'='*60}")
print("GAMES_RAW COLUMNS")
print(f"{'='*60}")
try:
    con = sqlite3.connect(DB)
    games = pd.read_sql("SELECT * FROM games_raw LIMIT 3", con)
    con.close()
    print(f"  Shape (first 3): {games.shape}")
    print(f"  Columns: {games.columns.tolist()}")
except Exception as e:
    print(f"  ERROR: {e}")

# ── feature_matrix join check ─────────────────────────────────
print(f"\n{'='*60}")
print("FEATURE MATRIX — BARTTORVIK JOIN RATE BY SEASON")
print(f"{'='*60}")
fm = Path("data/processed/feature_matrix.parquet")
if fm.exists():
    df = pd.read_parquet(fm)
    t_cols = [c for c in df.columns if c.startswith("t_")]
    o_cols = [c for c in df.columns if c.startswith("o_")]
    print(f"  t_ columns: {t_cols}")
    print(f"  o_ columns: {o_cols}")
    if "t_adj_o" in df.columns and "season" in df.columns:
        print(f"\n  t_adj_o coverage by season:")
        print(df.groupby("season")["t_adj_o"].apply(lambda x: f"{x.notna().mean():.1%}"))
    if "team_id" in df.columns and "t_adj_o" in df.columns:
        missing = df[df["t_adj_o"].isna()]["team_id"].value_counts().head(20)
        if len(missing):
            print(f"\n  Teams missing BartTorvik join (top 20):")
            for t, n in missing.items():
                print(f"    '{t}'  →  {n} games")