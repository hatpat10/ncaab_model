"""
Quick diagnostic — run after feature_builder.py completes.
    python check_feature_matrix.py
"""
import pandas as pd
from pathlib import Path

path = Path("data/processed/feature_matrix.parquet")
if not path.exists():
    print("ERROR: feature_matrix.parquet not found. Run feature_builder.py first.")
    exit(1)

df = pd.read_parquet(path)
print(f"\n{'='*55}")
print(f" Feature Matrix: {df.shape[0]:,} rows × {df.shape[1]} cols")
print(f"{'='*55}")

if "date" in df.columns:
    print(f" Date range  : {df['date'].min().date()} → {df['date'].max().date()}")
if "season" in df.columns:
    print(f" Seasons     : {sorted(df['season'].dropna().unique().astype(int).tolist())}")
if "game_id" in df.columns:
    print(f" Unique games: {df['game_id'].nunique():,}")
if "team_id" in df.columns:
    print(f" Unique teams: {df['team_id'].nunique()}")

print(f"\n{'─'*55}")
print(" Join Coverage (% non-null)")
print(f"{'─'*55}")

checks = {
    "BartTorvik AdjO  (t_adj_o)":         ("t_adj_o",         0.85),
    "BartTorvik AdjD  (t_adj_d)":         ("t_adj_d",         0.85),
    "BartTorvik Opp   (o_adj_d)":         ("o_adj_d",         0.85),
    "Rolling 5g pts   (roll5_pts)":       ("roll5_pts",       0.70),
    "Rolling 5g margin(roll5_margin)":    ("roll5_margin",     0.70),
    "Off eff delta    (off_eff_delta)":   ("off_eff_delta",   0.80),
    "Net rating delta (net_rating_delta)":("net_rating_delta",0.80),
    "Defense tier     (opp_def_tier)":    ("opp_def_tier",    0.80),
    "Days rest        (days_rest)":       ("days_rest",       0.95),
    "Vegas spread     (vegas_spread)":    ("vegas_spread",    0.20),  # only 2025-26
    "Implied prob     (implied_prob_home)":("implied_prob_home",0.20),
}

all_ok = True
for label, (col, threshold) in checks.items():
    if col in df.columns:
        rate = df[col].notna().mean()
        status = "✓" if rate >= threshold else "✗ LOW"
        if rate < threshold:
            all_ok = False
        print(f"  {status}  {label:<40} {rate:.1%}")
    else:
        print(f"  –  {label:<40} (column missing)")

print(f"\n{'─'*55}")
if all_ok:
    print(" All coverage checks passed — ready for model training!")
else:
    print(" Some columns below threshold — check team_aliases.json coverage")
    print(" and verify BartTorvik parquet contains 'adj_o', 'adj_d', 'adj_t'")
    # Show actual BartTorvik columns present
    torvik_cols = [c for c in df.columns if c.startswith("t_") or c.startswith("o_")]
    print(f"\n BartTorvik cols present ({len(torvik_cols)}): {torvik_cols[:15]}")

# Top missing teams (help fix aliases)
if "team_id" in df.columns and "t_adj_o" in df.columns:
    missing_torvik = df[df["t_adj_o"].isna()]["team_id"].value_counts().head(15)
    if len(missing_torvik):
        print(f"\n Top teams missing BartTorvik join (add to team_aliases.json):")
        for team, cnt in missing_torvik.items():
            print(f"   {team:<35} {cnt} games")
print()