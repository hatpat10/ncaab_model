# processing/coverage_audit.py
import pandas as pd
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from processing.data_loader import (
    load_team_box, load_schedule, load_barttorvik,
    load_current_season_games, load_odds
)

def audit():
    print("=" * 55)
    print("COVERAGE AUDIT")
    print("=" * 55)

    # --- Team box scores ---
    tb = load_team_box()
    print("\n[1] hoopR Team Box Scores")
    print(f"    Seasons:  {sorted(tb['season'].unique())}")
    print(f"    Games:    {tb['game_id'].nunique():,}")
    print(f"    Null pct: {tb.isnull().mean().mean()*100:.1f}%")

    # --- Schedule ---
    sc = load_schedule()
    print("\n[2] hoopR Schedule")
    print(f"    Seasons:  {sorted(sc['season'].unique())}")
    completed = sc['status_type_completed'].sum() if 'status_type_completed' in sc.columns else 'N/A'
    print(f"    Completed games: {completed}")

    # --- BartTorvik ---
    bt = load_barttorvik()
    print("\n[3] BartTorvik Ratings")
    print(f"    Years:    {sorted(bt['year'].unique())}")
    print(f"    Teams/yr: {bt.groupby('year')['team'].count().to_dict()}")
    print(f"    Cols:     {list(bt.columns)}")

    # --- Current season ---
    gs = load_current_season_games()
    print("\n[4] Current Season Games (ESPN)")
    print(f"    Total:     {len(gs):,}")
    print(f"    Completed: {gs['completed'].sum():,}")
    print(f"    Upcoming:  {(gs['completed']==0).sum():,}")
    print(f"    Date range: {gs['date'].min()} → {gs['date'].max()}")

    # --- Odds ---
    od = load_odds()
    print("\n[5] Odds")
    print(f"    Total rows: {len(od):,}")
    print(f"    Games:      {od['game_id'].nunique():,}")
    print(f"    Markets:    {od['market'].unique()}")
    print(f"    Bookmakers: {od['bookmaker'].unique()}")

    # --- Gap check ---
    print("\n[6] Gap Check")
    bt_years = set(bt['year'].dropna().astype(int).unique())
    box_seasons = set(tb['season'].unique())
    missing_bt = box_seasons - bt_years
    if missing_bt:
        print(f"    WARNING: BartTorvik missing seasons: {missing_bt}")
    else:
        print(f"    BartTorvik covers all box score seasons: OK")

    upcoming = gs[gs['completed'] == 0][['date','home_team','away_team']].head(5)
    print(f"\n    Next 5 games with odds coverage:")
    print(upcoming.to_string(index=False))

    print("\n" + "=" * 55)
    print("Audit complete.")
    print("=" * 55)

if __name__ == "__main__":
    audit()