import json, glob

# Look at one flagged game to see what fields are available
files = sorted(glob.glob('predictions/2026-*.json'))
print(f"Total prediction files: {len(files)}")
print(f"Date range: {files[0][:10]} to {files[-1][:10]}\n")

# Find a file with flagged bets
for f in files:
    data = json.load(open(f))
    flagged = [g for g in data if g.get('bets')]
    if flagged:
        g = flagged[0]
        print(f"=== Sample flagged game from {f} ===")
        print(f"Teams: {g.get('away_team')} @ {g.get('home_team')}")
        print(f"game_tier: {g.get('game_tier')}")
        print(f"Keys in record: {list(g.keys())}")
        print(f"\nPredictions sub-keys: {list(g.get('predictions', {}).keys())}")
        print(f"\nEdge sub-keys: {list(g.get('edge', {}).keys())}")
        print(f"\nBets: {g.get('bets')}")
        print(f"\nActual scores: {g.get('actual', 'NOT YET COMPLETED')}")
        break

# Summary of flagged bets by date
print("\n=== Flagged bets by date ===")
total_flags = 0
for f in files:
    data = json.load(open(f))
    flagged = [g for g in data if g.get('bets')]
    high = [g for g in flagged if any(b.get('confidence') == 'HIGH' for b in g.get('bets', []))]
    med  = [g for g in flagged if any(b.get('confidence') == 'MEDIUM' for b in g.get('bets', []))]
    if flagged:
        date = f.split('\\')[-1][:10]
        print(f"{date}: {len(flagged)} flagged games ({len(high)} HIGH, {len(med)} MEDIUM)")
        total_flags += len(flagged)
print(f"\nTotal flagged games across all files: {total_flags}")
