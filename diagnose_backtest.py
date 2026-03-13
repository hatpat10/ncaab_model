import json, glob, os

files = sorted(glob.glob("predictions/2026-*.json"))
print(f"Total files: {len(files)}\n")

total_with_actual = 0
total_with_bets = 0
total_both = 0

for f in files:
    data = json.load(open(f, encoding='utf-8'))
    with_actual = [g for g in data if g.get('actual')]
    with_bets   = [g for g in data if g.get('bets')]
    both        = [g for g in data if g.get('actual') and g.get('bets')]
    total_with_actual += len(with_actual)
    total_with_bets   += len(with_bets)
    total_both        += len(both)
    date = os.path.basename(f)[:10]
    if with_bets or with_actual:
        print(f"{date}: {len(data)} games | {len(with_actual)} with actual | {len(with_bets)} with bets | {len(both)} gradeable")

print(f"\nTotals: {total_with_actual} with actual, {total_with_bets} with bets, {total_both} gradeable")

# Show structure of one bet
for f in files:
    data = json.load(open(f, encoding='utf-8'))
    g = next((x for x in data if x.get('bets')), None)
    if g:
        print(f"\nSample from {os.path.basename(f)}:")
        print(f"  Keys: {list(g.keys())}")
        print(f"  actual: {g.get('actual')}")
        print(f"  bets[0]: {g['bets'][0]}")
        print(f"  edge keys: {list(g.get('edge', {}).keys())}")
        break