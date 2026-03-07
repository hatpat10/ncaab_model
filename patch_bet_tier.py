# patch_bet_tier.py
# Adds game_tier to each individual bet dict in daily_pipeline.py

path = 'predictions/daily_pipeline.py'
with open(path, encoding='utf-8') as f:
    content = f.read()

old = '        bets = recommend_bets(edge, preds, game_tier=game_tier)'
new = ('        bets = recommend_bets(edge, preds, game_tier=game_tier)\n'
       '        # Stamp game_tier onto each individual bet for backtesting\n'
       '        for b in bets:\n'
       '            b["game_tier"] = game_tier')

if old in content:
    content = content.replace(old, new)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("Fixed: game_tier now stamped on each bet dict")
else:
    print("Pattern not found - showing recommend_bets lines:")
    for i, line in enumerate(content.splitlines()):
        if 'recommend_bets' in line:
            print(f"  {i+1}: {line}")
