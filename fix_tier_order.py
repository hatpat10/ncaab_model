# fix_tier_order.py
path = 'predictions/daily_pipeline.py'
with open(path, encoding='utf-8') as f:
    content = f.read()

old = '''        # Compute edge vs Vegas
        edge = compute_edge(preds, odds_row)

        # Bet recommendations
        bets = recommend_bets(edge, preds, game_tier=game.get("game_tier", "unknown"))

        # Assemble output
        result = {
            "game_id":    game["game_id"],
            "date":       date_str,
            "home_team":  home,
            "away_team":  away,
            "tipoff":     game.get("tipoff_time", ""),
            "venue":      game.get("venue", ""),
            "neutral":    bool(game.get("neutral", 0)),
            "game_tier":  get_game_tier(home, away),'''

new = '''        # Compute edge vs Vegas
        edge = compute_edge(preds, odds_row)

        # Compute game tier BEFORE bet recommendations so tier filtering works
        game_tier = get_game_tier(home, away)

        # Bet recommendations
        bets = recommend_bets(edge, preds, game_tier=game_tier)

        # Assemble output
        result = {
            "game_id":    game["game_id"],
            "date":       date_str,
            "home_team":  home,
            "away_team":  away,
            "tipoff":     game.get("tipoff_time", ""),
            "venue":      game.get("venue", ""),
            "neutral":    bool(game.get("neutral", 0)),
            "game_tier":  game_tier,'''

if old in content:
    content = content.replace(old, new)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Fixed: game_tier now computed before recommend_bets()")
else:
    print("❌ Pattern not found — check for whitespace differences")
    # Debug: show what's around line 784
    lines = content.splitlines()
    for i, line in enumerate(lines[778:810], 779):
        print(i, repr(line))
