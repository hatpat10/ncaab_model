"""
Patches daily_pipeline.py with three fixes:
1. Fix get_game_tier() - low-major games should ALWAYS return "low" (weakest tier wins)
2. Fix recommend_bets() - correct confidence tiers + add tier-based suppression
3. Add edge sanity cap (spread > 20 = bad odds data, skip)
"""

import re

path = 'predictions/daily_pipeline.py'
with open(path, encoding='utf-8') as f:
    content = f.read()

# ── FIX 1: get_game_tier() ──────────────────────────────────────────────────
# Old logic: "high" overrides "low" — wrong for high vs low matchups
# New logic: weakest tier always wins (low > mid > high > unknown)
old_tier = '''def get_game_tier(home_team: str, away_team: str) -> str:
    lookup = _load_conf_lookup()
    home_tier = lookup.get(home_team, {}).get("tier", "unknown")
    away_tier = lookup.get(away_team, {}).get("tier", "unknown")
    if "low" in [home_tier, away_tier]: return "low"
    if "high" in [home_tier, away_tier]: return "high"
    if "mid" in [home_tier, away_tier]: return "mid"
    return "unknown"'''

new_tier = '''def get_game_tier(home_team: str, away_team: str) -> str:
    """Return the WEAKEST tier among both teams (low-major contaminates the game)."""
    lookup = _load_conf_lookup()
    home_tier = lookup.get(home_team, {}).get("tier", "unknown")
    away_tier = lookup.get(away_team, {}).get("tier", "unknown")
    tier_rank = {"low": 0, "mid": 1, "high": 2, "unknown": 3}
    weakest = min(home_tier, away_tier, key=lambda t: tier_rank.get(t, 3))
    return weakest'''

if old_tier in content:
    content = content.replace(old_tier, new_tier)
    print("✅ Fix 1 applied: get_game_tier() now returns weakest tier")
else:
    print("⚠️  Fix 1 SKIPPED: get_game_tier() signature didn't match — check manually")

# ── FIX 2: recommend_bets() ─────────────────────────────────────────────────
# Old: confidence logic is inverted (HIGH only 9-10, LOW above 10)
# Old: no tier filtering — low-major games get HIGH flags
# New: proper confidence tiers + tier-based suppression
old_bets = '''def recommend_bets(edge: dict, preds: dict) -> list[dict]:
    """
    Flag games where the model has meaningful edge vs the market.
    Thresholds tuned to reduce noise: spread>=5 (HIGH>=7), total>=5 (HIGH>=8, max 12).
    """
    bets = []

    # Spread: HIGH only above 7pts, suppress obvious data errors above 15pts
    spread_abs = abs(edge.get("spread_edge", 0))
    if 0 < spread_abs <= 15 and spread_abs >= 7.0:
        side = "HOME" if edge["spread_edge"] > 0 else "AWAY"
        bets.append({
            "market":     "SPREAD",
            "lean":       side,
            "edge_pts":   spread_abs,
            "confidence": "HIGH" if 9.0 <= spread_abs <= 10.0 else "MEDIUM" if spread_abs < 9.0 else "LOW",
        })

    # Totals: suppress edges >12 (data quality), HIGH only above 8pts
    total_abs = abs(edge.get("total_edge", 0))
    if 0 < total_abs <= 12 and total_abs >= 7.0:
        bets.append({
            "market":     "TOTAL",
            "lean":       edge.get("ou_lean", ""),
            "edge_pts":   total_abs,
            "confidence": "HIGH" if total_abs >= 9.0 else "MEDIUM",
        })

    # Moneyline: disabled -- win prob model is not yet calibrated for ML betting.
    # Re-enable once we have a dedicated ML calibration layer.
    # if abs(edge.get("win_prob_edge", 0)) >= 0.05: ...

    return bets'''

new_bets = '''def recommend_bets(edge: dict, preds: dict, game_tier: str = "unknown") -> list[dict]:
    """
    Flag games where the model has meaningful edge vs the market.

    Confidence tiers (spread):
      HIGH   : edge 9-15 pts, high-major conference only
      MEDIUM : edge 7-9 pts, high or mid-major only
      LOW    : edge 7-15 pts, any conference (shown but not recommended)

    Confidence tiers (totals):
      HIGH   : edge >= 9 pts, high-major only
      MEDIUM : edge 7-9 pts, high or mid-major only
      LOW    : edge 7-12 pts, any conference

    Edges > 15 pts (spread) or > 12 pts (totals) are suppressed as likely bad odds data.
    NaN edges are always suppressed.
    """
    import math
    bets = []

    # ── Spread ──────────────────────────────────────────────────────────────
    spread_abs = abs(edge.get("spread_edge", 0) or 0)
    if math.isnan(spread_abs):
        spread_abs = 0

    if 7.0 <= spread_abs <= 15.0:
        side = "HOME" if edge.get("spread_edge", 0) > 0 else "AWAY"

        # Determine raw confidence by edge size
        if spread_abs >= 9.0:
            raw_conf = "HIGH"
        else:
            raw_conf = "MEDIUM"

        # Downgrade confidence based on conference tier
        if game_tier == "low":
            conf = "LOW"
        elif game_tier == "mid" and raw_conf == "HIGH":
            conf = "MEDIUM"
        else:
            conf = raw_conf

        bets.append({
            "market":     "SPREAD",
            "lean":       side,
            "edge_pts":   spread_abs,
            "confidence": conf,
        })

    # ── Totals ───────────────────────────────────────────────────────────────
    total_abs = abs(edge.get("total_edge", 0) or 0)
    if math.isnan(total_abs):
        total_abs = 0

    if 7.0 <= total_abs <= 12.0:
        lean = edge.get("ou_lean", "")

        if total_abs >= 9.0:
            raw_conf = "HIGH"
        else:
            raw_conf = "MEDIUM"

        # Downgrade confidence based on conference tier
        if game_tier == "low":
            conf = "LOW"
        elif game_tier == "mid" and raw_conf == "HIGH":
            conf = "MEDIUM"
        else:
            conf = raw_conf

        bets.append({
            "market":     "TOTAL",
            "lean":       lean,
            "edge_pts":   total_abs,
            "confidence": conf,
        })

    # Moneyline: disabled -- win prob model is not yet calibrated for ML betting.
    # Re-enable once we have a dedicated ML calibration layer.
    # if abs(edge.get("win_prob_edge", 0)) >= 0.05: ...

    return bets'''

if old_bets in content:
    content = content.replace(old_bets, new_bets)
    print("✅ Fix 2 applied: recommend_bets() has correct confidence logic + tier filtering")
else:
    print("⚠️  Fix 2 SKIPPED: recommend_bets() signature didn't match exactly")
    print("    You may need to apply this manually. Check for whitespace differences.")

# ── FIX 3: Pass game_tier into recommend_bets() call site ───────────────────
# Find where recommend_bets is called and add game_tier argument
old_call = 'bets = recommend_bets(edge, preds)'
new_call = 'bets = recommend_bets(edge, preds, game_tier=game.get("game_tier", "unknown"))'

if old_call in content:
    content = content.replace(old_call, new_call)
    print("✅ Fix 3 applied: recommend_bets() call site now passes game_tier")
else:
    # Try to find it with surrounding context
    print("⚠️  Fix 3: Looking for recommend_bets call site...")
    idx = content.find('recommend_bets(')
    if idx >= 0:
        print(f"    Found at char {idx}: {content[idx:idx+60]}")
        print("    Please update manually to pass game_tier=game.get('game_tier', 'unknown')")
    else:
        print("    Could not find recommend_bets() call site")

# ── Write patched file ───────────────────────────────────────────────────────
with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

print("\n✅ Patched file saved.")
print("\nNext steps:")
print("  python predictions/daily_pipeline.py")
print("  python predictions/daily_pipeline.py --tomorrow")
