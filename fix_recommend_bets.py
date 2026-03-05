"""
Surgically replaces just the recommend_bets function signature and body.
"""
import re

path = 'predictions/daily_pipeline.py'
with open(path, encoding='utf-8') as f:
    content = f.read()

# Find the function by locating its def line and replacing until the next top-level def
start_marker = 'def recommend_bets('
end_marker = '\ndef save_predictions_to_db('  # the function that follows it

start_idx = content.find(start_marker)
end_idx = content.find(end_marker)

if start_idx == -1:
    print("ERROR: Could not find recommend_bets function")
    exit(1)
if end_idx == -1:
    print("ERROR: Could not find end boundary (save_predictions_to_db)")
    exit(1)

print(f"Found recommend_bets at chars {start_idx}-{end_idx}")
print("Current function:")
print(content[start_idx:end_idx])
print("\n--- Replacing with new version ---\n")

new_func = '''def recommend_bets(edge: dict, preds: dict, game_tier: str = "unknown") -> list[dict]:
    """
    Flag games where the model has meaningful edge vs the market.

    Confidence tiers (spread):
      HIGH   : edge 9-15 pts, high-major conference only
      MEDIUM : edge 7-9 pts, high or mid-major only
      LOW    : edge 7-15 pts, any conference (shown but not recommended)

    Edges > 15 pts (spread) or > 12 pts (totals) are suppressed as likely bad odds data.
    NaN edges are always suppressed.
    """
    import math
    bets = []

    # -- Spread ---------------------------------------------------------------
    raw_spread = edge.get("spread_edge", 0) or 0
    spread_abs = abs(raw_spread)
    if math.isnan(spread_abs):
        spread_abs = 0

    if 7.0 <= spread_abs <= 15.0:
        side = "HOME" if raw_spread > 0 else "AWAY"
        raw_conf = "HIGH" if spread_abs >= 9.0 else "MEDIUM"

        if game_tier == "low":
            conf = "LOW"
        elif game_tier in ("mid", "unknown") and raw_conf == "HIGH":
            conf = "MEDIUM"
        else:
            conf = raw_conf

        bets.append({
            "market":     "SPREAD",
            "lean":       side,
            "edge_pts":   spread_abs,
            "confidence": conf,
        })

    # -- Totals ---------------------------------------------------------------
    raw_total = edge.get("total_edge", 0) or 0
    total_abs = abs(raw_total)
    if math.isnan(total_abs):
        total_abs = 0

    if 7.0 <= total_abs <= 12.0:
        lean = edge.get("ou_lean", "")
        raw_conf = "HIGH" if total_abs >= 9.0 else "MEDIUM"

        if game_tier == "low":
            conf = "LOW"
        elif game_tier in ("mid", "unknown") and raw_conf == "HIGH":
            conf = "MEDIUM"
        else:
            conf = raw_conf

        bets.append({
            "market":     "TOTAL",
            "lean":       lean,
            "edge_pts":   total_abs,
            "confidence": conf,
        })

    # Moneyline: disabled -- win prob model not yet calibrated for ML betting.
    # Re-enable once we have a dedicated ML calibration layer.
    # if abs(edge.get("win_prob_edge", 0)) >= 0.05: ...

    return bets'''

content = content[:start_idx] + new_func + content[end_idx:]

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ recommend_bets() replaced successfully.")

# Verify the new function is there
with open(path, encoding='utf-8') as f:
    verify = f.read()
if 'game_tier: str = "unknown"' in verify:
    print("✅ Verified: new signature is present in file.")
else:
    print("❌ Verification failed - check file manually.")
