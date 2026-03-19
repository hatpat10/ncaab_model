"""
patch_tournament_predict.py
===========================
Run from D:\\ncaab_model:
    python patch_tournament_predict.py

Fixes three things in tournament_predict.py:
  1. Backwards aliases (tennessee state, mcneese)
  2. Missing aliases for all problematic teams
  3. Adds a Vegas sanity check to auto-flag suspect predictions
"""
import re, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TP   = ROOT / "tournament_predict.py"
BAK  = ROOT / "tournament_predict.py.bak"

src = TP.read_text(encoding="utf-8")

# ── Backup ───────────────────────────────────────────────────────────────────
shutil.copy2(TP, BAK)
print(f"Backed up to {BAK}")

# ── Fix 1: Replace the entire TOURNAMENT_ALIASES dict ────────────────────────
# The current dict has backwards entries. Replace the whole thing.

old_aliases = re.search(
    r"TOURNAMENT_ALIASES\s*:\s*dict\[str,\s*str\]\s*=\s*\{[^}]+\}",
    src, re.DOTALL
)

if not old_aliases:
    print("ERROR: Could not find TOURNAMENT_ALIASES dict — check manually")
    exit(1)

print(f"Found TOURNAMENT_ALIASES at chars {old_aliases.start()}-{old_aliases.end()}")
print("Old content:")
print(old_aliases.group(0))

NEW_ALIASES = '''TOURNAMENT_ALIASES: dict[str, str] = {
    # ── Bracket name → feature matrix team_id ──────────────────────────────
    # Run: grep team_id in feature_matrix to find exact keys
    # Format: "bracket name (lowercased)" -> "matrix team_id"

    # California Baptist
    "ca baptist":            "california_baptist",
    "cal baptist":           "california_baptist",
    "california baptist":    "california_baptist",

    # North Dakota State
    "n dakota st":           "n_dakota_st",
    "north dakota st":       "n_dakota_st",
    "north dakota state":    "n_dakota_st",
    "n_dakota_st":           "n_dakota_st",

    # Tennessee State — KEY FIX: was wrongly mapped to "tennessee st"
    # which normalize_team() resolves to "tennessee" (a completely different team)
    "tennessee state":       "tennessee_state",
    "tennessee st":          "tennessee_state",
    "tennessee_st":          "tennessee_state",

    # McNeese — KEY FIX: was wrongly mapped to "mcneese state" (doesn't exist)
    # Correct matrix key is "mcneese"
    "mcneese state":         "mcneese",
    "mcneese_state":         "mcneese",
    "mcneese":               "mcneese",

    # LIU / Long Island
    "long island":           "liu",
    "liu":                   "liu",

    # Kennesaw State
    "kennesaw st":           "kennesaw_st",
    "kennesaw state":        "kennesaw_st",

    # Hawaii
    "hawaii":                "hawai'i",

    # Miami Ohio
    "miami oh":              "miami (oh)",
    "miami (ohio)":          "miami (oh)",
    "miami_oh":              "miami (oh)",

    # Queens
    "queens":                "queens (nc)",
    "queens nc":             "queens (nc)",

    # Saint Mary's
    "saint mary's":          "saint mary's (ca)",
    "saint marys":           "saint mary's (ca)",
    "st marys":              "saint mary's (ca)",
    "st mary's":             "saint mary's (ca)",

    # Wright State
    "wright state":          "wright_state",
    "wright st":             "wright_state",
    "wright_st":             "wright_state",

    # Prairie View A&M
    "prairie view a&m":      "prairie_view",
    "prairie view":          "prairie_view",

    # Other common bracket vs. matrix mismatches
    "unc":                   "north_carolina",
    "north carolina":        "north_carolina",
    "uconn":                 "uconn",
    "connecticut":           "uconn",
    "ohio st":               "ohio_st",
    "ohio state":            "ohio_st",
    "michigan st":           "michigan_st",
    "michigan state":        "michigan_st",
    "penn":                  "penn",
    "pennsylvania":          "penn",
    "vcu":                   "vcu",
}'''

src = src[:old_aliases.start()] + NEW_ALIASES + src[old_aliases.end():]
print("\nReplaced TOURNAMENT_ALIASES ✓")

# ── Fix 2: Add Vegas sanity check after preds are built ─────────────────────
# Find where predict_game is called and results are returned
# We'll add a sanity check that flags predictions where model deviates
# from Vegas by > 18 pts (impossible for a real calibrated model)

SANITY_CHECK = '''
    # ── Vegas sanity check ────────────────────────────────────────────────────
    # If the model spread deviates from Vegas by > 18 pts, something is wrong
    # (bad feature data, name collision, etc.). Flag it so it's excluded from
    # confident plays but still show the Vegas line for reference.
    if odds_row is not None:
        vegas_spread = odds_row.get("home_spread") or odds_row.get("spread")
        if vegas_spread is not None:
            try:
                model_spread = preds.get("spread", 0) or 0
                deviation = abs(float(model_spread) - float(vegas_spread))
                if deviation > 18:
                    log.warning(
                        "  ⚠️  SANITY CHECK FAILED: model spread %.1f vs Vegas %.1f "
                        "(deviation %.1f pts > 18 pt threshold) — flagging as suspect",
                        model_spread, vegas_spread, deviation
                    )
                    preds["suspect"] = True
                    preds["suspect_reason"] = (
                        f"Model spread {model_spread:+.1f} deviates {deviation:.1f} pts "
                        f"from Vegas {vegas_spread:+.1f}"
                    )
            except (TypeError, ValueError):
                pass
'''

# Find "# Run base models" and insert sanity check after predict_game call
# Look for where preds = predict_game(...) is followed by the return block
preds_call = re.search(
    r"(preds\s*=\s*predict_game\(features,\s*models\))",
    src
)
if preds_call:
    insert_pos = preds_call.end()
    src = src[:insert_pos] + "\n" + SANITY_CHECK + src[insert_pos:]
    print("Added Vegas sanity check ✓")
else:
    print("WARNING: Could not find 'preds = predict_game(features, models)' — sanity check NOT added")

# ── Fix 3: Make suspect=True suppress HIGH/MEDIUM confidence flags ───────────
# Find where bet flags are set and add a suspect guard
# Look for the confidence/flag assignment block

suspect_guard = '''
    # ── Suppress flags on suspect predictions ──────────────────────────────
    if result.get("suspect"):
        result["confidence"] = None
        result["bet_type"]   = None
        log.warning("  ⚠️  Prediction flagged as suspect — confidence suppressed")
'''

# Find the return statement in predict_tournament_game to insert before it
# The result dict is built and returned — find "return result" or "return {"
return_match = re.search(
    r"(\s+result\s*\[.suspect.\]\s*=\s*True)",
    src
)
if not return_match:
    # Try to find where result is assembled and returned
    result_return = re.search(
        r"(\s+return\s+\{[^}]{0,200}\"home_team\")",
        src, re.DOTALL
    )
    if result_return:
        insert_pos = result_return.start()
        src = src[:insert_pos] + "\n" + suspect_guard + src[insert_pos:]
        print("Added suspect flag suppression ✓")
    else:
        print("WARNING: Could not find result return block — suspect suppression NOT added")
        print("  (You can manually add: if result.get('suspect'): result['confidence'] = None)")

# ── Write the patched file ───────────────────────────────────────────────────
TP.write_text(src, encoding="utf-8")
print(f"\nPatched {TP}")

# ── Verify the fix ───────────────────────────────────────────────────────────
print("\n" + "="*70)
print("VERIFICATION — Reading back alias section")
print("="*70)
src_check = TP.read_text(encoding="utf-8")
alias_check = re.search(r"TOURNAMENT_ALIASES.*?\}", src_check, re.DOTALL)
if alias_check:
    print(alias_check.group(0)[:1500])

print("\n" + "="*70)
print("NEXT STEPS")
print("="*70)
print("""
1. Run predictions:
     python tournament_predict.py --bracket bracket.json

2. Check the formerly-suspect matchups:
     - Tennessee State vs Iowa State  (should now show Iowa St as ~-24 favorite)
     - McNeese vs Vanderbilt          (should now show Vanderbilt favored)
     - Cal Baptist vs Kansas          (should now show Kansas as clear favorite)

3. If any HIGH confidence flags remain on these games, they're legitimately
   flagged by the model on real features — not data artifacts.
""")