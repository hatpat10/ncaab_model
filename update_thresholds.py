"""
update_thresholds.py

Raises edge thresholds for MEDIUM confidence bets before March Madness.
Based on backtesting: HIGH spreads = 64% (profitable), MEDIUM = 47% (below break-even).

Changes:
  - HIGH spread threshold:   7 pts  → 8 pts  (tighten slightly, protect the edge)
  - MEDIUM spread threshold: 5 pts  → 7 pts  (was generating too many losing bets)
  - HIGH total threshold:    7 pts  → 9 pts  (totals underperforming at HIGH, raise bar)
  - MEDIUM total threshold:  5 pts  → 8 pts  (totals variance too high at MEDIUM)

Run: python update_thresholds.py
Then re-run daily_pipeline.py to verify output looks right.
"""

import re
from pathlib import Path

PIPELINE = Path("predictions/daily_pipeline.py")

if not PIPELINE.exists():
    print(f"ERROR: {PIPELINE} not found. Run from D:\\ncaab_model root.")
    exit(1)

text = PIPELINE.read_text(encoding="utf-8")
original = text

# ── Print current threshold values before patching ───────────────────────────
print("=" * 60)
print("CURRENT threshold lines in daily_pipeline.py:")
print("=" * 60)
for i, line in enumerate(text.splitlines(), 1):
    if any(kw in line for kw in [
        "HIGH_SPREAD", "MEDIUM_SPREAD", "HIGH_TOTAL", "MEDIUM_TOTAL",
        "high_threshold", "medium_threshold", "spread_threshold", "total_threshold",
        "edge_threshold", "BET_THRESHOLD", "SPREAD_HIGH", "SPREAD_MED",
        "TOTAL_HIGH", "TOTAL_MED", "confidence", "HIGH =", "MEDIUM ="
    ]):
        print(f"  Line {i:4d}: {line.rstrip()}")

print()

# ── Strategy: patch known constant patterns ───────────────────────────────────
# The pipeline uses one of two common patterns; we try both.

PATCHES = [
    # Pattern A: named constants (most likely given project structure)
    ("HIGH_SPREAD_THRESHOLD = 7",   "HIGH_SPREAD_THRESHOLD = 8"),
    ("HIGH_SPREAD_THRESHOLD = 6",   "HIGH_SPREAD_THRESHOLD = 8"),
    ("MEDIUM_SPREAD_THRESHOLD = 5", "MEDIUM_SPREAD_THRESHOLD = 7"),
    ("MEDIUM_SPREAD_THRESHOLD = 4", "MEDIUM_SPREAD_THRESHOLD = 7"),
    ("HIGH_TOTAL_THRESHOLD = 7",    "HIGH_TOTAL_THRESHOLD = 9"),
    ("HIGH_TOTAL_THRESHOLD = 6",    "HIGH_TOTAL_THRESHOLD = 9"),
    ("MEDIUM_TOTAL_THRESHOLD = 5",  "MEDIUM_TOTAL_THRESHOLD = 8"),
    ("MEDIUM_TOTAL_THRESHOLD = 4",  "MEDIUM_TOTAL_THRESHOLD = 8"),

    # Pattern B: inline comparisons — abs(edge) >= N
    ("abs(spread_edge) >= 7 else 'MEDIUM'", "abs(spread_edge) >= 8 else 'MEDIUM'"),
    ("abs(spread_edge) >= 5 else 'LOW'",    "abs(spread_edge) >= 7 else 'LOW'"),
    ("abs(total_edge) >= 7 else 'MEDIUM'",  "abs(total_edge) >= 9 else 'MEDIUM'"),
    ("abs(total_edge) >= 5 else 'LOW'",     "abs(total_edge) >= 8 else 'LOW'"),

    # Pattern C: dict / tuple thresholds
    ("'HIGH': 7,",   "'HIGH': 8,"),
    ("'MEDIUM': 5,", "'MEDIUM': 7,"),
    ("'HIGH': 6,",   "'HIGH': 8,"),
    ("'MEDIUM': 4,", "'MEDIUM': 7,"),
]

patched_count = 0
for old, new in PATCHES:
    if old in text:
        text = text.replace(old, new)
        print(f"  PATCHED: '{old}' → '{new}'")
        patched_count += 1

print()

if patched_count == 0:
    print("WARNING: No automatic patches applied.")
    print("The threshold pattern in your pipeline didn't match the expected forms.")
    print()
    print("MANUAL EDIT INSTRUCTIONS:")
    print("Open predictions/daily_pipeline.py and find the function that assigns")
    print("'HIGH', 'MEDIUM', or 'LOW' confidence to bets.")
    print()
    print("Apply these changes:")
    print("  Spread HIGH threshold:  raise to 8 pts")
    print("  Spread MEDIUM threshold: raise to 7 pts")
    print("  Total HIGH threshold:   raise to 9 pts")
    print("  Total MEDIUM threshold: raise to 8 pts")
    print()
    print("Search for: HIGH_SPREAD, MEDIUM_SPREAD, HIGH_TOTAL, MEDIUM_TOTAL")
    print("Or search for the string 'MEDIUM' near edge comparison logic.")
else:
    # Write patched file
    PIPELINE.write_text(text, encoding="utf-8")
    print(f"SUCCESS: {patched_count} patch(es) applied to {PIPELINE}")
    print()
    print("Verify by running:")
    print("  python -m predictions.daily_pipeline --date 2026-03-11")
    print("Then compare how many HIGH/MEDIUM bets are flagged vs. before.")
    print("Expect fewer MEDIUM flags, same or slightly fewer HIGH flags.")
    print()

    # Show diff summary
    orig_lines = original.splitlines()
    new_lines = text.splitlines()
    print("DIFF:")
    for i, (ol, nl) in enumerate(zip(orig_lines, new_lines), 1):
        if ol != nl:
            print(f"  Line {i}: - {ol.strip()}")
            print(f"  Line {i}: + {nl.strip()}")

print()
print("THRESHOLD RATIONALE:")
print("  Spread HIGH  8+: preserves the 8-4 (64%) winning tier")
print("  Spread MED   7+: cuts losing MEDIUM plays (was 47%, below break-even)")
print("  Total HIGH   9+: totals had too much conference tournament noise at 7+")
print("  Total MED    8+: raise bar until more tournament data accumulated")
print()
print("Revisit thresholds after 20+ tournament games graded.")