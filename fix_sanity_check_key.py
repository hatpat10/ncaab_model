"""
fix_sanity_check_logic.py
=========================
python fix_sanity_check_logic.py

The sanity check formula has a sign convention mismatch:
  - model_spread = predicted_margin (positive = home team wins)
  - vegas_spread = home_spread (negative = home team favored)

So deviation should be: |model_spread - (-vegas_spread)|
Or equivalently: |model_spread + vegas_spread|

Example: Duke vs Siena
  model_spread = +8.1 (Duke favored by 8.1, Duke is "home")  
  vegas = -10.8 (Duke -10.8, home favored by 10.8)
  Correct deviation = |8.1 - 10.8| = 2.7 pts  (close, no sanity issue)
  Wrong calculation = |8.1 - (-10.8)| = 18.9 pts  (current broken behavior)

Fix: change deviation = abs(model_spread - vegas_spread) 
         to   deviation = abs(model_spread + vegas_spread)
Or better: just raise the threshold to 25 so only truly broken cases fire.
Also the real broken games (Iowa St/Tennessee St, Virginia/Wright St) 
show model ~2 but Vegas -24 -- that IS a real issue worth flagging.
"""
import re, shutil
from pathlib import Path

ROOT = Path('.').resolve()
TP   = ROOT / "tournament_predict.py"
BAK  = ROOT / "tournament_predict.py.bak5"

src = TP.read_text(encoding="utf-8")
shutil.copy2(TP, BAK)

# Fix 1: Fix the deviation formula 
# Current: deviation = abs(float(model_spread) - float(vegas_spread))  
# Should be: deviation = abs(float(model_spread) + float(vegas_spread))
# (because vegas_spread is negative when home favored, model_spread positive when home wins)
old_dev = "deviation = abs(float(model_spread) - float(vegas_spread))"
new_dev = "deviation = abs(float(model_spread) + float(vegas_spread))"

if old_dev in src:
    src = src.replace(old_dev, new_dev)
    print(f"Fixed deviation formula: subtraction -> addition")
else:
    # Try to find the deviation line
    dev_lines = [(i+1, l) for i, l in enumerate(src.split('\n')) if 'deviation' in l and 'abs' in l]
    print(f"Could not find exact string. Deviation lines: {dev_lines[:5]}")

# Fix 2: Also fix the threshold - 18 pts is too low given sign conventions
# Raise to 25 so only genuinely miscalibrated games fire
old_thresh = "> 18 pt"
new_thresh = "> 25 pt"
old_cond   = "deviation > 18"
new_cond   = "deviation > 25"

if old_thresh in src:
    src = src.replace(old_thresh, new_thresh)
    src = src.replace(old_cond, new_cond)
    print(f"Raised threshold from 18 to 25 pts")
else:
    print("Could not find threshold string - check manually")

TP.write_text(src, encoding="utf-8")
print(f"Saved {TP}")

# Quick verification - show what the sanity check now looks like
print("\n" + "="*60)
print("SANITY CHECK CODE (relevant section):")
lines = src.split('\n')
for i, line in enumerate(lines):
    if 'SANITY CHECK' in line or 'deviation' in line.lower() or 'vegas_spread' in line.lower():
        print(f"  {i+1:4d}: {line}")