"""
find_model_and_show_fixes.py
============================
Run from D:\\ncaab_model:
    python find_model_and_show_fixes.py

Finds the model files and prints the EXACT lines to change in tournament_predict.py
"""
import sys, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ── Find model files ─────────────────────────────────────────────────────────
print("="*70)
print("FINDING MODEL FILES")
print("="*70)
for pkl in ROOT.rglob("*.pkl"):
    print(f"  {pkl}")
for pkl in ROOT.rglob("*.joblib"):
    print(f"  {pkl}")

# ── Read tournament_predict.py around the alias section ──────────────────────
print("\n" + "="*70)
print("CURRENT ALIAS SECTION (lines 90-140)")
print("="*70)
tp = ROOT / "tournament_predict.py"
lines = tp.read_text(encoding="utf-8", errors="replace").split("\n")
for i, line in enumerate(lines[89:145], start=90):
    print(f"  {i:4d}: {line}")

# ── Show what normalize_team does ────────────────────────────────────────────
print("\n" + "="*70)
print("normalize_team FUNCTION")
print("="*70)
src = "\n".join(lines)
# Find normalize_team definition
match = re.search(r"def normalize_team.*?(?=\ndef |\Z)", src, re.DOTALL)
if match:
    print(match.group(0)[:800])

# ── Show build_game_features to understand feature column usage ───────────────
print("\n" + "="*70)
print("build_game_features FUNCTION (first 80 lines)")
print("="*70)
match2 = re.search(r"def build_game_features.*?(?=\ndef |\Z)", src, re.DOTALL)
if match2:
    bgf = match2.group(0)
    bgf_lines = bgf.split("\n")
    for line in bgf_lines[:80]:
        print(f"  {line}")
    if len(bgf_lines) > 80:
        print(f"  ... ({len(bgf_lines)-80} more lines)")

# ── Show predict_tournament_game to understand the full flow ─────────────────
print("\n" + "="*70)
print("predict_tournament_game FUNCTION (first 60 lines)")
print("="*70)
match3 = re.search(r"def predict_tournament_game.*?(?=\ndef |\Z)", src, re.DOTALL)
if match3:
    ptg = match3.group(0)
    for line in ptg.split("\n")[:60]:
        print(f"  {line}")