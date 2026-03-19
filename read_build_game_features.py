"""
read_build_game_features.py
===========================
Run from D:\\ncaab_model:
    python read_build_game_features.py

Reads and prints the full build_game_features function so we know
exactly what features it computes and where it fails for sparse teams.
Also prints what predict_game does with the output.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent

pipeline_path = ROOT / "predictions" / "daily_pipeline.py"
src = pipeline_path.read_text(encoding="utf-8", errors="replace")
lines = src.split("\n")

# Find build_game_features
print("="*70)
print("build_game_features FULL FUNCTION")
print("="*70)
in_func = False
depth = 0
start_line = None
for i, line in enumerate(lines):
    if "def build_game_features" in line:
        in_func = True
        start_line = i + 1
        depth = 0
    if in_func:
        depth += line.count("(") - line.count(")")
        # Also track def-level indentation
        if start_line and i > start_line and line and not line[0].isspace() and not line.startswith("#"):
            break
        print(f"  {i+1:4d}: {line}")
        if i > start_line + 200:
            print("  ... (truncated at 200 lines)")
            break

print("\n" + "="*70)
print("predict_game FUNCTION")
print("="*70)
in_func = False
for i, line in enumerate(lines):
    if "def predict_game" in line:
        in_func = True
        start_line = i + 1
    if in_func:
        print(f"  {i+1:4d}: {line}")
        if i > start_line + 60:
            print("  ... (truncated)")
            break

# Also check what the 34-item Series looks like
print("\n" + "="*70)
print("WHAT DOES build_game_features RETURN?")
print("="*70)

# Find return statement in build_game_features
bgf_match = re.search(r"def build_game_features.*?(?=\ndef |\Z)", src, re.DOTALL)
if bgf_match:
    bgf = bgf_match.group(0)
    # Find all return statements
    returns = [(m.start(), bgf[m.start():m.start()+200]) for m in re.finditer(r"return ", bgf)]
    print(f"  Return statements ({len(returns)}):")
    for pos, ctx in returns:
        print(f"\n  {ctx[:300]}")
        print("  ---")
    
    # Find all dict/Series keys being set
    feat_keys = re.findall(r'feat\["([^"]+)"\]', bgf)
    feat_keys += re.findall(r"feat\['([^']+)'\]", bgf)
    print(f"\n  Feature keys set in function ({len(set(feat_keys))}):")
    for k in sorted(set(feat_keys)):
        print(f"    {k}")