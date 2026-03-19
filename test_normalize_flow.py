"""
test_normalize_flow.py
======================
python test_normalize_flow.py
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from predictions.daily_pipeline import normalize_team
import tournament_predict as tp
import pandas as pd

fm = pd.read_parquet(ROOT / "data" / "processed" / "feature_matrix_full.parquet")

# Simulate exactly what predict_tournament_game does:
# 1. alias lookup
# 2. normalize_team on the result
# 3. matrix lookup

bracket_names = [
    "Siena", "Duke", "Furman", "UConn", "LIU", "Arizona",
    "Kennesaw St", "Gonzaga", "Queens", "Purdue", "Lehigh", "Florida",
    "Penn", "Illinois", "Idaho", "Houston", "Wright State", "Virginia",
    "Tennessee State", "Iowa State",
]

print(f"{'BRACKET':<22} {'AFTER_ALIAS':<22} {'AFTER_NORMALIZE':<22} {'IN_MATRIX'}")
print("-"*90)
for name in bracket_names:
    alias = tp.TOURNAMENT_ALIASES.get(name.lower(), name)
    norm  = normalize_team(alias)
    rows  = fm[fm["team_id"] == norm]
    torvik_rows = rows[rows["t_adj_o"].notna()] if len(rows) else rows
    
    status = f"{len(rows)} rows ({len(torvik_rows)} w/torvik)"
    flag = " <-- PROBLEM" if len(rows) == 0 or len(torvik_rows) == 0 else ""
    print(f"  {name:<22} {alias:<22} {norm:<22} {status}{flag}")