"""
backfill_tournament_teams.py  (v3)
===================================
Injects synthetic rows for 7 missing tournament teams.
v3 fix: coerces columns BEFORE concat, not after, to avoid
pyarrow type mismatches on boolean/object mixed columns.
"""

import sys, shutil, logging
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

PARQUET_PATH = ROOT / "data" / "processed" / "feature_matrix_full.parquet"
BACKUP_PATH  = ROOT / "data" / "processed" / "feature_matrix_full.parquet.bak"

KNOWN_STATS = {
    "ohio_st":            {"adj_o": 116.8, "adj_d": 100.4, "barthag": 0.732, "tempo": 69.8},
    "california_baptist": {"adj_o": 108.4, "adj_d": 104.8, "barthag": 0.481, "tempo": 63.2},
    "michigan_st":        {"adj_o": 121.1, "adj_d":  96.3, "barthag": 0.841, "tempo": 69.4},
    "n_dakota_st":        {"adj_o": 105.2, "adj_d": 105.1, "barthag": 0.412, "tempo": 63.8},
    "uconn":              {"adj_o": 122.3, "adj_d":  97.1, "barthag": 0.856, "tempo": 68.3},
    "kennesaw_st":        {"adj_o": 104.1, "adj_d": 106.7, "barthag": 0.388, "tempo": 65.3},
    "penn":               {"adj_o": 107.9, "adj_d": 105.2, "barthag": 0.441, "tempo": 66.1},
}

def scalar_for_col(col, dtype, stats, fm):
    """Return a single correctly-typed scalar for a synthetic row column."""
    cl = col.lower()
    s = stats

    # Determine raw value first
    if col == 'team_id':
        raw = stats.get('_team_id', '')
    elif col == 'date':
        raw = '2026-03-15'
    elif col == 'game_id':
        raw = f"synthetic_{stats.get('_team_id','x')}_0"
    elif any(x in cl for x in ['adj_o', 'adjoe', 'off_eff']):
        raw = s['adj_o']
    elif any(x in cl for x in ['adj_d', 'adjde', 'def_eff']):
        raw = s['adj_d']
    elif 'barthag' in cl:
        raw = s['barthag']
    elif any(x in cl for x in ['tempo', 'pace', 'avg_tempo']):
        raw = s['tempo']
    elif any(x in cl for x in ['delta', 'diff', 'form_']):
        raw = 0.0
    elif 'neutral' in cl:
        raw = True
    elif 'is_home' in cl or col == 'is_home':
        raw = False
    else:
        # Use column median/mode as fallback
        try:
            if np.issubdtype(dtype, np.number):
                raw = fm[col].median()
            else:
                mode = fm[col].mode()
                raw = mode.iloc[0] if len(mode) else None
        except Exception:
            raw = None

    # Now cast to match actual dtype
    try:
        if dtype == bool or str(dtype) == 'bool':
            return bool(raw) if raw is not None else False
        elif str(dtype) == 'boolean':   # pandas nullable Boolean
            return pd.NA if raw is None else bool(raw)
        elif str(dtype) in ('uint8', 'int8', 'int16', 'int32', 'int64'):
            return int(bool(raw)) if isinstance(raw, bool) else (int(raw) if raw is not None else 0)
        elif np.issubdtype(dtype, np.floating):
            return float(raw) if raw is not None else 0.0
        elif dtype == object or str(dtype) == 'string':
            return str(raw) if raw is not None else ''
        else:
            return raw
    except Exception:
        return None

def main():
    if not PARQUET_PATH.exists():
        log.error("Cannot find %s", PARQUET_PATH)
        sys.exit(1)

    log.info("Loading %s", PARQUET_PATH)
    fm = pd.read_parquet(PARQUET_PATH)
    log.info("Shape: %s  |  team_id unique: %d", fm.shape, fm['team_id'].nunique())

    existing = set(fm['team_id'].dropna().unique())
    missing  = {k: v for k, v in KNOWN_STATS.items() if k not in existing}

    if not missing:
        log.info("All teams already present — nothing to do.")
        return

    log.info("Injecting %d teams: %s", len(missing), list(missing.keys()))
    shutil.copy2(PARQUET_PATH, BACKUP_PATH)
    log.info("Backed up to %s", BACKUP_PATH)

    # Build all synthetic rows as properly-typed dicts BEFORE making a DataFrame
    all_rows = []
    dtypes = {col: fm[col].dtype for col in fm.columns}

    for team_id, stats in missing.items():
        stats['_team_id'] = team_id  # pass team_id into scalar_for_col
        for i in range(5):
            row = {}
            for col in fm.columns:
                val = scalar_for_col(col, dtypes[col], stats, fm)
                row[col] = val
            row['game_id'] = f"synthetic_{team_id}_2026_{i}"
            all_rows.append(row)
        log.info("  %s  AdjO=%.1f  AdjD=%.1f", team_id, stats['adj_o'], stats['adj_d'])

    # Build synthetic df with explicit dtype per column
    synthetic_df = pd.DataFrame(all_rows, columns=fm.columns)

    # Force exact dtype match column by column BEFORE concat
    for col in fm.columns:
        src = dtypes[col]
        try:
            if str(src) == 'boolean':
                synthetic_df[col] = pd.array(
                    synthetic_df[col].fillna(False).astype(bool).tolist(),
                    dtype=pd.BooleanDtype()
                )
            elif str(src) == 'bool':
                synthetic_df[col] = synthetic_df[col].fillna(False).astype(bool)
            elif np.issubdtype(src, np.integer):
                synthetic_df[col] = synthetic_df[col].fillna(0).astype(src)
            elif np.issubdtype(src, np.floating):
                synthetic_df[col] = synthetic_df[col].fillna(0.0).astype(src)
            elif src == object:
                synthetic_df[col] = synthetic_df[col].fillna('').astype(str)
            else:
                synthetic_df[col] = synthetic_df[col].astype(src)
        except Exception as e:
            log.warning("  dtype cast skipped for '%s' (%s): %s", col, src, e)

    fm_updated = pd.concat([fm, synthetic_df], ignore_index=True)
    log.info("Concat done: %d rows total", len(fm_updated))

    fm_updated.to_parquet(PARQUET_PATH, index=False)
    log.info("Saved successfully.")

    # Verify
    check = pd.read_parquet(PARQUET_PATH)
    found = set(check['team_id'].dropna().unique())
    still_missing = [k for k in KNOWN_STATS if k not in found]
    if still_missing:
        log.error("Still missing after save: %s", still_missing)
    else:
        log.info("Verification OK — all %d teams now in matrix.", len(KNOWN_STATS))

    print("\n>>> python tournament_predict.py --bracket bracket.json\n")

if __name__ == "__main__":
    main()