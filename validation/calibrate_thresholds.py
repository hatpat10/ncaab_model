"""
validation/calibrate_thresholds.py
Updates pipeline thresholds based on backtest findings.
Run after backtester.py to apply recommended settings.

Usage:
    python -m validation.calibrate_thresholds --backtest-report reports/backtest_report.json
    python -m validation.calibrate_thresholds --spread-threshold 6 --total-max-edge 10
"""

import json
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

CONFIG_PATH = Path("config/pipeline_config.json")
DEFAULT_CONFIG = {
    "spread": {
        "high_edge_threshold": 6.0,    # Flag plays with |edge| >= this
        "display_threshold": 2.0,      # Minimum edge to show in output
        "suppress_above": 20.0,        # Likely data error if edge > this
    },
    "totals": {
        "high_edge_threshold": 8.0,
        "display_threshold": 3.0,
        "suppress_above": 12.0,        # 15pt edges are almost always bad data
    },
    "moneyline": {
        "enabled": False,              # Disabled until win prob model calibrated
        "min_edge_pct": 5.0,           # Minimum % edge vs implied prob
    },
    "output": {
        "show_nan_odds": False,        # Suppress +nan displays
        "max_flags_per_slate": 20,     # Warn if > this many flagged plays
    },
    "backtesting": {
        "last_run": None,
        "games_graded": 0,
        "spread_mae": None,
        "spread_ats": None,
        "recommended_spread_threshold": None,
    }
}


def load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            cfg = json.load(f)
        # Merge with defaults for any missing keys
        for section, values in DEFAULT_CONFIG.items():
            if section not in cfg:
                cfg[section] = values
            else:
                for k, v in values.items():
                    if k not in cfg[section]:
                        cfg[section][k] = v
        return cfg
    return DEFAULT_CONFIG.copy()


def save_config(cfg: dict):
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)
    log.info(f"Config saved to {CONFIG_PATH}")


def apply_backtest_report(cfg: dict, report_path: str) -> dict:
    with open(report_path) as f:
        report = json.load(f)

    spread = report.get("spread", {})
    totals = report.get("totals", {})
    recommended = spread.get("recommended_threshold")

    if recommended:
        old = cfg["spread"]["high_edge_threshold"]
        cfg["spread"]["high_edge_threshold"] = recommended
        log.info(f"Spread threshold: {old} → {recommended}")

    # Update backtest metadata
    cfg["backtesting"].update({
        "last_run": report.get("generated_at"),
        "games_graded": report.get("games_graded", 0),
        "spread_mae": spread.get("mae"),
        "spread_ats": spread.get("overall_ats"),
        "recommended_spread_threshold": recommended,
    })

    # Auto-suppress totals edges above observed noise floor
    if totals.get("mae") and totals["mae"] > 12:
        cfg["totals"]["suppress_above"] = 10.0
        log.warning(f"Totals MAE={totals['mae']:.1f} is high — suppressing edges >10")

    return cfg


def print_current_config():
    cfg = load_config()
    print("\nCurrent Pipeline Config:")
    print(json.dumps(cfg, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backtest-report", help="Path to backtest JSON report")
    parser.add_argument("--spread-threshold", type=float, help="Override spread HIGH threshold")
    parser.add_argument("--total-max-edge", type=float, help="Override totals suppress_above")
    parser.add_argument("--show", action="store_true", help="Print current config and exit")
    args = parser.parse_args()

    if args.show:
        print_current_config()
        exit(0)

    cfg = load_config()

    if args.backtest_report:
        cfg = apply_backtest_report(cfg, args.backtest_report)

    if args.spread_threshold is not None:
        old = cfg["spread"]["high_edge_threshold"]
        cfg["spread"]["high_edge_threshold"] = args.spread_threshold
        log.info(f"Spread threshold manually set: {old} → {args.spread_threshold}")

    if args.total_max_edge is not None:
        cfg["totals"]["suppress_above"] = args.total_max_edge
        log.info(f"Totals suppress_above set to {args.total_max_edge}")

    # Apply immediate fixes from Phase 6 known issues
    cfg["output"]["show_nan_odds"] = False
    cfg["totals"]["suppress_above"] = min(cfg["totals"].get("suppress_above", 12), 12.0)

    save_config(cfg)
    print_current_config()