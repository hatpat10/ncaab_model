"""
tournament_predict.py
======================
NCAAB ML — March Madness Tournament Predictor
2026 NCAA Tournament

Manually input any two teams and get full model predictions.
Designed for use after Selection Sunday when the bracket is set
but games aren't yet on ESPN's schedule.

Usage:
    # Single matchup
    python tournament_predict.py --home "Duke" --away "Mount St. Mary's"

    # With seeds
    python tournament_predict.py --home "Duke" --away "Mount St. Mary's" --home-seed 1 --away-seed 16

    # With game date (defaults to today)
    python tournament_predict.py --home "Auburn" --away "Louisville" --date 2026-03-20

    # Load full bracket from JSON file (after Selection Sunday)
    python tournament_predict.py --bracket bracket_2026.json

    # Interactive mode — prompts for matchups one at a time
    python tournament_predict.py --interactive

Output:
    predictions/tournament/YYYY-MM-DD_tournament_predictions.json
    predictions/tournament/YYYY-MM-DD_tournament_predictions.csv

Bracket JSON format (bracket_2026.json):
    [
      {"home": "Auburn",   "away": "Alabama State", "home_seed": 1, "away_seed": 16, "region": "East",  "round": 1, "date": "2026-03-20"},
      {"home": "Iowa St",  "away": "Lipscomb",      "home_seed": 2, "away_seed": 15, "region": "East",  "round": 1, "date": "2026-03-20"},
      ...
    ]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ── Add project root to path so we can import from predictions/ ──────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Import the core pipeline functions directly — no duplication
from predictions.daily_pipeline import (
    load_models,
    load_feature_matrix,
    build_game_features,
    predict_game,
    compute_edge,
    recommend_bets,
    fetch_current_odds,
    normalize_team,
    get_game_tier,
    PRED_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

TOURN_DIR = PRED_DIR / "tournament"
TOURN_DIR.mkdir(parents=True, exist_ok=True)

# ── Tournament-specific constants ────────────────────────────────────────────

ROUND_NAMES = {
    1: "Round of 64",
    2: "Round of 32",
    3: "Sweet 16",
    4: "Elite Eight",
    5: "Final Four",
    6: "Championship",
}

REGIONS = ["East", "West", "South", "Midwest"]

# ── Tournament team name aliases ─────────────────────────────────────────────
# Maps bracket JSON names -> feature matrix keys for teams whose name in the
# bracket doesn't match normalize_team(). Add entries here when you see
# "No history found for team" warnings.
TOURNAMENT_ALIASES: dict[str, str] = {
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
    "uconn":                 "connecticut",
    "connecticut":           "connecticut",
    "ohio st":               "ohio_st",
    "ohio state":            "ohio_st",
    "michigan st":           "michigan_st",
    "michigan state":        "michigan_st",
    "penn":                  "pennsylvania",
    "pennsylvania":          "pennsylvania",
    "vcu":                   "vcu",
    # ── Teams where normalize_team() output doesn't match matrix key ──────────
    # St John's — matrix key is 'st_johns'
    "st johns":              "st_johns",
    "st. johns":             "st_johns",
    "st. john's":            "st_johns",
    "saint johns":           "st_johns",

    # Hawaii — normalize returns 'hawaii', matrix has 'hawaii' ✓
    # (already resolves correctly via normalize_team, but add explicit alias)
    "hawai'i":               "hawaii",

    # Miami FL — matrix key is 'miami_fl'
    "miami fl":              "miami_fl",
    "miami (fl)":            "miami_fl",
    "miami florida":         "miami_fl",

    # Queens NC — matrix key is 'queens'
    "queens (nc)":           "queens",
    "queens nc":             "queens",
    "queens (n.c.)":         "queens",

    # Texas A&M — matrix key is 'texas_am'
    "texas a&m":             "texas_am",
    "texas a&amp;m":         "texas_am",

    # Saint Mary's CA — matrix key is 'saint_marys'
    "saint mary's (ca)":     "saint_marys",
    "saint mary's":          "saint_marys",
    "saint marys":           "saint_marys",
    "st. mary's":            "saint_marys",
    "st. marys":             "saint_marys",

    # Miami OH — matrix key is 'miami_oh'
    "miami (oh)":            "miami_oh",
    "miami oh":              "miami_oh",
    "miami ohio":            "miami_oh",
    "miami (ohio)":          "miami_oh",

}

# Seed-based calibration adjustments (post-processing layer on top of ML output)
# Based on historical NCAA tournament ATS data:
# - Double-digit seeds (11-16) cover more often than their seed implies
# - Top seeds (1-3) are slightly over-valued by the market in R64
# These are modest adjustments — the ML model does the heavy lifting.
SEED_ATS_ADJUSTMENT = {
    # (fav_seed, dog_seed) -> points to ADD to underdog (reduce spread)
    (1, 16): -1.5,   # 1 seeds are slightly over-priced vs 16s
    (1, 15): -1.5,
    (2, 15): -1.0,
    (2, 14): -0.5,
    (3, 14): -0.5,
    (3, 11): +1.0,   # 11 seeds historically cover more
    (4, 13): +0.5,
    (5, 12): +1.5,   # 12 seeds are the famous upset pick
    (6, 11): +1.0,
    (7, 10): +0.5,
    (8,  9): 0.0,    # Classic coin flip
}

# Upset probability boost: seeds where model win prob gets a slight bump
# These are empirically derived from 10+ years of tournament data
UPSET_SEED_BOOST = {
    12: +0.04,  # 12 seeds beat 5 seeds ~35% historically, models underrate
    11: +0.03,
    10: +0.02,
    13: +0.02,
    15: +0.01,
}


# ── Core prediction function ─────────────────────────────────────────────────

def predict_tournament_game(
    home_team: str,
    away_team: str,
    home_seed: int | None,
    away_seed: int | None,
    game_date: str,
    round_num: int,
    region: str,
    models: dict,
    feature_matrix: pd.DataFrame,
    odds_df: pd.DataFrame,
) -> dict:
    """
    Predict a single tournament matchup.
    All tournament games are neutral site — hardcoded.
    """
    # Resolve tournament-specific aliases before normalization
    home_team = TOURNAMENT_ALIASES.get(home_team.lower(), home_team)
    away_team = TOURNAMENT_ALIASES.get(away_team.lower(), away_team)

    home_norm = normalize_team(home_team)
    away_norm = normalize_team(away_team)

    log.info("─" * 55)
    log.info("  %s [%s] vs %s [%s] | %s | %s",
             away_team, f"#{away_seed}" if away_seed else "?",
             home_team, f"#{home_seed}" if home_seed else "?",
             region, ROUND_NAMES.get(round_num, f"Round {round_num}"))

    # Find odds if available
    odds_row = None
    if not odds_df.empty:
        match = odds_df[
            (odds_df["home_team"] == home_norm) | (odds_df["away_team"] == home_norm) |
            (odds_df["home_team"] == away_norm) | (odds_df["away_team"] == away_norm)
        ]
        if len(match):
            odds_row = match.iloc[0].to_dict()

    # Build features — ALWAYS neutral site for tournament
    features = build_game_features(
        home_team=home_team,
        away_team=away_team,
        is_neutral=1,           # All tournament games neutral site
        target_date=game_date,
        feature_matrix=feature_matrix,
        odds_row=odds_row,
    )

    if features is None:
        log.warning("  Could not build features for this matchup — check team name aliases")
        return {
            "error": f"Could not build features for {away_team} vs {home_team}",
            "home_team": home_team,
            "away_team": away_team,
        }

    # Run base models
    preds = predict_game(features, models)

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


    # Apply seed-based calibration adjustment to spread prediction
    seed_adj = _get_seed_adjustment(home_seed, away_seed, preds)
    if seed_adj != 0.0:
        original_margin = preds.get("predicted_margin", 0)
        preds["predicted_margin"] = round(original_margin + seed_adj, 1)
        preds["seed_adjustment"] = seed_adj
        log.info("  Seed adjustment applied: %+.1f pts (was %+.1f, now %+.1f)",
                 seed_adj, original_margin, preds["predicted_margin"])

    # Apply upset probability boost to win probability
    upset_boost = _get_upset_boost(home_seed, away_seed, preds)
    if upset_boost != 0.0:
        original_wp = preds.get("home_win_prob", 0.5)
        # Boost goes to the underdog (higher seed number = underdog)
        if away_seed and home_seed and away_seed > home_seed:
            # Away team is underdog — boost away win prob
            preds["away_win_prob"] = round(min(0.95, preds.get("away_win_prob", 0.5) + upset_boost), 3)
            preds["home_win_prob"] = round(1 - preds["away_win_prob"], 3)
        else:
            preds["home_win_prob"] = round(min(0.95, original_wp + upset_boost), 3)
            preds["away_win_prob"] = round(1 - preds["home_win_prob"], 3)
        log.info("  Upset boost applied: %+.3f to underdog win prob", upset_boost)

    # Compute edge vs Vegas
    edge = compute_edge(preds, odds_row)

    # Game tier — tournament is always high-major
    game_tier = "high"

    # Bet recommendations with tournament context
    bets = recommend_bets(edge, preds, game_tier=game_tier)
    for b in bets:
        b["game_tier"] = game_tier
        b["tournament_round"] = ROUND_NAMES.get(round_num, f"Round {round_num}")

    # Determine favored team
    margin = preds.get("predicted_margin", 0)
    favored = home_team if margin >= 0 else away_team
    underdog = away_team if margin >= 0 else home_team
    spread_val = abs(margin)

    # Seed matchup string for display
    if home_seed and away_seed:
        seed_str = f"#{min(home_seed,away_seed)} vs #{max(home_seed,away_seed)}"
    else:
        seed_str = "seeding unknown"

    # Compute upset flag
    is_upset_alert = _check_upset_alert(home_seed, away_seed, preds)

    result = {
        # Game info
        "game_id": f"tourn_{game_date}_{normalize_team(away_team)}_{normalize_team(home_team)}",
        "date": game_date,
        "home_team": home_team,
        "away_team": away_team,
        "home_seed": home_seed,
        "away_seed": away_seed,
        "region": region,
        "round": round_num,
        "round_name": ROUND_NAMES.get(round_num, f"Round {round_num}"),
        "neutral": True,
        "game_tier": game_tier,
        "seed_matchup": seed_str,

        # Predictions
        "predictions": preds,

        # Vegas edge
        "edge": edge,

        # Bet recommendations
        "bets": bets,

        # Derived display fields
        "favored_team": favored,
        "underdog_team": underdog,
        "predicted_spread": f"{favored} -{spread_val:.1f}",
        "home_win_prob_pct": f"{100 * preds.get('home_win_prob', 0.5):.1f}%",
        "away_win_prob_pct": f"{100 * preds.get('away_win_prob', 0.5):.1f}%",
        "is_upset_alert": is_upset_alert,
    }

    # Console output
    wp_home = preds.get("home_win_prob", 0.5)
    wp_away = preds.get("away_win_prob", 0.5)
    total = preds.get("predicted_total", 0)
    vegas_spread = edge.get("vegas_spread", None)
    vegas_str = f" (Vegas: {vegas_spread:+.1f})" if vegas_spread is not None else " (no line yet)"

    log.info("  → %s favored by %.1f%s", favored, spread_val, vegas_str)
    log.info("  → Win prob: %s %.1f%%  |  %s %.1f%%  |  Total: %.1f",
             home_team[:15], 100*wp_home, away_team[:15], 100*wp_away, total)

    if is_upset_alert:
        log.info("  ⚠️  UPSET ALERT: Model likes %s despite being the higher seed", underdog)

    if bets:
        for b in bets:
            log.info("  ★ BET: %s %s — %.1f pt edge [%s]",
                     b["market"], b["lean"], b.get("edge_pts", 0), b["confidence"])

    return result


# ── Seed adjustment helpers ──────────────────────────────────────────────────

def _get_seed_adjustment(home_seed, away_seed, preds) -> float:
    """Look up seed-based ATS adjustment. Returns points to add to predicted_margin."""
    if not home_seed or not away_seed:
        return 0.0

    fav_seed = min(home_seed, away_seed)
    dog_seed = max(home_seed, away_seed)
    adj = SEED_ATS_ADJUSTMENT.get((fav_seed, dog_seed), 0.0)

    # If the adjustment favors the underdog and home team is the underdog, flip sign
    margin = preds.get("predicted_margin", 0)
    home_is_dog = home_seed > away_seed

    # adj is always "add to underdog" so:
    # if home is underdog: add adj to margin (positive)
    # if away is underdog: subtract adj from margin (negative)
    if home_is_dog:
        return adj
    else:
        return -adj


def _get_upset_boost(home_seed, away_seed, preds) -> float:
    """Return win probability boost for known upset-prone seeds."""
    if not home_seed or not away_seed:
        return 0.0
    dog_seed = max(home_seed, away_seed)
    return UPSET_SEED_BOOST.get(dog_seed, 0.0)


def _check_upset_alert(home_seed, away_seed, preds) -> bool:
    """Flag if the model likes the underdog (higher seed) to win."""
    if not home_seed or not away_seed:
        return False
    margin = preds.get("predicted_margin", 0)
    home_win_prob = preds.get("home_win_prob", 0.5)

    # Home is underdog if home_seed > away_seed
    if home_seed > away_seed and home_win_prob > 0.45:
        return True
    # Away is underdog if away_seed > home_seed and model gives them real chance
    if away_seed > home_seed and home_win_prob < 0.52:
        return True
    return False


# ── Output helpers ───────────────────────────────────────────────────────────

def print_tournament_summary(predictions: list[dict]):
    """Print a clean bracket-style summary table."""
    if not predictions:
        return

    print("\n" + "=" * 85)
    print("  NCAA TOURNAMENT PREDICTIONS")
    print("=" * 85)
    print(f"  {'MATCHUP':<38} {'SEEDS':>8} {'SPREAD':>9} {'TOTAL':>7} {'HOME WP':>8}  FLAGS")
    print("─" * 85)

    by_region = {}
    for p in predictions:
        r = p.get("region", "Unknown")
        by_region.setdefault(r, []).append(p)

    for region in (REGIONS + ["Unknown"]):
        games = by_region.get(region, [])
        if not games:
            continue
        print(f"\n  ── {region.upper()} REGION ──")
        for p in sorted(games, key=lambda x: x.get("round", 1)):
            if "error" in p:
                print(f"  ⚠ ERROR: {p['error']}")
                continue

            preds = p.get("predictions", {})
            edge = p.get("edge", {})
            bets = p.get("bets", [])

            h_seed = p.get("home_seed", "?")
            a_seed = p.get("away_seed", "?")
            seed_str = f"#{a_seed} v #{h_seed}"

            matchup = f"{p['away_team'][:17]} @ {p['home_team'][:17]}"
            margin = preds.get("predicted_margin", 0)
            spread_str = f"{margin:+.1f}"
            total_str = f"{preds.get('predicted_total', 0):.1f}"
            wp_str = f"{100*preds.get('home_win_prob', 0.5):.0f}%"

            flags = ""
            if p.get("is_upset_alert"):
                flags += "⚠️UPSET "
            if bets:
                flags += "★" * len(bets)

            print(f"  {matchup:<38} {seed_str:>8} {spread_str:>9} {total_str:>7} {wp_str:>8}  {flags}")

    # Flagged plays
    flagged = [(p, b) for p in predictions if "error" not in p for b in p.get("bets", [])]
    if flagged:
        print(f"\n  {'─'*85}")
        print(f"  FLAGGED TOURNAMENT PLAYS ({len(flagged)} total):")
        print(f"  {'─'*85}")
        for p, b in flagged:
            h_seed = p.get("home_seed", "?")
            a_seed = p.get("away_seed", "?")
            matchup = f"#{a_seed} {p['away_team']} @ #{h_seed} {p['home_team']}"
            print(f"  [{b['confidence']}] {b['market']} {b['lean']:5} — {matchup}")
            print(f"         {p.get('round_name', b.get('tournament_round', 'Round of 64'))} | Edge: {b.get('edge_pts', 0):.1f} pts | {p['region']} Region")
    else:
        print(f"\n  No flagged plays (no lines available yet or edge below threshold)")

    print("=" * 85 + "\n")


def save_tournament_predictions(predictions: list[dict], date_str: str):
    """Save tournament predictions to JSON and CSV."""
    # JSON
    json_path = TOURN_DIR / f"{date_str}_tournament_predictions.json"
    with open(json_path, "w") as f:
        json.dump(predictions, f, indent=2, default=str)
    log.info("Saved: %s", json_path)

    # CSV — flat format for easy viewing
    rows = []
    for p in predictions:
        if "error" in p:
            continue
        preds = p.get("predictions", {})
        edge = p.get("edge", {})
        bets = p.get("bets", [])
        bet_str = ", ".join(f"{b['market']} {b['lean']} [{b['confidence']}]" for b in bets)

        rows.append({
            "round":            p.get("round_name", ""),
            "region":           p.get("region", ""),
            "away_seed":        p.get("away_seed", ""),
            "away_team":        p.get("away_team", ""),
            "home_seed":        p.get("home_seed", ""),
            "home_team":        p.get("home_team", ""),
            "favored":          p.get("favored_team", ""),
            "predicted_spread": p.get("predicted_spread", ""),
            "predicted_total":  preds.get("predicted_total", ""),
            "home_win_prob":    p.get("home_win_prob_pct", ""),
            "away_win_prob":    p.get("away_win_prob_pct", ""),
            "vegas_spread":     edge.get("vegas_spread", ""),
            "spread_edge":      edge.get("spread_edge", ""),
            "vegas_total":      edge.get("vegas_total", ""),
            "total_edge":       edge.get("total_edge", ""),
            "upset_alert":      p.get("is_upset_alert", False),
            "bets":             bet_str,
            "date":             p.get("date", ""),
        })

    csv_path = TOURN_DIR / f"{date_str}_tournament_predictions.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    log.info("Saved: %s", csv_path)

    return json_path, csv_path


# ── Bracket loader ───────────────────────────────────────────────────────────

def load_bracket(json_path: str) -> list[dict]:
    """Load bracket matchups from a JSON file."""
    with open(json_path) as f:
        games = json.load(f)
    log.info("Loaded %d matchups from bracket file: %s", len(games), json_path)
    return games


def build_bracket_template(output_path: str = "bracket_2026.json"):
    """
    Write a blank bracket template JSON that you fill in after Selection Sunday.
    Run: python tournament_predict.py --template
    """
    template = []
    round1_seeds = [(1,16),(8,9),(5,12),(4,13),(6,11),(3,14),(7,10),(2,15)]
    date_map = {
        "East":    {"1": "2026-03-19", "2": "2026-03-20"},
        "West":    {"1": "2026-03-19", "2": "2026-03-20"},
        "South":   {"1": "2026-03-21", "2": "2026-03-22"},
        "Midwest": {"1": "2026-03-21", "2": "2026-03-22"},
    }

    for region in REGIONS:
        day_idx = 0
        for fav_seed, dog_seed in round1_seeds:
            day_key = "1" if day_idx < 4 else "2"
            template.append({
                "home": f"[{region} #{fav_seed} seed]",
                "away": f"[{region} #{dog_seed} seed]",
                "home_seed": fav_seed,
                "away_seed": dog_seed,
                "region": region,
                "round": 1,
                "date": date_map[region][day_key],
            })
            day_idx += 1

    with open(output_path, "w") as f:
        json.dump(template, f, indent=2)

    print(f"\n✅ Bracket template written to: {output_path}")
    print("Fill in 'home' and 'away' team names after Selection Sunday (March 15)")
    print("Then run: python tournament_predict.py --bracket bracket_2026.json\n")


# ── Interactive mode ─────────────────────────────────────────────────────────

def interactive_mode(models, feature_matrix, odds_df, date_str):
    """Prompt for matchups one at a time until user quits."""
    print("\n" + "=" * 55)
    print("  NCAAB TOURNAMENT PREDICTOR — INTERACTIVE MODE")
    print("  Type 'quit' to exit, 'save' to save all predictions")
    print("=" * 55)

    all_predictions = []

    while True:
        print()
        away = input("  Away team (or 'quit'/'save'): ").strip()
        if away.lower() == "quit":
            break
        if away.lower() == "save":
            if all_predictions:
                save_tournament_predictions(all_predictions, date_str)
                print_tournament_summary(all_predictions)
            break

        home = input("  Home team (neutral site): ").strip()
        if not home:
            continue

        away_seed_str = input("  Away seed (or Enter to skip): ").strip()
        home_seed_str = input("  Home seed (or Enter to skip): ").strip()
        region = input("  Region (East/West/South/Midwest, or Enter to skip): ").strip() or "Unknown"
        round_str = input("  Round number (1=R64, 2=R32, 3=S16, 4=E8, 5=FF, 6=Champ) [default 1]: ").strip() or "1"

        away_seed = int(away_seed_str) if away_seed_str.isdigit() else None
        home_seed = int(home_seed_str) if home_seed_str.isdigit() else None
        round_num = int(round_str) if round_str.isdigit() else 1

        pred = predict_tournament_game(
            home_team=home,
            away_team=away,
            home_seed=home_seed,
            away_seed=away_seed,
            game_date=date_str,
            round_num=round_num,
            region=region,
            models=models,
            feature_matrix=feature_matrix,
            odds_df=odds_df,
        )
        all_predictions.append(pred)

        # Quick result
        if "error" not in pred:
            preds = pred.get("predictions", {})
            print(f"\n  → {pred['predicted_spread']}")
            print(f"  → Total: {preds.get('predicted_total', 'N/A'):.1f}")
            print(f"  → {pred['home_team']} win prob: {pred['home_win_prob_pct']}")
            print(f"  → {pred['away_team']} win prob: {pred['away_win_prob_pct']}")
            if pred.get("is_upset_alert"):
                print(f"  ⚠️  UPSET ALERT")
            if pred.get("bets"):
                for b in pred["bets"]:
                    print(f"  ★ {b['market']} {b['lean']} — {b.get('edge_pts',0):.1f} pt edge [{b['confidence']}]")

    return all_predictions


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NCAAB Tournament Predictor")
    parser.add_argument("--home",        help="Home team name (neutral site)")
    parser.add_argument("--away",        help="Away team name")
    parser.add_argument("--home-seed",   type=int, default=None, help="Home team seed (1-16)")
    parser.add_argument("--away-seed",   type=int, default=None, help="Away team seed (1-16)")
    parser.add_argument("--region",      default="Unknown", help="Tournament region")
    parser.add_argument("--round",       type=int, default=1, help="Round number (1-6)")
    parser.add_argument("--date",        default=None, help="Game date YYYY-MM-DD (default: today)")
    parser.add_argument("--bracket",     default=None, help="Path to bracket JSON file")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--template",    action="store_true", help="Write blank bracket template and exit")
    args = parser.parse_args()

    # Generate bracket template and exit
    if args.template:
        build_bracket_template()
        return

    date_str = args.date or datetime.now().strftime("%Y-%m-%d")
    log.info("=" * 55)
    log.info("  NCAAB TOURNAMENT PREDICTOR — %s", date_str)
    log.info("=" * 55)

    # Load models and feature matrix once
    log.info("Loading models...")
    models = load_models()
    if not models:
        log.error("No models found. Run model_trainer first.")
        sys.exit(1)

    log.info("Loading feature matrix...")
    try:
        feature_matrix = load_feature_matrix()
    except Exception as e:
        log.error("Failed to load feature matrix: %s", e)
        sys.exit(1)

    log.info("Fetching current odds (if available)...")
    try:
        odds_df = fetch_current_odds()
    except Exception:
        odds_df = pd.DataFrame()

    # ── Interactive mode ──────────────────────────────────────────────────
    if args.interactive:
        predictions = interactive_mode(models, feature_matrix, odds_df, date_str)
        if predictions:
            save_tournament_predictions(predictions, date_str)
            print_tournament_summary(predictions)
        return

    # ── Bracket file mode ─────────────────────────────────────────────────
    if args.bracket:
        games = load_bracket(args.bracket)
        predictions = []
        for g in games:
            pred = predict_tournament_game(
                home_team=g["home"],
                away_team=g["away"],
                home_seed=g.get("home_seed"),
                away_seed=g.get("away_seed"),
                game_date=g.get("date", date_str),
                round_num=g.get("round", 1),
                region=g.get("region", "Unknown"),
                models=models,
                feature_matrix=feature_matrix,
                odds_df=odds_df,
            )
            predictions.append(pred)
        save_tournament_predictions(predictions, date_str)
        print_tournament_summary(predictions)
        return

    # ── Single matchup mode ───────────────────────────────────────────────
    if args.home and args.away:
        pred = predict_tournament_game(
            home_team=args.home,
            away_team=args.away,
            home_seed=args.home_seed,
            away_seed=args.away_seed,
            game_date=date_str,
            round_num=args.round,
            region=args.region,
            models=models,
            feature_matrix=feature_matrix,
            odds_df=odds_df,
        )
        predictions = [pred]
        save_tournament_predictions(predictions, date_str)
        print_tournament_summary(predictions)
        return

    # ── No args — print help and go interactive ───────────────────────────
    print("\nNo matchup specified. Use --help to see options, or entering interactive mode...\n")
    predictions = interactive_mode(models, feature_matrix, odds_df, date_str)
    if predictions:
        save_tournament_predictions(predictions, date_str)
        print_tournament_summary(predictions)


if __name__ == "__main__":
    main()