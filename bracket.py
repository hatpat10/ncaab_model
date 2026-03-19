"""
dashboard/pages/bracket.py
NCAA Tournament 2026 Bracket Predictions Page
"""
import streamlit as st
import json
from pathlib import Path
import pandas as pd

st.set_page_config(page_title="Tournament Bracket", page_icon="🏀", layout="wide")

# ── Styles ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Outfit:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

.stApp { background: #080d14; }

h1, h2, h3 { font-family: 'Bebas Neue', sans-serif; letter-spacing: 2px; }

.bracket-header {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.4rem;
    color: #e8b84b;
    letter-spacing: 4px;
    margin-bottom: 0.2rem;
}
.bracket-sub {
    font-family: 'Outfit', sans-serif;
    font-size: 0.8rem;
    color: #5a7a9a;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 2rem;
}

.region-label {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.4rem;
    letter-spacing: 3px;
    padding: 0.4rem 1rem;
    margin-bottom: 1rem;
    border-left: 3px solid;
}
.region-east  { color: #4d9de0; border-color: #4d9de0; }
.region-west  { color: #e8b84b; border-color: #e8b84b; }
.region-south { color: #e05d44; border-color: #e05d44; }
.region-midwest { color: #7dc17d; border-color: #7dc17d; }

.matchup-card {
    background: #0d1620;
    border: 1px solid #1a2535;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.6rem;
    font-family: 'Outfit', sans-serif;
    position: relative;
    transition: border-color 0.2s;
}
.matchup-card:hover { border-color: #2a3f5a; }

.matchup-card.flag-HIGH { border-left: 3px solid #e8b84b; }
.matchup-card.flag-MEDIUM { border-left: 3px solid #4d9de0; }

.seed-badge {
    display: inline-block;
    background: #1a2535;
    color: #5a7a9a;
    font-size: 0.65rem;
    font-family: 'JetBrains Mono', monospace;
    padding: 1px 5px;
    border-radius: 3px;
    margin-right: 5px;
    vertical-align: middle;
}

.team-name {
    font-weight: 600;
    font-size: 0.85rem;
    color: #d0dde8;
    text-transform: capitalize;
}
.team-name.winner { color: #ffffff; }

.spread-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: #5a7a9a;
}
.spread-pos { color: #7dc17d; }
.spread-neg { color: #e05d44; }

.total-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #4a6a8a;
}

.wp-bar-wrap {
    height: 4px;
    background: #1a2535;
    border-radius: 2px;
    margin: 6px 0 4px;
    overflow: hidden;
}
.wp-bar-fill {
    height: 100%;
    border-radius: 2px;
    background: linear-gradient(90deg, #4d9de0, #e8b84b);
}

.flag-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    font-weight: 600;
    padding: 2px 6px;
    border-radius: 3px;
    letter-spacing: 1px;
}
.flag-HIGH   { background: rgba(232,184,75,0.15); color: #e8b84b; }
.flag-MEDIUM { background: rgba(77,157,224,0.15); color: #4d9de0; }

.bet-line {
    font-family: 'Outfit', sans-serif;
    font-size: 0.7rem;
    color: #7a9ab8;
    margin-top: 4px;
}
.bet-line strong { color: #c0c8d0; }

.divider { border: none; border-top: 1px solid #1a2535; margin: 1.5rem 0; }

.summary-box {
    background: #0d1620;
    border: 1px solid #1a2535;
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1.5rem;
}
.metric-val {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2rem;
    line-height: 1;
}
.metric-lbl {
    font-family: 'Outfit', sans-serif;
    font-size: 0.7rem;
    color: #5a7a9a;
    letter-spacing: 2px;
    text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)

# ── Load predictions ────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
PRED_DIR = ROOT / "predictions" / "tournament"

@st.cache_data(ttl=300)
def load_tournament_preds():
    files = sorted(PRED_DIR.glob("*_tournament_predictions.json"), reverse=True)
    if not files:
        return None, None
    f = files[0]
    with open(f) as fh:
        data = json.load(fh)
    return data, f.stem.replace("_tournament_predictions", "")

preds, pred_date = load_tournament_preds()

# ── Header ──────────────────────────────────────────────────────────────────
st.markdown('<div class="bracket-header">🏀 NCAA TOURNAMENT 2026</div>', unsafe_allow_html=True)
st.markdown(f'<div class="bracket-sub">MODEL BRACKET PREDICTIONS · {pred_date or "—"}</div>', unsafe_allow_html=True)

if preds is None:
    st.error("No tournament predictions found. Run `python tournament_predict.py --bracket bracket.json` first.")
    st.stop()

games = preds.get("predictions", [])

# ── Summary metrics ─────────────────────────────────────────────────────────
high_flags  = [g for g in games if g.get("confidence") == "HIGH"]
med_flags   = [g for g in games if g.get("confidence") == "MEDIUM"]
all_flags   = high_flags + med_flags
upset_alerts = [g for g in games if g.get("upset_alert")]

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="summary-box">
        <div class="metric-val" style="color:#e8e8e8">{len(games)}</div>
        <div class="metric-lbl">Total Games</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="summary-box">
        <div class="metric-val" style="color:#e8b84b">{len(high_flags)}</div>
        <div class="metric-lbl">High Confidence</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="summary-box">
        <div class="metric-val" style="color:#4d9de0">{len(med_flags)}</div>
        <div class="metric-lbl">Medium Confidence</div>
    </div>""", unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <div class="summary-box">
        <div class="metric-val" style="color:#e05d44">{len(upset_alerts)}</div>
        <div class="metric-lbl">Upset Alerts</div>
    </div>""", unsafe_allow_html=True)

# ── Flagged Plays Summary ────────────────────────────────────────────────────
if all_flags:
    with st.expander(f"⭐ FLAGGED PLAYS ({len(all_flags)} total)", expanded=True):
        for g in sorted(all_flags, key=lambda x: -abs(x.get("spread_edge", 0) or x.get("total_edge", 0) or 0)):
            conf = g.get("confidence", "")
            bet_type = g.get("bet_type", "")
            away = g.get("away_team", "").title()
            home = g.get("home_team", "").title()
            away_seed = g.get("away_seed", "")
            home_seed = g.get("home_seed", "")
            region = g.get("region", "")
            edge = g.get("spread_edge") or g.get("total_edge") or 0

            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.markdown(
                    f'<span class="flag-badge flag-{conf}">{conf}</span> '
                    f'<span style="font-family:Outfit;font-size:0.85rem;color:#c0c8d0">'
                    f'<b>{bet_type}</b> — '
                    f'<span class="seed-badge">#{away_seed}</span>{away} @ '
                    f'<span class="seed-badge">#{home_seed}</span>{home}</span> '
                    f'<span style="font-size:0.7rem;color:#5a7a9a">· {region}</span>',
                    unsafe_allow_html=True
                )
            with col_b:
                color = "#e8b84b" if conf == "HIGH" else "#4d9de0"
                st.markdown(
                    f'<div style="text-align:right;font-family:JetBrains Mono;color:{color};font-size:0.85rem;font-weight:600">+{edge:.1f} pts</div>',
                    unsafe_allow_html=True
                )

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── Region colours ───────────────────────────────────────────────────────────
REGION_CLASS = {"East": "region-east", "West": "region-west", "South": "region-south", "Midwest": "region-midwest"}
REGION_ORDER = ["East", "West", "South", "Midwest"]

def render_matchup(g):
    away = g.get("away_team", "").title()
    home = g.get("home_team", "").title()
    away_seed = g.get("away_seed", "?")
    home_seed = g.get("home_seed", "?")
    spread = g.get("model_spread", 0)
    total = g.get("model_total", 0)
    home_wp = g.get("home_win_prob", 0.5)
    away_wp = 1 - home_wp
    conf = g.get("confidence", "")
    bet_type = g.get("bet_type", "")
    upset = g.get("upset_alert", False)
    spread_edge = g.get("spread_edge")
    total_edge = g.get("total_edge")
    vegas_spread = g.get("vegas_spread")

    card_class = f"matchup-card flag-{conf}" if conf else "matchup-card"

    # Determine projected winner
    away_winner = away_wp > home_wp

    spread_str = f"{spread:+.1f}" if spread else "—"
    spread_color = "spread-pos" if spread and spread > 0 else "spread-neg"

    # Vegas comparison
    vegas_str = ""
    if vegas_spread is not None:
        vegas_str = f" <span style='color:#4a6a8a'>· Vegas {vegas_spread:+.1f}</span>"

    upset_html = " <span style='color:#e8b84b;font-size:0.65rem'>⚡ UPSET ALERT</span>" if upset else ""
    flag_html = f'<span class="flag-badge flag-{conf}" style="float:right">{conf}</span>' if conf else ""

    bet_detail = ""
    if conf:
        edge = spread_edge or total_edge or 0
        bet_detail = f'<div class="bet-line">★ <strong>{bet_type}</strong> · edge: <strong style="color:{"#e8b84b" if conf=="HIGH" else "#4d9de0"}">{edge:+.1f} pts</strong></div>'

    html = f"""
    <div class="{card_class}">
        {flag_html}
        <div style="margin-bottom:4px">
            <span class="seed-badge">#{away_seed}</span>
            <span class="team-name {'winner' if away_winner else ''}">{away}</span>
            <span class="spread-val {spread_color}" style="float:right">{spread_str}{vegas_str}</span>
        </div>
        <div class="wp-bar-wrap">
            <div class="wp-bar-fill" style="width:{away_wp*100:.0f}%"></div>
        </div>
        <div style="margin-bottom:2px">
            <span class="seed-badge">#{home_seed}</span>
            <span class="team-name {'winner' if not away_winner else ''}">{home}</span>
            <span class="total-val" style="float:right">O/U {total:.0f}</span>
        </div>
        <div style="font-family:JetBrains Mono;font-size:0.65rem;color:#3a5a7a;margin-top:3px">
            WP: {away_wp*100:.0f}% / {home_wp*100:.0f}%{upset_html}
        </div>
        {bet_detail}
    </div>
    """
    return html

# ── Render regions ────────────────────────────────────────────────────────────
by_region = {}
for g in games:
    r = g.get("region", "Unknown")
    by_region.setdefault(r, []).append(g)

region_cols = st.columns(2)
for idx, region in enumerate(REGION_ORDER):
    region_games = by_region.get(region, [])
    rclass = REGION_CLASS.get(region, "")
    with region_cols[idx % 2]:
        st.markdown(f'<div class="region-label {rclass}">{region.upper()} REGION</div>', unsafe_allow_html=True)
        for g in region_games:
            st.markdown(render_matchup(g), unsafe_allow_html=True)
        if idx % 2 == 1:
            st.markdown('<hr class="divider">', unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown(
    '<div style="font-family:Outfit;font-size:0.7rem;color:#2a3f5a;text-align:center">'
    'NCAAB ML MODEL · TOURNAMENT PREDICTIONS · FOR INFORMATIONAL PURPOSES ONLY'
    '</div>',
    unsafe_allow_html=True
)
