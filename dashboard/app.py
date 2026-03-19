"""
dashboard/app.py
NCAAB ML Model Dashboard — Streamlit

Run with:
    streamlit run dashboard/app.py

Install deps first:
    pip install streamlit plotly pandas
"""

import streamlit as st
import sqlite3
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime, timedelta

# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH     = Path("data/ncaab.db")
PRED_DIR    = Path("predictions")
REPORT_PATH = Path("reports/backtest_latest.json")

st.set_page_config(
    page_title="NCAAB ML Dashboard",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Outfit:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

/* ── Base ── */
.stApp {
    background: #080d14;
    color: #c8cdd8;
}

/* ── Sidebar ── */
div[data-testid="stSidebar"] {
    background: #05080f;
    border-right: 1px solid #0f1a2e;
}
div[data-testid="stSidebar"] .stRadio label {
    font-family: 'Outfit', sans-serif;
    font-size: 0.85rem;
    font-weight: 500;
    letter-spacing: 0.03em;
    color: #7a8499;
    padding: 6px 0;
    transition: color 0.15s;
}
div[data-testid="stSidebar"] .stRadio label:hover {
    color: #e8b84b;
}

/* ── Page titles ── */
.page-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3.2rem;
    letter-spacing: 0.06em;
    color: #f0f3f8;
    line-height: 1;
    margin-bottom: 2px;
}
.page-subtitle {
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #3d5a7a;
    margin-bottom: 28px;
}

/* ── Section headers ── */
.section-header {
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #3d5a7a;
    margin: 32px 0 14px;
    padding-bottom: 8px;
    border-bottom: 1px solid #0f1a2e;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-header::before {
    content: '';
    display: inline-block;
    width: 3px;
    height: 12px;
    background: #e8b84b;
    border-radius: 2px;
}

/* ── Metric cards ── */
.metric-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 8px;
}
.metric-card {
    background: #0b1220;
    border: 1px solid #0f1a2e;
    border-radius: 10px;
    padding: 18px 20px 16px;
    position: relative;
    overflow: hidden;
}
.metric-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: #e8b84b;
    opacity: 0.4;
}
.metric-card.accent::after { opacity: 1; }
.metric-value {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.4rem;
    letter-spacing: 0.04em;
    color: #f0f3f8;
    line-height: 1;
}
.metric-value.gold  { color: #e8b84b; }
.metric-value.blue  { color: #4d9de0; }
.metric-value.green { color: #3eb489; }
.metric-value.red   { color: #e05c5c; }
.metric-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #3d5a7a;
    margin-top: 6px;
}

/* ── Bet cards ── */
.bet-card {
    background: #0b1220;
    border: 1px solid #0f1a2e;
    border-left: 3px solid #e8b84b;
    border-radius: 0 10px 10px 0;
    padding: 14px 18px;
    margin-bottom: 8px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    transition: border-color 0.15s, background 0.15s;
}
.bet-card:hover {
    background: #0f1928;
    border-left-color: #f5cc70;
}
.bet-card.medium {
    border-left-color: #4d9de0;
}
.bet-card.medium:hover {
    border-left-color: #6db3f2;
}
.bet-card-left {}
.bet-team {
    font-size: 0.95rem;
    font-weight: 600;
    color: #e8eaf0;
    letter-spacing: 0.01em;
}
.bet-detail {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #3d5a7a;
    margin-top: 5px;
    display: flex;
    gap: 10px;
    align-items: center;
}
.bet-detail-sep {
    color: #1e2d40;
}
.bet-market {
    background: #0f1a2e;
    color: #7a8499;
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 2px 8px;
    border-radius: 4px;
}
.bet-market.high { background: rgba(232,184,75,0.12); color: #e8b84b; }
.bet-market.medium { background: rgba(77,157,224,0.12); color: #4d9de0; }
.bet-edge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.15rem;
    font-weight: 500;
    text-align: right;
    white-space: nowrap;
}
.edge-high   { color: #e8b84b; }
.edge-medium { color: #4d9de0; }
.edge-sub {
    font-size: 0.65rem;
    color: #3d5a7a;
    text-align: right;
    margin-top: 2px;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Full slate table ── */
.stDataFrame {
    border: 1px solid #0f1a2e !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}

/* ── Refresh button ── */
div[data-testid="stSidebar"] .stButton button {
    background: #0f1a2e;
    color: #7a8499;
    border: 1px solid #1a2d45;
    border-radius: 8px;
    font-family: 'Outfit', sans-serif;
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.05em;
    width: 100%;
    transition: all 0.15s;
}
div[data-testid="stSidebar"] .stButton button:hover {
    background: #1a2d45;
    color: #e8b84b;
    border-color: #e8b84b;
}

/* ── Warning/info ── */
.stAlert {
    background: #0b1220 !important;
    border: 1px solid #0f1a2e !important;
    border-radius: 10px !important;
}

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: #0b1220 !important;
    border-color: #0f1a2e !important;
    border-radius: 8px !important;
}

/* ── Native st.metric override ── */
[data-testid="stMetric"] {
    background: #0b1220;
    border: 1px solid #0f1a2e;
    border-radius: 10px;
    padding: 16px 18px;
}
[data-testid="stMetricLabel"] {
    font-size: 0.65rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: #3d5a7a !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 2rem !important;
    color: #f0f3f8 !important;
}

/* ── Divider ── */
hr {
    border-color: #0f1a2e !important;
    margin: 16px 0 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Data loaders ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_predictions_db(min_date=None):
    if not DB_PATH.exists():
        return pd.DataFrame()
    with sqlite3.connect(DB_PATH) as con:
        q = "SELECT * FROM predictions"
        if min_date:
            q += f" WHERE date >= '{min_date}'"
        q += " ORDER BY date DESC"
        return pd.read_sql_query(q, con)

@st.cache_data(ttl=300)
def load_todays_json(date_str):
    path = PRED_DIR / f"{date_str}_predictions.json"
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)

@st.cache_data(ttl=300)
def load_backtest_report():
    if not REPORT_PATH.exists():
        return None
    with open(REPORT_PATH) as f:
        return json.load(f)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='font-family:Bebas Neue,sans-serif;font-size:1.6rem;"
        "letter-spacing:0.1em;color:#f0f3f8;padding:8px 0 4px'>🏀 NCAAB ML</div>",
        unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:0.62rem;font-weight:600;letter-spacing:0.18em;"
        "text-transform:uppercase;color:#3d5a7a;margin-bottom:16px'>Model Dashboard</div>",
        unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["Today's Slate", "Backtest Performance", "Prediction History"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    today    = datetime.now().strftime("%Y-%m-%d")
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    db_dot = '<span style="color:#3eb489">●</span>' if DB_PATH.exists() else '<span style="color:#e05c5c">●</span>'
    st.markdown(
        f"<div style='font-size:0.72rem;color:#3d5a7a;margin-bottom:4px'>"
        f"<span style='color:#4d6a80'>DATE</span>&nbsp;&nbsp;{today}</div>"
        f"<div style='font-size:0.72rem;color:#3d5a7a;margin-bottom:16px'>"
        f"<span style='color:#4d6a80'>DB</span>&nbsp;&nbsp;"
        f"{db_dot}"
        f"&nbsp;{DB_PATH}</div>",
        unsafe_allow_html=True)
    if st.button("⟳  Refresh Data"):
        st.cache_data.clear()
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: TODAY'S SLATE
# ══════════════════════════════════════════════════════════════════════════════
if page == "Today's Slate":
    preds, slate_date = [], tomorrow
    for date_str in [tomorrow, today]:
        preds = load_todays_json(date_str)
        if preds:
            slate_date = date_str
            break

    st.markdown("<div class='page-title'>Today's Slate</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='page-subtitle'>Predictions for {slate_date} — {len(preds)} games</div>",
        unsafe_allow_html=True)

    if not preds:
        st.warning(f"No predictions found for {slate_date}. Run the pipeline first.")
        st.code(f"python -m predictions.daily_pipeline --date {slate_date}")
    else:
        flagged    = [p for p in preds if p.get("bets")]
        high       = [p for p in flagged if any(b["confidence"] == "HIGH"   for b in p["bets"])]
        medium     = [p for p in flagged if any(b["confidence"] == "MEDIUM" for b in p["bets"])
                      and not any(b["confidence"] == "HIGH" for b in p["bets"])]
        odds_count = sum(1 for p in preds if p.get("edge", {}).get("vegas_spread") is not None)

        # ── Summary metrics ──
        st.markdown(
            f"""<div class='metric-row'>
              <div class='metric-card'>
                <div class='metric-value'>{len(preds)}</div>
                <div class='metric-label'>Games</div>
              </div>
              <div class='metric-card accent'>
                <div class='metric-value gold'>{len(high)}</div>
                <div class='metric-label'>High Flags</div>
              </div>
              <div class='metric-card'>
                <div class='metric-value blue'>{len(medium)}</div>
                <div class='metric-label'>Medium Flags</div>
              </div>
              <div class='metric-card'>
                <div class='metric-value green'>{odds_count}</div>
                <div class='metric-label'>With Odds</div>
              </div>
            </div>""",
            unsafe_allow_html=True)

        # ── HIGH flags ──
        if high:
            st.markdown("<div class='section-header'>High Confidence Flags</div>",
                        unsafe_allow_html=True)
            for p in high:
                for bet in [b for b in p["bets"] if b["confidence"] == "HIGH"]:
                    e  = p.get("edge", {})
                    pr = p.get("predictions", {})
                    model_v = f"{pr['predicted_margin']:+.1f}" if pr.get("predicted_margin") is not None else "—"
                    vegas_v = f"{e['vegas_spread']:+.1f}"      if e.get("vegas_spread")       is not None else "—"
                    lean    = bet.get("lean", "")
                    market  = bet.get("market", "")
                    st.markdown(
                        f"<div class='bet-card'>"
                        f"  <div class='bet-card-left'>"
                        f"    <div class='bet-team'>{p['away_team']} @ {p['home_team']}</div>"
                        f"    <div class='bet-detail'>"
                        f"      <span class='bet-market high'>{market} {lean}</span>"
                        f"      <span>Model {model_v}</span>"
                        f"      <span class='bet-detail-sep'>|</span>"
                        f"      <span>Vegas {vegas_v}</span>"
                        f"    </div>"
                        f"  </div>"
                        f"  <div>"
                        f"    <div class='bet-edge edge-high'>+{bet['edge_pts']:.1f} pts</div>"
                        f"    <div class='edge-sub'>edge</div>"
                        f"  </div>"
                        f"</div>",
                        unsafe_allow_html=True)

        # ── MEDIUM flags ──
        if medium:
            st.markdown("<div class='section-header'>Medium Confidence Flags</div>",
                        unsafe_allow_html=True)
            for p in medium:
                for bet in [b for b in p["bets"] if b["confidence"] == "MEDIUM"]:
                    pr = p.get("predictions", {})
                    e  = p.get("edge", {})
                    model_v = f"{pr['predicted_margin']:+.1f}" if pr.get("predicted_margin") is not None else "—"
                    vegas_v = f"{e['vegas_spread']:+.1f}"      if e.get("vegas_spread")       is not None else "—"
                    lean    = bet.get("lean", "")
                    market  = bet.get("market", "")
                    st.markdown(
                        f"<div class='bet-card medium'>"
                        f"  <div class='bet-card-left'>"
                        f"    <div class='bet-team'>{p['away_team']} @ {p['home_team']}</div>"
                        f"    <div class='bet-detail'>"
                        f"      <span class='bet-market medium'>{market} {lean}</span>"
                        f"      <span>Model {model_v}</span>"
                        f"      <span class='bet-detail-sep'>|</span>"
                        f"      <span>Vegas {vegas_v}</span>"
                        f"    </div>"
                        f"  </div>"
                        f"  <div>"
                        f"    <div class='bet-edge edge-medium'>+{bet['edge_pts']:.1f} pts</div>"
                        f"    <div class='edge-sub'>edge</div>"
                        f"  </div>"
                        f"</div>",
                        unsafe_allow_html=True)

        # ── Full slate table ──
        st.markdown("<div class='section-header'>Full Slate</div>", unsafe_allow_html=True)
        rows = []
        for p in preds:
            e, pr = p.get("edge", {}), p.get("predictions", {})
            rows.append({
                "Matchup":  f"{p['away_team']} @ {p['home_team']}",
                "Spread":   pr.get("predicted_margin"),
                "Vegas":    e.get("vegas_spread"),
                "S.Edge":   e.get("spread_edge"),
                "Total":    pr.get("predicted_total"),
                "V.Total":  e.get("vegas_total"),
                "T.Edge":   e.get("total_edge"),
                "Home WP":  pr.get("home_win_prob"),
                "Flags":    len(p.get("bets", [])),
            })
        df = pd.DataFrame(rows)
        st.dataframe(
            df.style.format({
                "Spread":  lambda x: f"{x:+.1f}" if pd.notna(x) else "—",
                "Vegas":   lambda x: f"{x:+.1f}" if pd.notna(x) else "—",
                "S.Edge":  lambda x: f"{x:+.1f}" if pd.notna(x) else "—",
                "Total":   lambda x: f"{x:.1f}"  if pd.notna(x) else "—",
                "V.Total": lambda x: f"{x:.1f}"  if pd.notna(x) else "—",
                "T.Edge":  lambda x: f"{x:+.1f}" if pd.notna(x) else "—",
                "Home WP": lambda x: f"{x:.0%}"  if pd.notna(x) else "—",
            }),
            use_container_width=True, height=600,
        )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: BACKTEST PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Backtest Performance":
    st.markdown("<div class='page-title'>Backtest Performance</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>Model validation against historical results</div>",
                unsafe_allow_html=True)

    report = load_backtest_report()
    df_all = load_predictions_db()

    if report:
        spread = report.get("spread", {})
        totals = report.get("totals", {})
        ats    = spread.get("overall_ats", 0)
        color_cls = "green" if ats > 0.524 else "red"

        st.markdown("<div class='section-header'>Model Overview</div>", unsafe_allow_html=True)
        st.markdown(
            f"""<div class='metric-row'>
              <div class='metric-card'>
                <div class='metric-value'>{report.get("games_graded", 0)}</div>
                <div class='metric-label'>Games Graded</div>
              </div>
              <div class='metric-card accent'>
                <div class='metric-value {color_cls}'>{ats:.1%}</div>
                <div class='metric-label'>Overall ATS</div>
              </div>
              <div class='metric-card'>
                <div class='metric-value blue'>{spread.get("mae", 0):.2f}</div>
                <div class='metric-label'>Spread MAE (pts)</div>
              </div>
              <div class='metric-card'>
                <div class='metric-value blue'>{totals.get("mae", 0):.2f}</div>
                <div class='metric-label'>Totals MAE (pts)</div>
              </div>
            </div>""",
            unsafe_allow_html=True)

        # ── Spread bucket chart ──
        buckets = spread.get("by_bucket", [])
        if buckets:
            st.markdown("<div class='section-header'>Spread ATS by Edge Bucket</div>",
                        unsafe_allow_html=True)
            bdf = pd.DataFrame(buckets)
            bdf["edge_range"] = bdf.apply(
                lambda r: f"{r['edge_min']:.0f}–{r['edge_max']}", axis=1)
            bdf["win_pct"]    = bdf["win_rate"] * 100
            bdf["bar_color"]  = bdf["profitable"].map({True: "#3eb489", False: "#e05c5c"})

            fig = go.Figure(go.Bar(
                x=bdf["edge_range"], y=bdf["win_pct"],
                marker_color=bdf["bar_color"],
                marker_line_width=0,
                text=bdf.apply(
                    lambda r: f"{r['win_pct']:.1f}%<br><span style='font-size:10px'>n={r['n_games']}</span>", axis=1),
                textposition="outside",
                textfont=dict(color="#7a8499", size=11, family="JetBrains Mono"),
            ))
            fig.add_hline(y=52.38, line_dash="dash", line_color="#e8b84b", line_width=1.5,
                          annotation_text="Break-even 52.4%",
                          annotation_font=dict(color="#e8b84b", size=11))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#0b1220",
                font=dict(color="#7a8499", family="Outfit"),
                yaxis=dict(title="ATS Win %", range=[0, 118],
                           gridcolor="#0f1a2e", tickfont=dict(size=11)),
                xaxis=dict(title="Edge Size (pts)",
                           tickfont=dict(size=11), linecolor="#0f1a2e"),
                showlegend=False,
                height=340,
                margin=dict(t=30, b=20, l=10, r=10),
            )
            st.plotly_chart(fig, use_container_width=True)

            # ROI table
            st.markdown("<div class='section-header'>ROI by Edge Bucket</div>",
                        unsafe_allow_html=True)
            rtbl = bdf[["edge_range", "n_games", "wins", "win_pct", "roi_pct", "profitable"]].copy()
            rtbl.columns = ["Edge Range", "Games", "Wins", "Win %", "ROI %", "Profitable"]
            st.dataframe(
                rtbl.style.format({"Win %": "{:.1f}%", "ROI %": "{:+.1f}%"}),
                use_container_width=True)

        # ── Totals bucket chart ──
        tbuckets = totals.get("by_bucket", [])
        if tbuckets:
            st.markdown("<div class='section-header'>Totals O/U by Edge Bucket</div>",
                        unsafe_allow_html=True)
            tdf = pd.DataFrame(tbuckets)
            tdf["edge_range"] = tdf.apply(
                lambda r: f"{r['edge_min']:.0f}–{r['edge_max']}", axis=1)
            tdf["win_pct"]    = tdf["win_rate"] * 100
            tdf["bar_color"]  = tdf["profitable"].map({True: "#3eb489", False: "#e05c5c"})

            fig2 = go.Figure(go.Bar(
                x=tdf["edge_range"], y=tdf["win_pct"],
                marker_color=tdf["bar_color"],
                marker_line_width=0,
                text=tdf.apply(
                    lambda r: f"{r['win_pct']:.1f}%<br>n={r['n_games']}", axis=1),
                textposition="outside",
                textfont=dict(color="#7a8499", size=11, family="JetBrains Mono"),
            ))
            fig2.add_hline(y=52.38, line_dash="dash", line_color="#e8b84b", line_width=1.5,
                           annotation_text="Break-even 52.4%",
                           annotation_font=dict(color="#e8b84b", size=11))
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#0b1220",
                font=dict(color="#7a8499", family="Outfit"),
                yaxis=dict(title="O/U Win %", range=[0, 118],
                           gridcolor="#0f1a2e", tickfont=dict(size=11)),
                xaxis=dict(title="Edge Size (pts)",
                           tickfont=dict(size=11), linecolor="#0f1a2e"),
                showlegend=False,
                height=340,
                margin=dict(t=30, b=20, l=10, r=10),
            )
            st.plotly_chart(fig2, use_container_width=True)

    else:
        st.warning("No backtest report found.")
        st.code("python -m validation.backtester --output reports/backtest_latest.json")

    # ── Coverage chart ──
    if not df_all.empty:
        st.markdown("<div class='section-header'>Grading Coverage by Date</div>",
                    unsafe_allow_html=True)
        cov = df_all.groupby("date").agg(
            total =("game_id",       "count"),
            graded=("actual_margin", lambda x: x.notna().sum())
        ).reset_index()

        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=cov["date"], y=cov["total"],
            name="Total",  marker_color="#0f1a2e", marker_line_width=0))
        fig3.add_trace(go.Bar(
            x=cov["date"], y=cov["graded"],
            name="Graded", marker_color="#4d9de0", marker_line_width=0))
        fig3.update_layout(
            barmode="overlay",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#0b1220",
            font=dict(color="#7a8499", family="Outfit"),
            yaxis=dict(gridcolor="#0f1a2e", tickfont=dict(size=11)),
            xaxis=dict(tickfont=dict(size=10), linecolor="#0f1a2e"),
            height=260,
            margin=dict(t=10, b=10, l=10, r=10),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        )
        st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: PREDICTION HISTORY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Prediction History":
    st.markdown("<div class='page-title'>Prediction History</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>Full database of model predictions</div>",
                unsafe_allow_html=True)

    df = load_predictions_db()
    if df.empty:
        st.warning("No predictions in database yet.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            dates = sorted(df["date"].unique(), reverse=True)
            selected_date = st.selectbox("Date", ["All"] + list(dates))
        with c2:
            graded_filter = st.selectbox("Results", ["All", "Graded only", "Pending only"])
        with c3:
            flagged_filter = st.selectbox("Flags", ["All", "Flagged only (edge >= 7)"])

        filtered = df.copy()
        if selected_date != "All":
            filtered = filtered[filtered["date"] == selected_date]
        if graded_filter == "Graded only":
            filtered = filtered[filtered["actual_margin"].notna()]
        elif graded_filter == "Pending only":
            filtered = filtered[filtered["actual_margin"].isna()]
        if flagged_filter == "Flagged only (edge >= 7)":
            filtered = filtered[filtered["spread_edge"].abs() >= 7]

        st.markdown(
            f"<div style='font-size:0.72rem;font-weight:500;letter-spacing:0.08em;"
            f"color:#3d5a7a;margin:8px 0 16px'>"
            f"Showing <span style='color:#7a8499'>{len(filtered)}</span> of "
            f"<span style='color:#7a8499'>{len(df)}</span> predictions</div>",
            unsafe_allow_html=True)

        # ── Summary metrics ──
        graded = filtered[filtered["actual_margin"].notna()].copy()
        if len(graded) > 0:
            graded["spread_error"] = (graded["actual_margin"] - graded["predicted_margin"]).abs()
            graded_line = graded[graded["vegas_spread"].notna()].copy()
            if len(graded_line) > 0:
                graded_line["home_covered"] = graded_line["actual_margin"] > -graded_line["vegas_spread"]
                graded_line["bet_home"]     = graded_line["spread_edge"] > 0
                graded_line["ats_win"]      = graded_line["bet_home"] == graded_line["home_covered"]
                ats_rate = graded_line["ats_win"].mean()
                ats_str  = f"{ats_rate:.1%}"
                ats_cls  = "green" if ats_rate > 0.524 else "red"
            else:
                ats_str, ats_cls = "—", ""

            mae_val = graded["spread_error"].mean()
            st.markdown(
                f"""<div style='display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:24px'>
                  <div class='metric-card'>
                    <div class='metric-value'>{len(filtered)}</div>
                    <div class='metric-label'>Total Games</div>
                  </div>
                  <div class='metric-card'>
                    <div class='metric-value blue'>{len(graded)}</div>
                    <div class='metric-label'>Graded</div>
                  </div>
                  <div class='metric-card'>
                    <div class='metric-value blue'>{mae_val:.2f}</div>
                    <div class='metric-label'>Spread MAE (pts)</div>
                  </div>
                  <div class='metric-card accent'>
                    <div class='metric-value {ats_cls}'>{ats_str}</div>
                    <div class='metric-label'>ATS Win Rate</div>
                  </div>
                </div>""",
                unsafe_allow_html=True)

        # ── Table ──
        cols = ["date", "away_team", "home_team", "predicted_margin", "vegas_spread",
                "spread_edge", "predicted_total", "vegas_total", "total_edge",
                "actual_margin", "actual_total"]
        cols = [c for c in cols if c in filtered.columns]
        show = filtered[cols].copy()
        show.columns = [c.replace("_", " ").title() for c in cols]

        st.dataframe(
            show.style.format({
                "Predicted Margin": lambda x: f"{x:+.1f}" if pd.notna(x) else "—",
                "Vegas Spread":     lambda x: f"{x:+.1f}" if pd.notna(x) else "—",
                "Spread Edge":      lambda x: f"{x:+.1f}" if pd.notna(x) else "—",
                "Total Edge":       lambda x: f"{x:+.1f}" if pd.notna(x) else "—",
                "Actual Margin":    lambda x: f"{x:+.0f}" if pd.notna(x) else "⏳",
                "Actual Total":     lambda x: f"{x:.0f}"  if pd.notna(x) else "⏳",
                "Predicted Total":  lambda x: f"{x:.1f}"  if pd.notna(x) else "—",
                "Vegas Total":      lambda x: f"{x:.1f}"  if pd.notna(x) else "—",
            }),
            use_container_width=True, height=700,
        )

        # ── Scatter: predicted vs actual ──
        if len(graded) > 10:
            st.markdown(
                "<div class='section-header'>Predicted vs Actual Margin</div>",
                unsafe_allow_html=True)
            fig4 = px.scatter(
                graded, x="predicted_margin", y="actual_margin",
                hover_data=["home_team", "away_team", "date"],
                trendline="ols",
                color_discrete_sequence=["#4d9de0"],
            )
            fig4.add_hline(y=0, line_color="#1e2d40", line_dash="dot")
            fig4.add_vline(x=0, line_color="#1e2d40", line_dash="dot")
            fig4.update_traces(
                marker=dict(size=6, opacity=0.7, line=dict(width=0)),
                selector=dict(mode='markers'))
            fig4.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#0b1220",
                font=dict(color="#7a8499", family="Outfit"),
                xaxis=dict(title="Predicted Margin", gridcolor="#0f1a2e",
                           zeroline=False, tickfont=dict(size=11)),
                yaxis=dict(title="Actual Margin",    gridcolor="#0f1a2e",
                           zeroline=False, tickfont=dict(size=11)),
                height=420,
                margin=dict(t=10, l=10, r=10, b=10),
            )
            st.plotly_chart(fig4, use_container_width=True)