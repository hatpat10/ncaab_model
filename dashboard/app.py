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

# ── Config ─────────────────────────────────────────────────────────────────
DB_PATH     = Path("data/ncaab.db")
PRED_DIR    = Path("predictions")
REPORT_PATH = Path("reports/backtest_latest.json")

st.set_page_config(
    page_title="NCAAB ML Dashboard",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0a0e1a; color: #e8eaf0; }
.metric-card {
    background: #111827; border: 1px solid #1e293b;
    border-radius: 12px; padding: 20px 24px; margin-bottom: 12px;
}
.metric-value {
    font-family: 'DM Mono', monospace; font-size: 2rem;
    font-weight: 500; color: #38bdf8; line-height: 1;
}
.metric-label {
    font-size: 0.75rem; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; color: #64748b; margin-top: 6px;
}
.bet-card {
    background: #111827; border-left: 3px solid #38bdf8;
    border-radius: 0 8px 8px 0; padding: 14px 18px; margin-bottom: 8px;
}
.bet-card.high  { border-left-color: #f59e0b; }
.bet-card.medium{ border-left-color: #38bdf8; }
.bet-team  { font-size: 0.95rem; font-weight: 600; color: #e8eaf0; }
.bet-detail{ font-family: 'DM Mono', monospace; font-size: 0.8rem; color: #64748b; margin-top: 4px; }
.bet-edge  { font-family: 'DM Mono', monospace; font-size: 1.1rem; font-weight: 500; }
.edge-high { color: #f59e0b; }
.edge-med  { color: #38bdf8; }
.section-header {
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.15em;
    text-transform: uppercase; color: #475569;
    margin: 24px 0 12px; padding-bottom: 8px; border-bottom: 1px solid #1e293b;
}
div[data-testid="stSidebar"] { background: #070b14; border-right: 1px solid #1e293b; }
</style>
""", unsafe_allow_html=True)

# ── Data loaders ────────────────────────────────────────────────────────────
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

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏀 NCAAB ML")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["Today's Slate", "Backtest Performance", "Prediction History"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    today    = datetime.now().strftime("%Y-%m-%d")
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    st.caption(f"Today: {today}")
    st.caption(f"DB: {'✅' if DB_PATH.exists() else '❌'} {DB_PATH}")
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: TODAY'S SLATE
# ══════════════════════════════════════════════════════════════════════════════
if page == "Today's Slate":
    # Try tomorrow first (predictions run evening before), fallback to today
    preds, slate_date = [], tomorrow
    for date_str in [tomorrow, today]:
        preds = load_todays_json(date_str)
        if preds:
            slate_date = date_str
            break

    st.markdown("# Today's Slate")
    st.markdown(
        f"<div class='section-header'>Predictions for {slate_date} — {len(preds)} games</div>",
        unsafe_allow_html=True)

    if not preds:
        st.warning(f"No predictions found for {slate_date}. Run the pipeline first.")
        st.code(f"python -m predictions.daily_pipeline --date {slate_date}")
    else:
        flagged = [p for p in preds if p.get("bets")]
        high    = [p for p in flagged if any(b["confidence"] == "HIGH"   for b in p["bets"])]
        medium  = [p for p in flagged if any(b["confidence"] == "MEDIUM" for b in p["bets"])
                   and not any(b["confidence"] == "HIGH" for b in p["bets"])]
        odds_count = sum(1 for p in preds
                         if p.get("edge", {}).get("vegas_spread") is not None)

        c1, c2, c3, c4 = st.columns(4)
        for col, val, label, color in [
            (c1, len(preds),    "Games",        "#e8eaf0"),
            (c2, len(high),     "HIGH Flags",   "#f59e0b"),
            (c3, len(medium),   "MEDIUM Flags", "#38bdf8"),
            (c4, odds_count,    "With Odds",    "#22c55e"),
        ]:
            with col:
                col.markdown(
                    f"<div class='metric-card'>"
                    f"<div class='metric-value' style='color:{color}'>{val}</div>"
                    f"<div class='metric-label'>{label}</div></div>",
                    unsafe_allow_html=True)

        # HIGH flags
        if high:
            st.markdown("<div class='section-header'>⚡ HIGH Confidence Flags</div>",
                        unsafe_allow_html=True)
            for p in high:
                for bet in [b for b in p["bets"] if b["confidence"] == "HIGH"]:
                    e  = p.get("edge", {})
                    pr = p.get("predictions", {})
                    model_spread = f"{pr['predicted_margin']:+.1f}" if pr.get("predicted_margin") is not None else "—"
                    vegas_spread = f"{e['vegas_spread']:+.1f}"       if e.get("vegas_spread")       is not None else "—"
                    st.markdown(
                        f"<div class='bet-card high'>"
                        f"<div class='bet-team'>{p['away_team']} @ {p['home_team']}</div>"
                        f"<div style='display:flex;justify-content:space-between;align-items:center;margin-top:6px'>"
                        f"<div class='bet-detail'>{bet['market']} {bet['lean']}"
                        f" &nbsp;|&nbsp; Model: {model_spread}"
                        f" &nbsp;|&nbsp; Vegas: {vegas_spread}</div>"
                        f"<div class='bet-edge edge-high'>+{bet['edge_pts']:.1f} pts</div>"
                        f"</div></div>",
                        unsafe_allow_html=True)

        # MEDIUM flags
        if medium:
            st.markdown("<div class='section-header'>📊 MEDIUM Confidence Flags</div>",
                        unsafe_allow_html=True)
            for p in medium:
                for bet in [b for b in p["bets"] if b["confidence"] == "MEDIUM"]:
                    pr = p.get("predictions", {})
                    model_spread = f"{pr['predicted_margin']:+.1f}" if pr.get("predicted_margin") is not None else "—"
                    st.markdown(
                        f"<div class='bet-card medium'>"
                        f"<div class='bet-team'>{p['away_team']} @ {p['home_team']}</div>"
                        f"<div style='display:flex;justify-content:space-between;align-items:center;margin-top:6px'>"
                        f"<div class='bet-detail'>{bet['market']} {bet['lean']}"
                        f" &nbsp;|&nbsp; Model: {model_spread}</div>"
                        f"<div class='bet-edge edge-med'>+{bet['edge_pts']:.1f} pts</div>"
                        f"</div></div>",
                        unsafe_allow_html=True)

        # Full slate table
        st.markdown("<div class='section-header'>Full Slate</div>", unsafe_allow_html=True)
        rows = []
        for p in preds:
            e, pr = p.get("edge", {}), p.get("predictions", {})
            rows.append({
                "Matchup":      f"{p['away_team']} @ {p['home_team']}",
                "Spread":       pr.get("predicted_margin"),
                "Vegas":        e.get("vegas_spread"),
                "S.Edge":       e.get("spread_edge"),
                "Total":        pr.get("predicted_total"),
                "V.Total":      e.get("vegas_total"),
                "T.Edge":       e.get("total_edge"),
                "Home WP":      pr.get("home_win_prob"),
                "Flags":        len(p.get("bets", [])),
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
    st.markdown("# Backtest Performance")

    report = load_backtest_report()
    df_all = load_predictions_db()

    if report:
        spread = report.get("spread", {})
        totals = report.get("totals", {})

        st.markdown("<div class='section-header'>Model Overview</div>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        ats   = spread.get("overall_ats", 0)
        color = "#22c55e" if ats > 0.524 else "#ef4444"
        for col, val, label, clr in [
            (c1, str(report.get("games_graded", 0)),     "Games Graded",   "#e8eaf0"),
            (c2, f"{ats:.1%}",                           "Overall ATS",     color),
            (c3, f"{spread.get('mae', 0):.2f}",          "Spread MAE (pts)","#e8eaf0"),
            (c4, f"{totals.get('mae', 0):.2f}",          "Totals MAE (pts)","#e8eaf0"),
        ]:
            with col:
                col.markdown(
                    f"<div class='metric-card'>"
                    f"<div class='metric-value' style='color:{clr}'>{val}</div>"
                    f"<div class='metric-label'>{label}</div></div>",
                    unsafe_allow_html=True)

        # Spread bucket bar chart
        buckets = spread.get("by_bucket", [])
        if buckets:
            st.markdown("<div class='section-header'>Spread ATS by Edge Bucket</div>",
                        unsafe_allow_html=True)
            bdf = pd.DataFrame(buckets)
            bdf["edge_range"] = bdf.apply(
                lambda r: f"{r['edge_min']:.0f}–{r['edge_max']}", axis=1)
            bdf["win_pct"] = bdf["win_rate"] * 100
            bdf["bar_color"] = bdf["profitable"].map({True: "#22c55e", False: "#ef4444"})

            fig = go.Figure(go.Bar(
                x=bdf["edge_range"], y=bdf["win_pct"],
                marker_color=bdf["bar_color"],
                text=bdf.apply(
                    lambda r: f"{r['win_pct']:.1f}%<br>n={r['n_games']}", axis=1),
                textposition="outside",
            ))
            fig.add_hline(y=52.38, line_dash="dash", line_color="#f59e0b",
                          annotation_text="Break-even 52.4%")
            fig.update_layout(
                paper_bgcolor="#0a0e1a", plot_bgcolor="#111827",
                font=dict(color="#e8eaf0", family="DM Sans"),
                yaxis=dict(title="ATS Win %", range=[0, 115], gridcolor="#1e293b"),
                xaxis=dict(title="Edge Size (pts)"),
                showlegend=False, height=360, margin=dict(t=20, b=20),
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

        # Totals bucket chart
        tbuckets = totals.get("by_bucket", [])
        if tbuckets:
            st.markdown("<div class='section-header'>Totals O/U by Edge Bucket</div>",
                        unsafe_allow_html=True)
            tdf = pd.DataFrame(tbuckets)
            tdf["edge_range"] = tdf.apply(
                lambda r: f"{r['edge_min']:.0f}–{r['edge_max']}", axis=1)
            tdf["win_pct"]   = tdf["win_rate"] * 100
            tdf["bar_color"] = tdf["profitable"].map({True: "#22c55e", False: "#ef4444"})

            fig2 = go.Figure(go.Bar(
                x=tdf["edge_range"], y=tdf["win_pct"],
                marker_color=tdf["bar_color"],
                text=tdf.apply(
                    lambda r: f"{r['win_pct']:.1f}%<br>n={r['n_games']}", axis=1),
                textposition="outside",
            ))
            fig2.add_hline(y=52.38, line_dash="dash", line_color="#f59e0b",
                           annotation_text="Break-even 52.4%")
            fig2.update_layout(
                paper_bgcolor="#0a0e1a", plot_bgcolor="#111827",
                font=dict(color="#e8eaf0", family="DM Sans"),
                yaxis=dict(title="O/U Win %", range=[0, 115], gridcolor="#1e293b"),
                xaxis=dict(title="Edge Size (pts)"),
                showlegend=False, height=360, margin=dict(t=20, b=20),
            )
            st.plotly_chart(fig2, use_container_width=True)

    else:
        st.warning("No backtest report found.")
        st.code("python -m validation.backtester --output reports/backtest_latest.json")

    # Coverage chart
    if not df_all.empty:
        st.markdown("<div class='section-header'>Grading Coverage by Date</div>",
                    unsafe_allow_html=True)
        cov = df_all.groupby("date").agg(
            total =("game_id",       "count"),
            graded=("actual_margin", lambda x: x.notna().sum())
        ).reset_index()

        fig3 = go.Figure()
        fig3.add_trace(go.Bar(x=cov["date"], y=cov["total"],
                              name="Total",  marker_color="#1e293b"))
        fig3.add_trace(go.Bar(x=cov["date"], y=cov["graded"],
                              name="Graded", marker_color="#38bdf8"))
        fig3.update_layout(
            barmode="overlay", paper_bgcolor="#0a0e1a", plot_bgcolor="#111827",
            font=dict(color="#e8eaf0", family="DM Sans"),
            yaxis=dict(gridcolor="#1e293b"), height=280,
            margin=dict(t=10, b=10), legend=dict(bgcolor="#111827"),
        )
        st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: PREDICTION HISTORY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Prediction History":
    st.markdown("# Prediction History")

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

        st.caption(f"Showing {len(filtered)} of {len(df)} predictions")

        # Summary metrics
        graded = filtered[filtered["actual_margin"].notna()].copy()
        if len(graded) > 0:
            graded["spread_error"] = (graded["actual_margin"] - graded["predicted_margin"]).abs()
            graded_line = graded[graded["vegas_spread"].notna()].copy()
            if len(graded_line) > 0:
                graded_line["home_covered"] = graded_line["actual_margin"] > -graded_line["vegas_spread"]
                graded_line["bet_home"]     = graded_line["spread_edge"] > 0
                graded_line["ats_win"]      = graded_line["bet_home"] == graded_line["home_covered"]
                ats_rate = graded_line["ats_win"].mean()
            else:
                ats_rate = None

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Games",  len(filtered))
            m2.metric("Graded",       len(graded))
            m3.metric("Spread MAE",   f"{graded['spread_error'].mean():.2f} pts")
            m4.metric("ATS Win Rate", f"{ats_rate:.1%}" if ats_rate is not None else "—")

        # Table
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

        # Scatter: predicted vs actual
        if len(graded) > 10:
            st.markdown(
                "<div class='section-header'>Predicted vs Actual Margin</div>",
                unsafe_allow_html=True)
            fig4 = px.scatter(
                graded, x="predicted_margin", y="actual_margin",
                hover_data=["home_team", "away_team", "date"],
                trendline="ols",
                color_discrete_sequence=["#38bdf8"],
            )
            fig4.add_hline(y=0, line_color="#475569", line_dash="dot")
            fig4.add_vline(x=0, line_color="#475569", line_dash="dot")
            fig4.update_layout(
                paper_bgcolor="#0a0e1a", plot_bgcolor="#111827",
                font=dict(color="#e8eaf0", family="DM Sans"),
                xaxis=dict(title="Predicted Margin", gridcolor="#1e293b", zeroline=False),
                yaxis=dict(title="Actual Margin",    gridcolor="#1e293b", zeroline=False),
                height=450, margin=dict(t=10),
            )
            st.plotly_chart(fig4, use_container_width=True)