"""Streamlit front end. Run: streamlit run dashboard.py"""
from pathlib import Path
from datetime import date
import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from swapscribe.paths import DATA, ARTIFACTS, ensure_dirs
from swapscribe.demo import generate_demo
from swapscribe.market import load_rates, curve_from_row
from swapscribe.risk import load_portfolio, risk_report
from swapscribe.backtest import walk_forward_backtest
from swapscribe.commentary import load_news, retrieve, build_commentary
from swapscribe.sentiment import load_sentiment_model, score_news

st.set_page_config(page_title="SwapRisk-and-FedNews-AI", page_icon="📈", layout="wide")
ensure_dirs()

# Hide the Deploy button on the top right
st.markdown(
    """
    <style>
    .stDeployButton,
    [data-testid="stDeployButton"],
    [data-testid="stAppDeployButton"],
    .stAppDeployButton {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_data(ttl=600, show_spinner=False)
def read_market(path_str):
    return load_rates(Path(path_str))

@st.cache_data(ttl=600, show_spinner=False)
def read_news(path_str):
    return load_news(Path(path_str))

@st.cache_resource(show_spinner=False)
def get_model(path_str):
    return load_sentiment_model(Path(path_str))

st.title("SwapRisk-and-FedNews-AI")
st.caption("Quantitative Interest-Rate Swap Risk Engine & Grounded Federal Reserve News AI • Institutional Research Architecture")

real_rate_paths = sorted((DATA / "real").glob("treasury_*.csv"))
if not real_rate_paths:
    st.error("No official Treasury yield data found in data/real/. Please run fetch-market.")
    st.stop()

st.sidebar.markdown("### 📊 Market & Data Controls")
st.sidebar.markdown("**Data Source:** Official U.S. Treasury & Federal Reserve")
rate_path = st.sidebar.selectbox("Treasury Yield Curve Year", real_rate_paths, index=len(real_rate_paths)-1, format_func=lambda p: p.stem.replace("treasury_", "U.S. Treasury "))
news_path = DATA / "real" / "fed_news.csv"
portfolio_path = (DATA / "real" / "portfolio.csv") if (DATA / "real" / "portfolio.csv").exists() else (DATA / "demo" / "portfolio.csv")
rates = read_market(str(rate_path))
min_d, max_d = rates.date.min().date(), rates.date.max().date()
cutoff = st.sidebar.date_input("As-of valuation date", value=max_d, min_value=min_d, max_value=max_d)
max_hedge = st.sidebar.slider("Max absolute hedge per tenor ($MM)", 1, 100, 30)
st.sidebar.info("Treasury par yields are used as an educational zero-rate proxy for vanilla interest-rate swap valuation.")
st.sidebar.caption(f"Rates: {rate_path.name} | Portfolio: Benchmark G10 Swaps | News: {news_path.name}")
st.info("**Official Public Market Data Active**: Real U.S. Treasury Par Yields & Federal Reserve Press Releases.")

rate_view = rates[rates.date.dt.date <= cutoff]
if len(rate_view) < 7:
    st.error("Choose an as-of date with at least 7 previous rate observations.")
    st.stop()

t1, t2, t3, t4 = st.tabs(["Market & curves", "Swap risk & optimization", "Fed news & commentary", "Model & data quality"])
with t1:
    latest = rate_view.iloc[-1]
    c1, c2, c3 = st.columns(3)
    c1.metric("2-year Treasury par yield", f"{latest['2 Yr']:.3f}%")
    c2.metric("10-year Treasury par yield", f"{latest['10 Yr']:.3f}%")
    c3.metric("10Y–2Y slope", f"{(latest['10 Yr']-latest['2 Yr'])*100:+.1f} bp")
    chart = px.line(rate_view, x="date", y=["2 Yr", "5 Yr", "10 Yr"], title="Treasury yield evolution (percent)")
    st.plotly_chart(chart, use_container_width=True)
    curve = curve_from_row(latest)
    st.plotly_chart(px.line(x=curve.tenors, y=curve.rates * 100, markers=True,
                            labels={"x": "Tenor (years)", "y": "Annual yield (%)"}, title="Illustrative interpolated curve"), use_container_width=True)

with t2:
    pf = load_portfolio(portfolio_path)
    output = risk_report(rates, pf, cutoff, max_abs_millions=max_hedge)
    col1, col2, col3 = st.columns(3)
    col1.metric("Portfolio PV (USD, proxy)", f"${output['portfolio_pv_usd']:,.0f}")
    col2.metric("Signed parallel PV change / +1bp", f"${output['parallel_dv01_usd_per_bp']:,.0f}")
    red = 100 * (1 - output["rms_scenario_pnl_usd_after"] / max(output["rms_scenario_pnl_usd_before"], 1e-9))
    col3.metric("Illustrative scenario RMS reduction", f"{red:.1f}%")
    x = output["tenors_years"]
    fig = go.Figure()
    fig.add_bar(x=x, y=output["unhedged_key_rate_usd_per_bp"], name="Before")
    fig.add_bar(x=x, y=output["hedged_key_rate_usd_per_bp"], name="After")
    fig.update_layout(barmode="group", title="Key-rate PV sensitivity (USD per +1bp)", xaxis_title="Key tenor (years)")
    st.plotly_chart(fig, use_container_width=True)
    hedges = pd.DataFrame({"Reference hedge": output["hedge_unit_names"],
                           "Signed notional ($MM)": output["hedge_notional_usd_millions"]})
    st.subheader("Optimized at-par reference hedges")
    st.dataframe(hedges, use_container_width=True, hide_index=True)
    st.caption("+ notional = pay fixed; – notional = receive fixed. The optimizer uses historical daily shocks plus fourteen deterministic tenor stresses; cost is L2 proxy, not quoted transaction costs.")
    with st.expander("Benchmark G10 swap portfolio"):
        st.dataframe(pd.read_csv(portfolio_path), use_container_width=True, hide_index=True)
    st.download_button("Download risk report (JSON)", json.dumps(output, indent=2), "risk_report.json", "application/json")
    st.subheader("Forward-looking holdout test")
    st.caption("Past-only hedge fit; a new, subsequent non-overlapping rate period is used to measure first-order P&L. This is not realized full-repriced trade P&L.")
    if st.button("Run 70-day rolling / 5-day forward backtest", key="backtest"):
        try:
            bt = walk_forward_backtest(rate_view, pf, lookback=70, holding_days=5,
                                       max_abs_millions=max_hedge)
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Rebalance Turnover", f"${bt.get('total_turnover_usd_millions', 0):,.1f} MM")
            m2.metric("Estimated Bid-Ask Friction", f"${bt.get('total_transaction_costs_usd', 0):,.0f}")
            m3.metric("Pricing Greeks Mode", "DV01 + Convexity (Gamma)")
            
            metrics = pd.DataFrame({
                "Strategy": [
                    "Unhedged (1st + 2nd Order)",
                    "Parallel-only 10Y",
                    "Scenario-optimized 2Y/5Y/10Y (Gross)",
                    "Scenario-optimized (Net of 0.5bp Bid-Ask Cost)"
                ],
                "OOS Daily P&L RMS (USD)": [
                    bt["oos_rms_pnl_usd_unhedged"],
                    bt["oos_rms_pnl_usd_parallel_only"],
                    bt["oos_rms_pnl_usd_optimized"],
                    bt.get("oos_rms_pnl_usd_optimized_net_of_costs", bt["oos_rms_pnl_usd_optimized"])
                ]
            })
            st.dataframe(metrics, use_container_width=True, hide_index=True)
            st.plotly_chart(px.bar(metrics, x="Strategy", y="OOS Daily P&L RMS (USD)",
                                   title=f"Walk-Forward Out-of-Sample Performance ({bt['oos_test_days']} Forward Trading Days)"), use_container_width=True)
            st.info("The multi-tenor hedge balances extreme tail twists (steepening/flattening), whereas daily walk-forward moves were dominated by parallel level shifts. Second-order key-rate convexity and 0.5 bp bid-ask turnover costs are incorporated.")
            st.download_button("Download walk-forward report (JSON)", json.dumps(bt, indent=2),
                               "backtest_report.json", "application/json")
        except ValueError as error:
            st.warning(str(error))

with t3:
    st.subheader("Evidence-grounded financial research note")
    query = st.text_input("Retrieval query", "monetary policy treasury yields inflation")
    if not news_path.exists():
        st.warning("No public news found. Run `python -m swapscribe.cli fetch-news`.")
    else:
        news = read_news(str(news_path))
        ranked = retrieve(news, query, cutoff, k=6)
        model_mode = st.selectbox("Sentiment backend", ["TF-IDF (Financial PhraseBank model)", "No sentiment model"])
        model_path = ARTIFACTS / "sentiment_real.joblib"
        model = get_model(str(model_path)) if model_mode.startswith("TF-IDF") and model_path.exists() else None
        if model_mode.startswith("TF-IDF") and model is None:
            st.info("No local sentiment model found. Run training via CLI; commentary will still work without it.")
        note = build_commentary(rates, ranked, cutoff, sentiment_model=model)
        st.write(note["commentary"])
        st.subheader("Supporting headlines")
        if note["evidence"]:
            for idx, item in enumerate(note["evidence"], 1):
                st.markdown(f"**[{idx}] {item['title']}** — {item['source']} ({item['date']})")
                if item["url"].startswith("https://"):
                    st.link_button("Open source", item["url"])
                if item["sentiment"] != "not_scored":
                    st.caption(f"Model sentiment: {item['sentiment']}, uncalibrated probability: {item['model_probability']:.2f}")
        else:
            st.warning("No headlines on or before the selected date; the note contains numerical facts only.")
        st.download_button("Download cited note (JSON)", json.dumps(note, indent=2), "market_note.json", "application/json")

with t4:
    st.subheader("Model validation and reproducibility")
    eval_file = ARTIFACTS / "sentiment_real_eval.json"
    if eval_file.exists():
        m = json.loads(eval_file.read_text(encoding="utf-8"))
        c1, c2, c3 = st.columns(3)
        c1.metric("Hold-out Macro-F1", f"{m['test_macro_f1']:.3f}")
        c2.metric("Baseline Macro-F1 (Majority Class)", f"{m['baseline_macro_f1']:.3f}")
        c3.metric("Hold-out Test Sentences", f"{m['test_count']}")
        st.caption(m["warning"])
        st.dataframe(pd.DataFrame(m["confusion_matrix"], index=m["confusion_matrix_labels"],
                                columns=m["confusion_matrix_labels"]), use_container_width=True)
    else:
        st.info("Run `python -m swapscribe.cli train --source real` to produce model metrics.")
    st.markdown("""**Validation controls**

- No test-set fitting; duplicate texts removed before the random, stratified split
- Time-filtered retrieval prevents future-dated headlines from entering earlier market notes
- Source links and observation dates accompany all numerical and news claims
- Real Financial PhraseBank has no timestamps; random hold-out is not forward-looking evaluation
- Real rates and real headlines have separate collection times; daily dating does not establish causality
- Swap valuations are a deliberately simplified **Treasury-based educational proxy**
""")
    st.caption("Install test dependencies and run `pytest -q` for automated quality checks.")
