"""Read-only experiment review app. Run: streamlit run app.py"""
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="RetailCast", page_icon="📈", layout="wide")


@st.cache_data
def load_results():
    report = ROOT / "reports"
    return {
        "summary": json.loads((report / "run_summary.json").read_text()),
        "history": pd.read_csv(ROOT / "data/processed/daily_sales.csv", index_col="date", parse_dates=True),
        "holdout": pd.read_csv(report / "holdout_predictions.csv", parse_dates=["date"]),
        "future": pd.read_csv(report / "forecast_28d.csv", parse_dates=["date"]),
        "validation": pd.read_csv(report / "validation_by_market.csv"),
        "market_scores": pd.read_csv(report / "holdout_by_market.csv"),
        "intervals": pd.read_csv(report / "interval_metrics.csv"),
    }


if not (ROOT / "reports/run_summary.json").exists():
    st.error("Run `retailcast run` to generate experiment artifacts first.")
    st.stop()
d = load_results()
st.caption("RETAILCAST / RESEARCH PORTFOLIO")
st.title("28-day retail sales forecast")
st.write("A reproducible historical forecasting experiment with locked model selection and an untouched holdout.")
market = st.sidebar.selectbox("Market", d["summary"]["markets"])
view = st.sidebar.radio("Review window", ["Final holdout", "Historical continuation"])
st.sidebar.info("Data ends in 2011. The continuation is an illustration after the historical endpoint, not a present-day forecast.")
source = d["holdout"] if view == "Final holdout" else d["future"]
g = source.loc[(source.market == market) & (source.model == "selected_champion")]
scores = d["market_scores"].loc[d["market_scores"].market == market].set_index("model")
champ, naive = scores.loc["selected_champion"], scores.loc["seasonal_naive"]
interval = d["intervals"].loc[d["intervals"].market == market].iloc[0]
c = st.columns(4)
c[0].metric("Holdout WAPE", f"{champ.WAPE:.1%}")
c[1].metric("WAPE reduction vs naive", f"{1-champ.WAPE/naive.WAPE:+.1%}")
c[2].metric("80% interval test coverage", f"{interval.coverage_80:.1%}")
c[3].metric("28-day forecast units", f"{g.prediction.sum():,.0f}")
tabs = st.tabs(["Forecast", "Validation and diagnostics", "Protocol"])
with tabs[0]:
    fig = go.Figure()
    context = d["history"].loc[g.date.min()-pd.Timedelta(days=42):g.date.max(), market]
    fig.add_trace(go.Scatter(x=g.date, y=g.upper_95, line={"width": 0}, showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=g.date, y=g.lower_95, fill="tonexty", fillcolor="rgba(0,150,125,.10)", line={"width": 0}, name="95% interval"))
    fig.add_trace(go.Scatter(x=g.date, y=g.upper_80, line={"width": 0}, showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=g.date, y=g.lower_80, fill="tonexty", fillcolor="rgba(0,150,125,.22)", line={"width": 0}, name="80% interval"))
    fig.add_trace(go.Scatter(x=context.index, y=context, name="Observed", line={"color": "#263e59"}))
    fig.add_trace(go.Scatter(x=g.date, y=g.prediction, name="Champion", line={"color": "#008c7a", "width": 3}))
    if view == "Final holdout":
        n = source.loc[(source.market == market) & (source.model == "seasonal_naive")]
        fig.add_trace(go.Scatter(x=n.date, y=n.prediction, name="Seasonal naive", line={"color": "#d49a42", "dash": "dash"}))
    fig.update_layout(template="plotly_white", height=480, hovermode="x unified", yaxis_title="Units sold", legend={"orientation": "h"})
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Locked algorithm: {d['summary']['selection'][market]}. All forecasts use information available at the origin.")
    columns = ["date", "horizon", "prediction", "lower_80", "upper_80", "lower_95", "upper_95"]
    if "actual" in g:
        columns.insert(2, "actual")
    st.dataframe(g[columns], hide_index=True, width="stretch")
    st.download_button("Export forecast CSV", g.to_csv(index=False), file_name=f"retailcast_{market}.csv", mime="text/csv")
with tabs[1]:
    st.subheader("Validation, not holdout, determines selection")
    st.dataframe(d["validation"].loc[d["validation"].market == market].sort_values("MASE"), hide_index=True)
    st.subheader("Holdout comparisons")
    st.dataframe(scores, width="stretch")
    st.image(str(ROOT / "reports/figures/error_diagnostics.png"))
with tabs[2]:
    st.markdown("""1. Select five markets using the initial training period only.
2. Evaluate six candidates on four expanding 28-day validation windows.
3. Lock a model for each market by mean validation MASE.
4. Use two later windows to calibrate per-market absolute-error intervals.
5. Evaluate the final 28-day holdout once, then refit the locked algorithms.

**Limits:** sales are not unconstrained demand. No inventory, promotion plan or causal model is available. Temporal dependence and seasonal shifts mean empirical intervals do not have guaranteed coverage. Individual daily bounds do not form a valid interval for the monthly total.

Source: [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii), Chen, D. (2012), CC BY 4.0.
""")
