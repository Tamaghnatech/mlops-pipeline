import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
import requests as req
from pathlib import Path

st.set_page_config(
    page_title="MLOps Monitoring Dashboard",
    page_icon="📡",
    layout="wide"
)

st.markdown("""
<style>
    .alert-high {
        background: #3d1515;
        border-left: 4px solid #ff4444;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
    }
    .alert-ok {
        background: #0d3320;
        border-left: 4px solid #00cc66;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
    }
    div[data-testid="metric-container"] {
        background: #1e2130;
        border: 1px solid #2d3250;
        border-radius: 10px;
        padding: 15px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("# 📡 MLOps Monitoring Dashboard")
st.markdown("Real-time model health, drift detection, and retraining history.")
st.markdown("---")

drift_summary_path = Path("reports/drift/latest_drift_summary.json")
drift_reports_dir  = Path("reports/drift")
GITHUB_RAW = "https://raw.githubusercontent.com/Tamaghnatech/mlops-pipeline/master"

def load_drift_summary():
    if drift_summary_path.exists():
        return json.loads(drift_summary_path.read_text())
    try:
        r = req.get(f"{GITHUB_RAW}/reports/drift/latest_drift_summary.json", timeout=10)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return None

def load_all_drift_reports():
    history_path = drift_reports_dir / "drift_history.json"
    if history_path.exists():
        try:
            return json.loads(history_path.read_text())
        except:
            pass
    try:
        r = req.get(f"{GITHUB_RAW}/reports/drift/drift_history.json", timeout=10)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return []

def load_metrics():
    metrics_path = Path("reports/metrics.json")
    if metrics_path.exists():
        return json.loads(metrics_path.read_text())
    try:
        r = req.get(f"{GITHUB_RAW}/reports/metrics.json", timeout=10)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return None

summary     = load_drift_summary()
all_reports = load_all_drift_reports()
metrics     = load_metrics()

col1, col2, col3, col4, col5 = st.columns(5)

if summary:
    drift_pct = summary.get("drift_share", 0) * 100
    n_drifted = summary.get("n_drifted_features", 0)
    n_total   = summary.get("n_total_features", 0)
    retrained = summary.get("retraining_triggered", False)
    ts        = summary.get("timestamp", "N/A")

    col1.metric("Drift Share", f"{drift_pct:.1f}%",
                delta="⚠ High" if drift_pct > 20 else "✓ Normal")
    col2.metric("Drifted Features", f"{n_drifted}/{n_total}")
    col3.metric("Last Check", ts[9:15] if len(ts) > 14 else ts)
    col4.metric("Auto Retrain", "Triggered" if retrained else "Not needed")
    col5.metric("Total Runs", len(all_reports))
else:
    st.warning("No drift reports found. Run the drift monitor first.")

st.markdown("---")

left, right = st.columns(2)

with left:
    st.markdown("### Drift History")
    if all_reports:
        df = pd.DataFrame(all_reports)
        df["drift_pct"] = df["drift_share"] * 100
        df["run"] = range(1, len(df) + 1)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["run"], y=df["drift_pct"],
            mode="lines+markers",
            line=dict(color="#6c63ff", width=2),
            marker=dict(size=8, color=[
                "#ff4444" if v > 20 else "#00cc66"
                for v in df["drift_pct"]
            ]),
            name="Drift %"
        ))
        fig.add_hline(y=20, line_dash="dash", line_color="#ffaa00",
                      annotation_text="Threshold (20%)")
        fig.update_layout(
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            font_color="white",
            xaxis_title="Run #",
            yaxis_title="Drift Share (%)",
            height=300,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No history yet.")

with right:
    st.markdown("### Feature Drift Breakdown")
    if summary:
        n_drifted = summary.get("n_drifted_features", 0)
        n_ok      = summary.get("n_total_features", 0) - n_drifted
        fig = go.Figure(go.Pie(
            labels=["Drifted", "Stable"],
            values=[n_drifted, n_ok],
            marker_colors=["#ff4444", "#00cc66"],
            hole=0.5
        ))
        fig.update_layout(
            paper_bgcolor="#0e1117",
            font_color="white",
            height=300,
            margin=dict(l=20, r=20, t=20, b=20),
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

st.markdown("### All Drift Reports")
if all_reports:
    df_table = pd.DataFrame(all_reports)
    df_table["drift_pct"] = (df_table["drift_share"] * 100).round(1).astype(str) + "%"
    df_table["status"] = df_table["drift_share"].apply(
        lambda x: "ALERT" if x > 0.2 else "OK"
    )
    display_cols = ["timestamp", "drift_pct", "n_drifted_features",
                    "n_total_features", "status", "retraining_triggered"]
    display_cols = [c for c in display_cols if c in df_table.columns]
    st.dataframe(df_table[display_cols], use_container_width=True)
else:
    st.info("No reports yet.")

st.markdown("---")

st.markdown("### Latest Training Metrics")
if metrics:
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{metrics.get('accuracy', 0):.1%}")
    c2.metric("F1 Score", f"{metrics.get('f1', 0):.1%}")
    c3.metric("ROC AUC",  f"{metrics.get('roc_auc', 0):.1%}")
else:
    st.info("No training metrics found.")

st.markdown("---")

st.markdown("### Latest Evidently Drift Report")
if summary:
    report_path = Path(summary.get("report_path", ""))
    if report_path.exists():
        st.markdown(f"Report saved at: `{report_path}`")
        with open(report_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=600, scrolling=True)
    else:
        st.info("Evidently HTML report only available when running locally.")

st.markdown("---")
st.markdown(
    "<p style='color: #888; font-size: 12px;'>Data loaded from GitHub. "
    "Run <code>python monitoring/drift_monitoring.py --simulate</code> "
    "and push to update this dashboard.</p>",
    unsafe_allow_html=True
)