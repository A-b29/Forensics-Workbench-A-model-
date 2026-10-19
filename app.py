"""
app.py - Streamlit Visual Workbench for Model Forensics.
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from src.schema import validate_dataframe
from src.metrics import compute_aggregate_metrics, compute_per_class_metrics
from src.subgroup_analysis import analyze_subgroups
from src.report import generate_forensics_markdown, export_forensics_json

st.set_page_config(
    page_title="Model Forensics Workbench",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Model Forensics Workbench")
st.markdown("Uncover hidden failure modes, systematic demographic/device disparities, and uncalibrated errors.")

st.info(
    "💡 **Non-Causal Notice:** Identified error clusters represent observational associations and exploratory hypotheses, "
    "not causal mechanisms."
)

project_root = Path(__file__).resolve().parent
example_csv = project_root / "examples" / "sample_predictions.csv"

# Sidebar: Data Source
st.sidebar.header("1. Ingestion")
data_source = st.sidebar.radio("Dataset Source:", ["Demo Dataset (Chest X-Ray)", "Upload Custom CSV"])

df = None
if data_source == "Demo Dataset (Chest X-Ray)":
    if example_csv.exists():
        df = pd.read_csv(example_csv)
    else:
        st.error("Sample dataset not found.")
        st.stop()
else:
    uploaded = st.sidebar.file_uploader("Upload Predictions CSV", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)

if df is None:
    st.info("Please load or upload a dataset to begin.")
    st.stop()

# Schema Validation
val_res = validate_dataframe(df)
if not val_res.is_valid:
    st.error("❌ Schema Validation Failed:")
    for err in val_res.error_messages:
        st.write(f"- {err}")
    st.stop()

st.sidebar.success(f"✅ Schema Valid: {val_res.total_samples} samples, {val_res.num_classes} classes")

min_subgroup_size = st.sidebar.slider("Minimum Subgroup Size (for stability):", 5, 50, 15)

# 1. Overall Performance Metrics
agg = compute_aggregate_metrics(df)
per_class = compute_per_class_metrics(df)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Overall Accuracy", f"{agg['overall_accuracy']:.1%}")
col2.metric("Overall Error Rate", f"{agg['overall_error_rate']:.1%}")
col3.metric("Balanced Accuracy", f"{agg['balanced_accuracy']:.1%}")
col4.metric("High-Conf Errors (>80%)", f"{agg.get('high_confidence_errors', 'N/A')}")

st.divider()

# 2. Subgroup Disparity Analysis
st.subheader("📊 Subgroup Failure Analysis")
if not val_res.metadata_columns:
    st.warning("No metadata columns detected for subgroup slicing. Add metadata columns (e.g. age, scanner, view).")
else:
    selected_col = st.selectbox("Select Metadata Dimension to Disaggregate:", val_res.metadata_columns)
    subgroups = analyze_subgroups(df, selected_col, min_subgroup_size=min_subgroup_size)

    # Plotly Bar chart with Wilson Score error bars
    categories = [g["subgroup_value"] for g in subgroups]
    error_rates = [g["error_rate"] * 100 for g in subgroups]
    err_low = [(g["error_rate"] - g["ci_95_lower"]) * 100 for g in subgroups]
    err_high = [(g["ci_95_upper"] - g["error_rate"]) * 100 for g in subgroups]
    colors = ['#f59e0b' if g['is_unstable_small_cohort'] else '#dc2626' for g in subgroups]

    fig_sub = go.Figure()
    fig_sub.add_trace(go.Bar(
        x=categories,
        y=error_rates,
        error_y=dict(
            type='data',
            symmetric=False,
            array=err_high,
            arrayminus=err_low
        ),
        marker_color=colors,
        text=[f"{e:.1f}% (N={g['sample_count']})" for e, g in zip(error_rates, subgroups)],
        textposition='auto',
    ))

    # Population baseline line
    pop_error_pct = agg["overall_error_rate"] * 100
    fig_sub.add_hline(
        y=pop_error_pct,
        line_dash="dash",
        line_color="#4b5563",
        annotation_text=f"Population Baseline ({pop_error_pct:.1f}%)"
    )

    fig_sub.update_layout(
        title=f"Error Rate by {selected_col} with 95% Wilson Score Confidence Bounds",
        xaxis_title=selected_col,
        yaxis_title="Error Rate (%)",
        yaxis_range=[0, max(max(error_rates) + 20, 50)],
        height=380
    )
    st.plotly_chart(fig_sub, use_container_width=True)

    df_sub = pd.DataFrame(subgroups)[[
        "subgroup_value", "sample_count", "error_count", "error_rate",
        "ci_95_lower", "ci_95_upper", "risk_ratio", "is_unstable_small_cohort"
    ]]
    df_sub["error_rate"] = df_sub["error_rate"].map(lambda x: f"{x:.1%}")
    df_sub["95% Wilson CI"] = df_sub.apply(lambda r: f"[{r['ci_95_lower']:.1%}, {r['ci_95_upper']:.1%}]", axis=1)
    df_sub["risk_ratio"] = df_sub["risk_ratio"].map(lambda x: f"{x:.2f}x")
    df_sub["Cohort Flag"] = df_sub["is_unstable_small_cohort"].map(lambda x: "⚠️ Small Cohort (< Min N)" if x else "Reliable")
    
    st.dataframe(df_sub[["subgroup_value", "sample_count", "error_count", "error_rate", "95% Wilson CI", "risk_ratio", "Cohort Flag"]], use_container_width=True)

st.divider()

# 3. Confidence vs Correctness Distribution
if val_res.has_confidence:
    st.subheader("🎯 Confidence vs. Error Distribution")
    col_c1, col_c2 = st.columns(2)
    
    with col_c1:
        df["_Outcome"] = np.where(df["y_true"] == df["y_pred"], "Correct", "Error")
        fig_hist = px.histogram(
            df,
            x="confidence",
            color="_Outcome",
            barmode="overlay",
            nbins=20,
            color_discrete_map={"Correct": "#16a34a", "Error": "#dc2626"},
            title="Prediction Confidence by Correctness"
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_c2:
        st.markdown("#### Calibration Insights")
        st.write(f"- **Mean Confidence (Correct Cases):** `{agg.get('mean_conf_correct', 0.0):.2%}`")
        st.write(f"- **Mean Confidence (Misclassifications):** `{agg.get('mean_conf_incorrect', 0.0):.2%}`")
        st.write(f"- **Expected Calibration Error (ECE):** `{agg.get('ece', 0.0):.4f}`")
        st.write(
            "Overconfident errors indicate cases where the model fails silently without lowering confidence. "
            "These warrant high priority during qualitative review."
        )

# 4. Error Gallery / Inspection Table
st.divider()
st.subheader("🔎 Error Case Inspection")
errors_df = df[df["y_true"] != df["y_pred"]].copy()
st.write(f"Showing **{len(errors_df)}** misclassified cases out of {len(df)} total.")
st.dataframe(errors_df, use_container_width=True)

# Export Report
st.divider()
st.subheader("📄 Export Audit Artifacts")
if st.button("Generate Markdown & JSON Reports"):
    all_subgroups = {c: analyze_subgroups(df, c, min_subgroup_size=min_subgroup_size) for c in val_res.metadata_columns}
    out_dir = project_root / "reports"
    md_path = generate_forensics_markdown(agg, per_class, all_subgroups, out_dir / "forensics_report.md")
    json_path = export_forensics_json({"agg": agg, "per_class": per_class, "subgroups": all_subgroups}, out_dir / "forensics_findings.json")
    st.success(f"Reports saved to `{md_path.name}` and `{json_path.name}`!")
