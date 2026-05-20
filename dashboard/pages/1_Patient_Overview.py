import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import sys
from pathlib import Path
 
sys.path.append(str(Path(__file__).parent.parent.parent))
from shared.data_loader import load_predictions, get_patient, get_patient_list
 
st.set_page_config(page_title="AegisLife | Patient Profile", layout="wide")
 
try:
    sys.path.append(str(Path(__file__).parent.parent))
    from shared.styles import apply_global_style
    apply_global_style()
except:
    pass
 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
.stApp { background-color: #F8FAFC; }
 
/* ── Sidebar selectbox: main control ── */
[data-testid="stSidebar"] [data-baseweb="select"] > div:first-child {
    background-color: #1E293B !important;
    border: 1.5px solid #3B82F6 !important;
    border-radius: 10px !important;
    color: #F1F5F9 !important;
}
 
/* Selected value text */
[data-testid="stSidebar"] [data-baseweb="select"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-baseweb="select"] span,
[data-testid="stSidebar"] [data-baseweb="select"] div {
    color: #F1F5F9 !important;
}
 
/* Dropdown chevron icon */
[data-testid="stSidebar"] [data-baseweb="select"] svg {
    fill: #3B82F6 !important;
}
 
/* Dropdown options list */
[data-baseweb="popover"] ul {
    background-color: #1E293B !important;
    border: 1px solid #3B82F6 !important;
    border-radius: 10px !important;
}
 
/* Individual option */
[data-baseweb="popover"] li {
    background-color: #1E293B !important;
    color: #F1F5F9 !important;
}
 
/* Hovered option */
[data-baseweb="popover"] li:hover {
    background-color: #3B82F6 !important;
    color: #ffffff !important;
}
 
/* Selected / active option */
[data-baseweb="popover"] li[aria-selected="true"] {
    background-color: #2563EB !important;
    color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)
 
# ── Sidebar selector ──────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏥 AegisLife Context")
    patients = get_patient_list(split="test")
    selected = st.selectbox("🎯 Select Patient ID", patients,
                             label_visibility="collapsed")
    st.divider()
    st.info("AETHER Diagnostic Mode")
 
if not selected:
    st.warning("Please select a patient ID from the sidebar.")
    st.stop()
 
# Store in session state for other pages
st.session_state["selected_patient"] = str(selected)
 
data = get_patient(str(selected))
if not data:
    st.error(f"Patient {selected} not found.")
    st.stop()
 
name   = data.get("name",   "Unknown Patient")
gender = data.get("gender", "N/A")
age    = data.get("age",    "N/A")
 
# ── Header ────────────────────────────────────────────────────
st.markdown(f"""
<div style="background:linear-gradient(135deg,#0A1628 0%,#1E293B 100%);
            color:white;padding:2rem 2.5rem;border-radius:20px;
            border-left:8px solid #3B82F6;margin-bottom:1.5rem;">
    <div style="font-size:0.75rem;opacity:0.7;letter-spacing:1px;
                text-transform:uppercase">Active Case File</div>
    <div style="font-size:2rem;font-weight:800;margin:0.2rem 0">{name}</div>
    <div style="font-size:1rem;opacity:0.85">
        ID: {selected} &nbsp;|&nbsp; Gender: {gender}
        &nbsp;|&nbsp; Age: {age}
        &nbsp;|&nbsp; Source: MIMIC-IV
    </div>
</div>
""", unsafe_allow_html=True)
 
# ── Alert banner ──────────────────────────────────────────────
max_risk = max(data.get("risk_ckd",0),
               data.get("risk_sepsis",0),
               data.get("risk_nafld",0))
if max_risk >= 0.65:
    st.error("🚨 HIGH RISK — Immediate clinical review recommended.")
elif max_risk >= 0.35:
    st.warning("⚠️ MODERATE RISK — Schedule follow-up within 30 days.")
else:
    st.success("✅ LOW RISK — Continue routine monitoring.")
 
# ── Risk metrics ──────────────────────────────────────────────
m1, m2, m3 = st.columns(3)
m1.metric("Chronic Kidney Disease",
          f"{data.get('risk_ckd',0):.1%}",
          f"±{data.get('ci_ckd',0.05):.0%} CI",
          delta_color="off")
m2.metric("Sepsis Susceptibility",
          f"{data.get('risk_sepsis',0):.1%}",
          f"±{data.get('ci_sepsis',0.05):.0%} CI",
          delta_color="off")
m3.metric("Fatty Liver (NAFLD)",
          f"{data.get('risk_nafld',0):.1%}",
          f"±{data.get('ci_nafld',0.05):.0%} CI",
          delta_color="off")
 
st.divider()
 
# ── Multi-modal branch breakdown ──────────────────────────────
st.markdown("### 🛠️ Multi-Modal Evidence Architecture")
st.caption("Risk scores from each AI branch before AETHER fusion.")
 
branches = ["Branch 1\n(XGBoost)", "Branch 2\n(LSTM)",
            "Branch 3\n(BERT)", "Branch 4\n(LLM)"]
 
ckd_scores = [data.get(f"b{i}_ckd",    data.get("risk_ckd",0)) for i in range(1,5)]
sep_scores = [data.get(f"b{i}_sepsis", data.get("risk_sepsis",0)) for i in range(1,5)]
naf_scores = [data.get(f"b{i}_nafld",  data.get("risk_nafld",0)) for i in range(1,5)]
 
fig = go.Figure()
fig.add_trace(go.Bar(name="CKD",    x=branches, y=[v*100 for v in ckd_scores], marker_color="#1E3A8A"))
fig.add_trace(go.Bar(name="Sepsis", x=branches, y=[v*100 for v in sep_scores], marker_color="#EF4444"))
fig.add_trace(go.Bar(name="NAFLD",  x=branches, y=[v*100 for v in naf_scores], marker_color="#10B981"))
 
fig.update_layout(
    barmode="group", height=360,
    yaxis=dict(title="Risk Probability (%)", range=[0, 100]),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#F8FAFC",
    legend=dict(orientation="h", y=-0.2),
    font=dict(family="Plus Jakarta Sans"),
    margin=dict(t=10, b=10)
)
st.plotly_chart(fig, use_container_width=True)
 
st.divider()
 
# ── Population context scatter ────────────────────────────────
st.markdown("### 📊 Population Context")
st.caption("Where this patient sits within the full test cohort.")

df_all = load_predictions(split="test")

if not df_all.empty:
    # Force float — this is the critical fix
    x_vals = df_all["risk_ckd"].astype(float).tolist()
    y_vals = df_all["risk_nafld"].astype(float).tolist()
    c_vals = df_all["risk_sepsis"].astype(float).tolist()
    p_vals = df_all["patient_id"].astype(str).tolist()

    fig2 = go.Figure()

    # All patients as scatter
    fig2.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode="markers",
        marker=dict(
            size=5,
            color=c_vals,
            colorscale="RdYlBu_r",
            opacity=0.65,
            colorbar=dict(
                title="Sepsis Risk",
                tickformat=".0%",
                thickness=12
            ),
            showscale=True
        ),
        text=[f"ID: {p}<br>CKD: {x:.1%}<br>NAFLD: {y:.1%}<br>Sepsis: {c:.1%}"
              for p, x, y, c in zip(p_vals, x_vals, y_vals, c_vals)],
        hoverinfo="text",
        name="All Patients"
    ))

    # Selected patient as gold star
    fig2.add_trace(go.Scatter(
        x=[float(data.get("risk_ckd", 0))],
        y=[float(data.get("risk_nafld", 0))],
        mode="markers",
        marker=dict(
            size=20, color="#FBBF24",
            symbol="star",
            line=dict(color="#0F172A", width=2)
        ),
        text=[f"Selected: {selected}"],
        hoverinfo="text",
        name=f"Selected: {selected}"
    ))

    fig2.update_layout(
        height=420,
        xaxis=dict(
            title="CKD Risk",
            range=[0, 1],
            tickformat=".0%",
            showgrid=True,
            gridcolor="#E2E8F0"
        ),
        yaxis=dict(
            title="NAFLD Risk",
            range=[0, 1],
            tickformat=".0%",
            showgrid=True,
            gridcolor="#E2E8F0"
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#F8FAFC",
        legend=dict(orientation="h", y=-0.15),
        font=dict(family="Plus Jakarta Sans", size=12),
        margin=dict(t=10, b=10)
    )
    st.plotly_chart(fig2, use_container_width=True)
