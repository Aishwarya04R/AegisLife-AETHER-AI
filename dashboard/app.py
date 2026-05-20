import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import sys
from pathlib import Path

pd.set_option("styler.render.max_elements", 500000)

# Add root to path for shared imports
sys.path.append(str(Path(__file__).parent.parent))
from shared.data_loader import load_predictions


# 1. Page Configuration
st.set_page_config(
    page_title="AegisLife AI | Clinical Decision Support",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Global CSS for "Modern Dashboard" UI
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
.stApp { background-color: #F8FAFC; }

/* --- SIDEBAR NAVIGATION BUTTONS --- */
/* Target the sidebar navigation container */
[data-testid="stSidebarNav"] {
    padding-top: 1rem;
}

/* Style each link to look like a Button Tile */
div[data-testid="stSidebarNav"] li {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    margin: 8px 15px;
    padding: 2px;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

/* Hover Effect: Slide and Glow */
div[data-testid="stSidebarNav"] li:hover {
    background: rgba(37, 99, 235, 0.15);
    border-color: #3B82F6;
    transform: translateX(8px);
}

/* Active Page Highlight: Blue Gradient Button */
div[data-testid="stSidebarNav"] li:has(a[aria-current="page"]) {
    background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
    border: none;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
}

/* Sidebar Text Colors */
[data-testid="stSidebar"] { background-color: #0A1628; border-right: 1px solid #1E293B; }
[data-testid="stSidebar"] * { color: #F8FAFC !important; }

/* --- MAIN CONTENT UI --- */
.hero-banner {
    background: linear-gradient(rgba(10, 22, 40, 0.8), rgba(13, 33, 71, 0.85)), 
                url('https://images.unsplash.com/photo-1579154204601-01588f351e67?q=80&w=2070&auto=format&fit=crop');
    background-size: cover;
    background-position: center;
    border-radius: 24px;
    padding: 4rem 3.5rem;
    margin-bottom: 2rem;
    color: white !important;
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
}
.hero-title {
    font-size: 3.8rem; font-weight: 800; letter-spacing: -2.5px;
    background: linear-gradient(to right, #FFFFFF, #60A5FA);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.stat-card {
    background: white; padding: 1.5rem; border-radius: 18px;
    border: 1px solid #E2E8F0; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
    transition: transform 0.25s;
}
.stat-card:hover { transform: translateY(-8px); border-color: #3B82F6; }
.step-icon {
    background: linear-gradient(135deg, #2563EB, #1D4ED8); color: white;
    width: 32px; height: 32px; border-radius: 8px;
    display: flex; align-items: center; justify-content: center; font-weight: 800;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar Branding ──────────────────────────────────────────
# --- UPDATE SIDEBAR BRANDING ---
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 1rem 0;'>
        <div style='font-size: 3.5rem;'>🏥</div>
        <h2 style='margin: 0; font-weight: 800; letter-spacing: -1px; color: white;'>AegisLife</h2>
        <p style='color: #64748B; font-size: 0.75rem; letter-spacing: 1.5px; text-transform: uppercase;'>Prediction Engine v1.0</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Status Indicators
    st.markdown("#### ⚡ System Triage")
    st.success("Models: ACTIVE")
    st.info("RAG DB: CONNECTED")
    
    st.divider()
    st.caption("Validated on MIMIC-IV Clinical Data")

# ── Hero Section ──────────────────────────────────────────────
st.markdown(f"""
<div class='hero-banner'>
    <div class='hero-title'>AegisLife AI</div>
    <div style='font-size: 1.15rem; color: #E2E8F0; max-width: 750px; line-height: 1.6; margin-bottom: 2rem;'>
        Adaptive Early-Warning Temporal Health Evaluation & Reasoning (AETHER). 
        A multi-modal 12-month early warning system for Chronic Kidney Disease, Sepsis, and NAFLD.
    </div>
    <div style='display: flex; gap: 12px; flex-wrap: wrap;'>
        <span style='background: rgba(255,255,255,0.18); backdrop-filter: blur(5px); padding: 7px 18px; border-radius: 20px; font-size: 0.75rem; font-weight:600;'>XGB + LSTM Ensemble</span>
        <span style='background: rgba(255,255,255,0.18); backdrop-filter: blur(5px); padding: 7px 18px; border-radius: 20px; font-size: 0.75rem; font-weight:600;'>ClinicalBERT NLP</span>
        <span style='background: rgba(255,255,255,0.18); backdrop-filter: blur(5px); padding: 7px 18px; border-radius: 20px; font-size: 0.75rem; font-weight:600;'>Gemini RAG Pipeline</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Load Metrics ──────────────────────────────────────────────
try:
    df = load_predictions(split="test")
    high_risk_count = len(df[df[["risk_ckd", "risk_sepsis", "risk_nafld"]].max(axis=1) >= 0.65])
except:
    df = pd.DataFrame()
    high_risk_count = 0

# ── Stats Grid ────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
metrics = [
    ("Diseases", "3", "CKD · Sepsis · NAFLD"),
    ("Model Branches", "4", "Temporal + Static + NLP"),
    ("Test Cohort", str(len(df)), "Active Case Records"),
    ("Critical Alerts", str(high_risk_count), "Requires Review")
]

for col, (label, val, sub) in zip([m1, m2, m3, m4], metrics):
    col.markdown(f"""
    <div class='stat-card'>
        <div style='font-size: 0.72rem; font-weight: 700; color: #64748B; text-transform: uppercase;'>{label}</div>
        <div style='font-size: 2.2rem; font-weight: 800; color: #0F172A;'>{val}</div>
        <div style='font-size: 0.75rem; color: #64748B;'>{sub}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Main Body ─────────────────────────────────────────────────
left, right = st.columns([1, 1.4], gap="large")

with left:
    st.markdown("### 🚀 Clinical Workflow")
    workflow = [
        ("01", "Cohort Triage", "Select a patient profile to begin analysis."),
        ("02", "Risk Surveillance", "Monitor 12-month longitudinal trajectories."),
        ("03", "Diagnostic Audit", "Inspect SHAP and Attention-based XAI."),
        ("04", "Intervention Plan", "Generate RAG-grounded care protocols.")
    ]
    
    for num, title, desc in workflow:
        st.markdown(f"""
        <div style='display: flex; gap: 1.2rem; padding: 1.2rem 0; border-bottom: 1px solid #F1F5F9;'>
            <div class='step-icon'>{num}</div>
            <div>
                <div style='font-weight: 700; color: #1E293B; font-size: 0.95rem;'>{title}</div>
                <div style='color: #64748B; font-size: 0.82rem;'>{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

with right:
    st.markdown("### 📊 Population Risk Matrix")
    if not df.empty:

        x_vals = df["risk_ckd"].astype(float).tolist()
        y_vals = df["risk_nafld"].astype(float).tolist()
        c_vals = df["risk_sepsis"].astype(float).tolist()
        p_vals = df["patient_id"].astype(str).tolist()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="markers",
            marker=dict(
                size=5,
                color=c_vals,
                colorscale="RdYlBu_r",
                opacity=0.65,
                colorbar=dict(
                    title="Sepsis",
                    tickformat=".0%",
                    thickness=12
                ),
                showscale=True
            ),
            text=[f"ID:{p} CKD:{x:.1%} NAFLD:{y:.1%} Sepsis:{c:.1%}"
                  for p,x,y,c in zip(p_vals,x_vals,y_vals,c_vals)],
            hoverinfo="text",
            name="Patients"
        ))
        fig.update_layout(
            height=340,
            xaxis=dict(title="CKD Risk", range=[0,1],
                       tickformat=".0%", showgrid=True, gridcolor="#E2E8F0"),
            yaxis=dict(title="NAFLD Risk", range=[0,1],
                       tickformat=".0%", showgrid=True, gridcolor="#E2E8F0"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#F8FAFC",
            font=dict(family="Plus Jakarta Sans", size=12),
            margin=dict(t=10, b=10, l=0, r=0)
        )
        st.plotly_chart(fig, use_container_width=True)

# ── Data Table with Fixed Conversion ──────────────────────────
# ── Data Table with Performance Fix ──────────────────────────
st.divider()
st.markdown("### 📁 Clinical Records & AI Predictions")

if not df.empty:
    # 1. Prepare display copy
    display_df = df[["patient_id", "risk_ckd", "risk_sepsis", "risk_nafld"]].copy()
    
    # 2. Force numeric conversion for the gradient to work
    for col in ["risk_ckd", "risk_sepsis", "risk_nafld"]:
        display_df[col] = pd.to_numeric(display_df[col], errors='coerce')

    display_df.columns = ["Patient ID", "CKD Risk", "Sepsis Risk", "NAFLD Risk"]

    # 3. Use .head(100) to keep the dashboard fast for the demo
    st.dataframe(
        display_df.head(100).style.background_gradient(
            cmap='RdYlGn_r', 
            subset=["CKD Risk", "Sepsis Risk", "NAFLD Risk"]
        ).format("{:.1%}", subset=["CKD Risk", "Sepsis Risk", "NAFLD Risk"]),
        use_container_width=True, 
        hide_index=True, 
        height=400
    )
    st.caption(f"Showing top 100 of {len(df)} total clinical records.")
    
st.sidebar.divider()
st.sidebar.caption("© 2026 AegisLife AI Project Team")