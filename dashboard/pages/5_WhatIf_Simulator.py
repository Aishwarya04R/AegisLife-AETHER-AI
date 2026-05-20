import streamlit as st
import plotly.graph_objects as go
import numpy as np
import sys
from pathlib import Path

# Add root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from shared.data_loader import get_patient

# 1. Page Configuration
st.set_page_config(page_title="AegisLife | Simulation Lab", layout="wide")

sys.path.append(str(Path(__file__).parent.parent)) # Adjust path if in pages/
from shared.styles import apply_global_style
apply_global_style()

# 2. Professional Clinical CSS (Matching app.py)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
.stApp { background-color: #F8FAFC; }

/* Simulation Control Panel */
.sim-sidebar {
    background: white;
    padding: 2rem;
    border-radius: 20px;
    border: 1px solid #E2E8F0;
}

/* Result Card */
.impact-card {
    background: white;
    padding: 1.5rem;
    border-radius: 15px;
    border: 1px solid #E2E8F0;
    text-align: center;
    box-shadow: 0 4px 6px rgba(0,0,0,0.02);
}

/* Slider label styling */
.stSlider label {
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    color: #475569 !important;
}
</style>
""", unsafe_allow_html=True)

st.title("🧪 Clinical Simulation Lab")
st.caption("AETHER Counterfactual Reasoning | Analyzing Modifiable Risk Factors & Outcomes")

# ── Guard ─────────────────────────────────────────────────────
patient_id = st.session_state.get("selected_patient")
if not patient_id:
    st.warning("⚠️ Please select a patient on the Patient Overview page first.")
    st.stop()

data = get_patient(patient_id)
base_ckd = float(data["risk_ckd"])
base_sepsis = float(data["risk_sepsis"])
base_nafld = float(data["risk_nafld"])

st.divider()

# ── Layout ────────────────────────────────────────────────────
col_sliders, col_results = st.columns([1.2, 2], gap="large")

with col_sliders:
    st.markdown("### 🛠️ Modifiable Inputs")
    st.caption("Adjust lifestyle parameters to simulate future risk shifts.")
    
    with st.container():
        st.markdown("**Dietary Factors**")
        sodium = st.slider("Daily Sodium (g)", 0.5, 6.0, 3.0, 0.1, help="KDOQI: <2g/day")
        protein = st.slider("Daily Protein (g/kg)", 0.4, 1.4, 0.9, 0.05, help="KDOQI: 0.6-0.8g/kg")
        water = st.slider("Daily Hydration (L)", 0.5, 4.0, 2.0, 0.1)
        
        st.markdown("**Biometrics**")
        bmi = st.slider("Body Mass Index", 18, 45, 28, 1)
        bp = st.slider("Systolic BP (mmHg)", 90, 190, 135, 1)
        glucose = st.slider("Fasting Glucose (mg/dL)", 70, 300, 115, 5)
        
        st.markdown("**Lifestyle**")
        activity = st.select_slider(
            "Physical Activity Level",
            options=["Sedentary", "Light", "Moderate", "Active"],
            value="Light"
        )

# ── Simulation Logic (Counterfactual Engine) ──────────────────
activity_map = {"Sedentary": 1.12, "Light": 1.0, "Moderate": 0.85, "Active": 0.68}
af = activity_map[activity]

ckd_new = np.clip(
    (base_ckd + (sodium - 3.0)*0.035 + (protein - 0.9)*0.04 - (water - 2.0)*0.015 + (bp - 135)*0.002) * af,
    0.01, 0.99
)
nafld_new = np.clip(
    (base_nafld + (bmi - 28)*0.018 + (glucose - 115)*0.001) * af,
    0.01, 0.99
)
sepsis_new = np.clip(
    base_sepsis * (0.94 + (bp - 135) * 0.001),
    0.01, 0.99
)

# ── Results Panel ─────────────────────────────────────────────
with col_results:
    st.markdown("### 📊 Simulated Risk Trajectory")
    
    # Delta Comparison Metrics
    m1, m2, m3 = st.columns(3)
    
    def risk_delta(label, base, proj):
        delta = (proj - base) * 100
        # Inverse delta color because high risk is bad (Red for positive delta)
        st.metric(label, f"{int(proj*100)}%", f"{delta:+.1f}%", delta_color="inverse")

    with m1: risk_delta("CKD Projection", base_ckd, ckd_new)
    with m2: risk_delta("Sepsis Projection", base_sepsis, sepsis_new)
    with m3: risk_delta("NAFLD Projection", base_nafld, nafld_new)

    # Comparison Bar Chart
    fig = go.Figure()
    diseases = ["CKD", "Sepsis", "NAFLD"]
    
    fig.add_trace(go.Bar(
        name="Baseline", x=diseases, y=[base_ckd*100, base_sepsis*100, base_nafld*100],
        marker_color="#94A3B8", opacity=0.4
    ))
    fig.add_trace(go.Bar(
        name="Simulated", x=diseases, y=[float(ckd_new)*100, float(sepsis_new)*100, float(nafld_new)*100],
        marker_color="#1E3A8A"
    ))
    
    fig.update_layout(
        barmode='group', height=350,
        margin=dict(t=10, b=10, l=0, r=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#F8FAFC",
        yaxis=dict(title="Risk Probability (%)", range=[0, 100]),
        legend=dict(orientation="h", y=-0.2)
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Impact Prescription ──────────────────────────────────
    st.divider()
    st.markdown("### 📝 Optimization Strategy")
    
    impacts = {
        "Exercise Escalation": (base_ckd - (base_ckd * 0.85)),
        "Sodium Restriction (<2g)": max(0, (sodium - 2.0) * 0.035),
        "Weight Management (-5 BMI)": max(0, 5 * 0.018),
        "BP Normalization (120 mmHg)": max(0, (bp - 120) * 0.002)
    }
    
    best_action = max(impacts, key=impacts.get)
    best_val = impacts[best_action]

    if best_val > 0.01:
        st.success(f"**Primary Intervention:** {best_action} shows the highest projected efficacy for this patient, with a potential **{best_val*100:.1f}%** absolute risk reduction.")
    else:
        st.info("The current simulated profile is near the clinical optimum.")

st.sidebar.divider()
st.sidebar.caption("© 2026 AegisLife Simulation Engine")

