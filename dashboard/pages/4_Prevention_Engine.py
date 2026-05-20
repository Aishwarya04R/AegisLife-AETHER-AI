import streamlit as st
import sys
from pathlib import Path

# 1. Page Configuration (MUST BE FIRST)
st.set_page_config(page_title="AegisLife | Prevention Engine", layout="wide")

# 2. Path & Style Setup
sys.path.append(str(Path(__file__).parent.parent.parent))
from shared.styles import apply_global_style
from shared.data_loader import get_patient
from branch4_llm.prevention_engine import generate_prevention_plan

# Apply the professional unified theme
apply_global_style()

# 3. Enhanced Clinical CSS
st.markdown("""
<style>
/* This fixes the "Empty Box" issue by ensuring the paper only appears when there is text */
.report-paper {
    background-color: white;
    padding: 40px;
    border-radius: 20px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 10px 30px rgba(0,0,0,0.05);
    color: #1E293B;
    line-height: 1.8;
    font-size: 16px;
    margin-top: 20px;
}
.source-tag {
    background: #F0FDF4;
    color: #166534;
    padding: 4px 12px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 700;
    margin-right: 8px;
    border: 1px solid #BBF7D0;
}
</style>
""", unsafe_allow_html=True)

st.title("🛡️ Prevention Engine")
st.caption("AETHER RAG Architecture | Knowledge-Grounded Clinical Intervention")

# ── Guard ─────────────────────────────────────────────────────
patient_id = st.session_state.get("selected_patient")
if not patient_id:
    st.warning("⚠️ Please select a patient on the Patient Overview page first.")
    st.stop()

data = get_patient(patient_id)

# ── Sidebar: Clinical Metadata ────────────────────────────────
with st.sidebar:
    st.markdown("### 📋 Generation Context")
    st.info(f"Targeting Patient: **{patient_id}**")
    st.divider()
    st.markdown("#### Real-time Risk Metrics")
    st.metric("CKD", f"{int(data['risk_ckd']*100)}%")
    st.metric("Sepsis", f"{int(data['risk_sepsis']*100)}%")
    st.metric("NAFLD", f"{int(data['risk_nafld']*100)}%")

# ── Header Dashboard ──────────────────────────────────────────
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("<small style='color:#64748B; font-weight:700;'>DATABASE STATUS</small>", unsafe_allow_html=True)
    st.markdown("#### ChromaDB Vector Store")
with c2:
    st.markdown("<small style='color:#64748B; font-weight:700;'>GUIDELINES LOADED</small>", unsafe_allow_html=True)
    st.markdown("#### KDOQI, KDIGO, AASLD")
with c3:
    st.markdown("<small style='color:#64748B; font-weight:700;'>AI ENGINE</small>", unsafe_allow_html=True)
    st.markdown("#### Gemini 1.5 Flash")

st.divider()

# ── Execution ─────────────────────────────────────────────────
if st.button("🚀 Generate Personalized Prevention Protocol", use_container_width=True, type="primary"):
    with st.spinner("🧠 Consulting Medical Knowledge Base..."):
        # Calling your Branch 4 logic
        result = generate_prevention_plan(patient_id, data)
        st.session_state["last_plan"] = result

# ── Display Logic (FIXED EMPTY BOX) ───────────────────────────
if "last_plan" in st.session_state:
    result = st.session_state["last_plan"]
    
    # 1. Show RAG Sources
    st.markdown("##### 📚 Verified Knowledge Sources Used:")
    source_html = "".join([f'<span class="source-tag">{s}</span>' for s in result['sources']])
    st.markdown(source_html, unsafe_allow_html=True)

    # 2. Main Report (Fixed: Text is now properly nested inside the CSS class)
    st.markdown(f"""
    <div class="report-paper">
        <h2 style="margin-top:0;">Clinical Protocol for {patient_id}</h2>
        <hr style="border-color:#F1F5F9;">
        {result["plan_text"]}
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    
    # 3. Export Action
    st.download_button(
        label="📄 Export Clinical Protocol (TXT)",
        data=result["plan_text"],
        file_name=f"AegisLife_Protocol_{patient_id}.txt",
        mime="text/plain"
    )

else:
    # Professional Empty State
    st.markdown("""
    <div style="text-align: center; padding: 5rem; background: white; border-radius: 24px; border: 2px dashed #E2E8F0;">
        <div style="font-size: 4rem; margin-bottom: 1rem;">🛡️</div>
        <h3 style="color: #0F172A;">Protocol Framework Ready</h3>
        <p style="color: #64748B; max-width: 500px; margin: 0 auto; line-height: 1.6;">
            Click the button above to generate a 7-day personalized meal plan, 
            exercise prescription, and monitoring schedule grounded in 
            <b>KDOQI 2020</b> and <b>AASLD 2023</b> guidelines.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.sidebar.divider()
st.sidebar.caption("© 2026 AegisLife AI • Clinical RAG v1.0")