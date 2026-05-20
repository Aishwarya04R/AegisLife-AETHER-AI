import streamlit as st
import sys
from pathlib import Path

# 1. Page Configuration (MUST BE FIRST)
st.set_page_config(page_title="AegisLife | AI Assistant", layout="wide")

# 2. Styles and Path Setup
sys.path.append(str(Path(__file__).parent.parent.parent))
from shared.data_loader import get_patient
from branch4_llm.prevention_engine import chat_response

# Path adjustment for shared styles
sys.path.append(str(Path(__file__).parent.parent)) 
from shared.styles import apply_global_style
apply_global_style()

# 3. Professional Clinical CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
.stApp { background-color: #F8FAFC; }

/* --- SIDEBAR BUTTON FIX (Changes the white bar to a visible Red Button) --- */
[data-testid="stSidebar"] .stButton > button {
    background-color: rgba(239, 68, 68, 0.15) !important;
    border: 1px solid #EF4444 !important;
    color: #EF4444 !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}

[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #EF4444 !important;
    color: white !important;
}

/* Consultation Chat Bubbles */
.stChatMessage {
    background-color: white !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 15px !important;
    box-shadow: 0 4px 6px rgba(0,0,0,0.02) !important;
    padding: 1.2rem !important;
    margin-bottom: 1rem !important;
}

/* Suggestion Chips (Main Area) */
.stButton > button {
    border-radius: 20px;
    border: 1px solid #E2E8F0;
    background-color: white;
    color: #475569;
    font-size: 0.85rem;
    transition: all 0.3s;
}
.stButton > button:hover {
    border-color: #3B82F6;
    color: #3B82F6;
    background-color: #EFF6FF;
}

/* Sidebar Customization */
[data-testid="stSidebar"] { background-color: #0A1628; }
</style>
""", unsafe_allow_html=True)

st.title("🤖 Clinical Intelligence Terminal")
st.caption("AETHER Knowledge-Base | RAG-Enabled Clinical Decision Support")

# ── Guard ─────────────────────────────────────────────────────
patient_id = st.session_state.get("selected_patient")
if not patient_id:
    st.warning("⚠️ Please select a patient on the Patient Overview page first to provide clinical context.")
    st.stop()

data = get_patient(patient_id)

# ── Sidebar: Clinical Context ─────────────────────────────────
with st.sidebar:
    st.markdown("### 📋 Neural Context")
    st.info(f"Active Analysis: **Patient {patient_id}**")
    
    st.divider()
    st.subheader("Fused Risk Matrix")
    st.metric("CKD", f"{int(data['risk_ckd']*100)}%")
    st.metric("Sepsis", f"{int(data['risk_sepsis']*100)}%")
    st.metric("NAFLD", f"{int(data['risk_nafld']*100)}%")
    
    st.divider()
    # This button will now be visible thanks to the CSS fix above
    if st.sidebar.button("🗑️ Reset Consultation", use_container_width=True):
        st.session_state["chat_history"] = []
        st.rerun()

# ── Chat Logic & Suggestions ──────────────────────────────────
st.session_state.setdefault("chat_history", [])

st.markdown("### 💡 Quick Queries")
suggestions = [
    "What are the primary risk drivers?",
    "Generate a renal-safe diet summary.",
    "Explain sepsis susceptibility logic.",
    "Recommended activity intensity."
]

s_cols = st.columns(4)
for i, s_text in enumerate(suggestions):
    if s_cols[i].button(s_text, key=f"sug_{i}", use_container_width=True):
        st.session_state["chat_history"].append(("user", s_text))
        with st.spinner("🧠 Consulting AETHER Knowledge Base..."):
            answer = chat_response(s_text, data, st.session_state["chat_history"])
        st.session_state["chat_history"].append(("assistant", answer))
        st.rerun()

st.divider()

# ── Display Conversation ──────────────────────────────────────
for role, msg in st.session_state["chat_history"]:
    avatar = "👤" if role == "user" else "🏥"
    with st.chat_message(role, avatar=avatar):
        st.markdown(msg)

# ── Input Area ────────────────────────────────────────────────
if prompt := st.chat_input("Inquire about clinical guidelines, diet, or risk factors..."):
    st.session_state["chat_history"].append(("user", prompt))
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🏥"):
        status_placeholder = st.empty()
        with status_placeholder.status("🔍 Accessing Verified Guidelines...", expanded=False) as status:
            st.write("Extracting patient biometric context...")
            st.write("Performing vector search on KDOQI & AASLD repositories...")
            st.write("Synthesizing multi-modal reasoning...")
            
            answer = chat_response(prompt, data, st.session_state["chat_history"])
            status.update(label="✅ Response Synthesized", state="complete", expanded=False)
        
        st.markdown(answer)
    
    st.session_state["chat_history"].append(("assistant", answer))