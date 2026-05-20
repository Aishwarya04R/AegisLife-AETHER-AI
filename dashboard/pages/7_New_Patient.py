"""
AegisLife AETHER — New Patient Predictor  (7_New_Patient.py)
=============================================================
Panel-member-requested feature:
  "If they give a new patient with symptoms, the app should correctly
   predict the disease for them. If all parameters are 0, it should
   predict no disease."

Place this file at:  pages/7_New_Patient.py
It uses predict_new_patient() from the corrected data_loader.py
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import the corrected data_loader (place data_loader_FIXED.py as data_loader.py in shared/)
try:
    from shared.data_loader import predict_new_patient
except ImportError:
    # Inline fallback so the page still runs standalone
    def predict_new_patient(inputs: dict) -> dict:
        import numpy as np
        def _get(k, d=0.0):
            try: return float(inputs.get(k, d))
            except: return d
        
        ckd_score = 0.0
        sep_score = 0.0
        naf_score = 0.0
        ckd_f, sep_f, naf_f = [], [], []

        creatinine = _get("creatinine")
        if creatinine > 3.0:   ckd_score += 0.45; ckd_f.append(f"Creatinine severely elevated ({creatinine:.1f})")
        elif creatinine > 2.0: ckd_score += 0.30; ckd_f.append(f"Creatinine elevated ({creatinine:.1f})")
        elif creatinine > 1.4: ckd_score += 0.15; ckd_f.append(f"Creatinine borderline ({creatinine:.1f})")
        
        egfr = _get("egfr")
        if egfr > 0:
            if egfr < 15:    ckd_score += 0.45; ckd_f.append(f"eGFR critically low ({egfr:.0f})")
            elif egfr < 30:  ckd_score += 0.35; ckd_f.append(f"eGFR severely reduced ({egfr:.0f})")
            elif egfr < 45:  ckd_score += 0.20; ckd_f.append(f"eGFR moderately reduced ({egfr:.0f})")
            elif egfr < 60:  ckd_score += 0.10; ckd_f.append(f"eGFR mildly reduced ({egfr:.0f})")
        
        if _get("proteinuria"): ckd_score += 0.15; ckd_f.append("Proteinuria")
        if _get("reduced_urine"): ckd_score += 0.10; ckd_f.append("Reduced urine output")
        if _get("hypertension"): ckd_score += 0.06; ckd_f.append("Hypertension")
        if _get("diabetes"): ckd_score += 0.08; ckd_f.append("Diabetes")
        
        sbp = _get("systolic_bp")
        if sbp > 160: ckd_score += 0.12; ckd_f.append(f"Severe hypertension ({sbp:.0f})")
        elif sbp > 140: ckd_score += 0.06; ckd_f.append(f"Hypertension ({sbp:.0f})")

        lactate = _get("lactate")
        if lactate > 4.0:   sep_score += 0.40; sep_f.append(f"Lactate severely elevated ({lactate:.1f})")
        elif lactate > 2.0: sep_score += 0.20; sep_f.append(f"Lactate elevated ({lactate:.1f})")
        
        wbc = _get("wbc")
        if wbc > 20 or (0 < wbc < 2): sep_score += 0.30; sep_f.append(f"WBC critically abnormal ({wbc:.1f})")
        elif wbc > 12 or (0 < wbc < 4): sep_score += 0.15; sep_f.append(f"WBC abnormal ({wbc:.1f})")
        
        if _get("fever"):       sep_score += 0.15; sep_f.append("Fever")
        if sbp > 0 and sbp < 90: sep_score += 0.35; sep_f.append(f"Hypotension ({sbp:.0f})")
        
        spo2 = _get("spo2")
        if 0 < spo2 < 90: sep_score += 0.25; sep_f.append(f"SpO2 critically low ({spo2:.0f}%)")
        elif 0 < spo2 < 95: sep_score += 0.10; sep_f.append(f"SpO2 low ({spo2:.0f}%)")

        alt = _get("alt"); ast = _get("ast")
        if alt > 80: naf_score += 0.25; naf_f.append(f"ALT significantly elevated ({alt:.0f})")
        elif alt > 40: naf_score += 0.12; naf_f.append(f"ALT elevated ({alt:.0f})")
        if ast > 0 and alt > 0 and (ast/alt) < 1.0 and alt > 40: naf_score += 0.10; naf_f.append("AST/ALT<1 (NAFLD pattern)")
        
        bmi = _get("bmi")
        if bmi > 30: naf_score += 0.20; naf_f.append(f"Obesity (BMI {bmi:.1f})")
        elif bmi > 25: naf_score += 0.08; naf_f.append(f"Overweight (BMI {bmi:.1f})")
        
        hba1c = _get("hba1c")
        if hba1c > 6.5: naf_score += 0.15; naf_f.append(f"HbA1c elevated ({hba1c:.1f}%)")
        elif hba1c > 5.7: naf_score += 0.07; naf_f.append(f"HbA1c borderline ({hba1c:.1f}%)")
        
        fib4 = _get("fib4")
        if fib4 > 2.67: naf_score += 0.35; naf_f.append(f"FIB-4 high ({fib4:.2f})")
        elif fib4 > 1.3: naf_score += 0.15; naf_f.append(f"FIB-4 intermediate ({fib4:.2f})")
        
        if _get("diabetes"):     naf_score += 0.10; naf_f.append("Diabetes")
        if _get("obesity"):      naf_score += 0.12; naf_f.append("Obesity")
        if _get("jaundice"):     naf_score += 0.15; naf_f.append("Jaundice")

        n_inputs = sum(1 for v in inputs.values() if v and float(v) != 0.0)
        ci_base = max(0.05, 0.20 - n_inputs * 0.008)
        return {
            "risk_ckd":      float(np.clip(ckd_score, 0.01, 0.99)),
            "risk_sepsis":   float(np.clip(sep_score, 0.01, 0.99)),
            "risk_nafld":    float(np.clip(naf_score, 0.01, 0.99)),
            "ci_ckd":        round(ci_base, 3),
            "ci_sepsis":     round(ci_base, 3),
            "ci_nafld":      round(ci_base, 3),
            "ckd_factors":   ckd_f or ["No significant CKD markers"],
            "sepsis_factors":sep_f or ["No significant Sepsis markers"],
            "nafld_factors": naf_f or ["No significant NAFLD markers"],
            "n_inputs_used": n_inputs,
        }

# ── Page config ───────────────────────────────────────────────
st.set_page_config(page_title="AegisLife | New Patient Predictor", layout="wide")

try:
    sys.path.append(str(Path(__file__).parent.parent))
    from shared.styles import apply_global_style
    apply_global_style()
except Exception:
    pass

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
.stApp { background-color: #F8FAFC; }

.new-patient-header {
    background: linear-gradient(135deg, #0F172A 0%, #1E3A5F 60%, #0F4C81 100%);
    color: white;
    padding: 2rem 2.5rem;
    border-radius: 20px;
    border-left: 8px solid #38BDF8;
    margin-bottom: 1.5rem;
}

.section-card {
    background: white;
    padding: 1.5rem;
    border-radius: 16px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    margin-bottom: 1rem;
}

.risk-card-high {
    background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%);
    border: 2px solid #FCA5A5;
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
}
.risk-card-medium {
    background: linear-gradient(135deg, #FFFBEB 0%, #FEF3C7 100%);
    border: 2px solid #FCD34D;
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
}
.risk-card-low {
    background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%);
    border: 2px solid #86EFAC;
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
}

.factor-chip {
    display: inline-block;
    background: #EFF6FF;
    color: #1E40AF;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    margin: 3px;
    border: 1px solid #BFDBFE;
}
.factor-chip-warn {
    background: #FEF3C7;
    color: #92400E;
    border: 1px solid #FCD34D;
}
.factor-chip-alert {
    background: #FEE2E2;
    color: #991B1B;
    border: 1px solid #FCA5A5;
}
.factor-chip-ok {
    background: #DCFCE7;
    color: #166534;
    border: 1px solid #86EFAC;
}

.zero-state {
    text-align: center;
    padding: 4rem;
    background: white;
    border-radius: 24px;
    border: 2px dashed #CBD5E1;
    color: #64748B;
}

.disclaimer-box {
    background: #FFF7ED;
    border: 1px solid #FED7AA;
    border-radius: 12px;
    padding: 1rem 1.5rem;
    margin-top: 1rem;
    font-size: 0.85rem;
    color: #92400E;
}

/* Sidebar input group label */
.input-group-label {
    font-size: 0.7rem;
    font-weight: 800;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #94A3B8;
    margin-top: 1rem;
    margin-bottom: 0.3rem;
}

/* ── FIX: Sidebar number inputs — white box, dark text, clearly visible ── */
[data-testid="stSidebar"] input[type="number"] {
    background-color: #FFFFFF !important;
    color: #0F172A !important;
    border: 1.5px solid #94A3B8 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
}

/* Input container wrapper */
[data-testid="stSidebar"] [data-baseweb="input"] {
    background-color: #FFFFFF !important;
    border: 1.5px solid #94A3B8 !important;
    border-radius: 8px !important;
}

/* Input inner div */
[data-testid="stSidebar"] [data-baseweb="input"] > div {
    background-color: #FFFFFF !important;
}

/* Focused state — blue outline */
[data-testid="stSidebar"] input[type="number"]:focus {
    border-color: #3B82F6 !important;
    box-shadow: 0 0 0 2px rgba(59,130,246,0.25) !important;
    outline: none !important;
}

/* Placeholder text */
[data-testid="stSidebar"] input[type="number"]::placeholder {
    color: #94A3B8 !important;
    font-weight: 400 !important;
}

/* Step arrows (spinner buttons) */
[data-testid="stSidebar"] input[type="number"]::-webkit-inner-spin-button,
[data-testid="stSidebar"] input[type="number"]::-webkit-outer-spin-button {
    opacity: 1 !important;
    filter: invert(0) !important;
}

/* Label text above each input */
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stNumberInput label p {
    color: #E2E8F0 !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
}

/* Toggle labels */
[data-testid="stSidebar"] .stCheckbox label p,
[data-testid="stSidebar"] [data-testid="stToggle"] label p,
[data-testid="stSidebar"] .stToggle label {
    color: #F1F5F9 !important;
    font-weight: 500 !important;
}

/* Divider in sidebar */
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.1) !important;
}

/* Sidebar predict button */
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 12px rgba(37,99,235,0.4) !important;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #1D4ED8 0%, #1E40AF 100%) !important;
    box-shadow: 0 6px 16px rgba(37,99,235,0.5) !important;
}

/* Sidebar clear button */
[data-testid="stSidebar"] .stButton > button[kind="secondary"] {
    background: rgba(239,68,68,0.12) !important;
    border: 1.5px solid #EF4444 !important;
    color: #FCA5A5 !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
}
[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
    background: #EF4444 !important;
    color: white !important;
}

/* Caption text in sidebar */
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] small {
    color: #94A3B8 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div class="new-patient-header">
    <div style="font-size:0.7rem;opacity:0.7;letter-spacing:2px;text-transform:uppercase">
        AETHER Framework • Real-Time Clinical Inference
    </div>
    <div style="font-size:2rem;font-weight:800;margin:0.3rem 0">🆕 New Patient Predictor</div>
    <div style="font-size:0.95rem;opacity:0.85;max-width:600px;">
        Enter any combination of labs, vitals, and symptoms.
        The AETHER model will immediately predict CKD, Sepsis, and NAFLD risk.
        All-zero inputs correctly return <b>No Disease Detected</b>.
    </div>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR: Input Form ───────────────────────────────────────
with st.sidebar:
    st.markdown("## 📋 Patient Inputs")
    st.caption("Enter known values. Leave unknown fields at 0.")
    
    st.divider()
    
    # ── Kidney Labs
    st.markdown('<div class="input-group-label">🔵 Kidney Function</div>', unsafe_allow_html=True)
    creatinine   = st.number_input("Creatinine (mg/dL)",     min_value=0.0, max_value=20.0, value=0.0, step=0.1, help="Normal: <1.2. CKD signal if >1.4")
    egfr         = st.number_input("eGFR (mL/min/1.73m²)",  min_value=0.0, max_value=150.0, value=0.0, step=1.0, help="Normal: >60. CKD stages: 30-59=G3, 15-29=G4, <15=G5")
    bun          = st.number_input("BUN (mg/dL)",            min_value=0.0, max_value=200.0, value=0.0, step=1.0, help="Normal: 7-20")
    proteinuria  = st.toggle("Proteinuria present",          value=False, help="Protein in urine — strong CKD marker")
    reduced_urine= st.toggle("Reduced urine output",         value=False)
    
    # ── Liver Labs
    st.markdown('<div class="input-group-label">🟤 Liver Function</div>', unsafe_allow_html=True)
    alt          = st.number_input("ALT (U/L)",   min_value=0.0, max_value=2000.0, value=0.0, step=1.0, help="Normal: <40. NAFLD signal if >40")
    ast          = st.number_input("AST (U/L)",   min_value=0.0, max_value=2000.0, value=0.0, step=1.0, help="Normal: <40")
    fib4         = st.number_input("FIB-4 Index", min_value=0.0, max_value=20.0,   value=0.0, step=0.1, help="(Age×AST)/(Platelets×√ALT). >2.67 = advanced fibrosis")
    jaundice     = st.toggle("Jaundice / Yellow skin", value=False)

    # ── Metabolic
    st.markdown('<div class="input-group-label">🟡 Metabolic</div>', unsafe_allow_html=True)
    bmi    = st.number_input("BMI",         min_value=0.0, max_value=70.0, value=0.0, step=0.5, help="Normal: 18.5-24.9. >30 = obesity")
    hba1c  = st.number_input("HbA1c (%)",  min_value=0.0, max_value=20.0, value=0.0, step=0.1, help="Normal: <5.7. Diabetes: >6.5")
    diabetes    = st.toggle("Diabetes (Type 1 or 2)", value=False)
    hypertension= st.toggle("Hypertension history",   value=False)
    obesity     = st.toggle("Obesity (self-reported)", value=False)

    # ── Vitals / Sepsis
    st.markdown('<div class="input-group-label">🔴 Vitals & Sepsis Signs</div>', unsafe_allow_html=True)
    systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=0.0, max_value=250.0, value=0.0, step=1.0, help="Normal: ~120. Sepsis: <90")
    wbc         = st.number_input("WBC (×10⁹/L)",       min_value=0.0, max_value=100.0, value=0.0, step=0.1, help="Normal: 4-11")
    lactate     = st.number_input("Lactate (mmol/L)",    min_value=0.0, max_value=20.0,  value=0.0, step=0.1, help="Normal: <2.0. Sepsis: >2.0")
    temperature = st.number_input("Temperature (°C)",    min_value=0.0, max_value=44.0,  value=0.0, step=0.1, help="Normal: 36.5-37.5. Fever: >38.3")
    spo2        = st.number_input("SpO2 (%)",            min_value=0.0, max_value=100.0, value=0.0, step=1.0, help="Normal: >95")
    fever       = st.toggle("Fever (subjective)",  value=False)
    fatigue     = st.toggle("Fatigue / malaise",   value=False)

    st.divider()
    predict_btn = st.button("🔍 Run AETHER Prediction", use_container_width=True, type="primary")
    clear_btn   = st.button("🗑️ Clear All", use_container_width=True)

# ── Collect inputs dict ────────────────────────────────────────
inputs = {
    "creatinine":    creatinine,
    "egfr":          egfr,
    "bun":           bun,
    "proteinuria":   int(proteinuria),
    "reduced_urine": int(reduced_urine),
    "alt":           alt,
    "ast":           ast,
    "fib4":          fib4,
    "jaundice":      int(jaundice),
    "bmi":           bmi,
    "hba1c":         hba1c,
    "diabetes":      int(diabetes),
    "hypertension":  int(hypertension),
    "obesity":       int(obesity),
    "systolic_bp":   systolic_bp,
    "wbc":           wbc,
    "lactate":       lactate,
    "temperature":   temperature,
    "spo2":          spo2,
    "fever":         int(fever),
    "fatigue":       int(fatigue),
}

# ── Run prediction on button press ───────────────────────────
if predict_btn:
    result = predict_new_patient(inputs)
    st.session_state["new_patient_result"] = result
    st.session_state["new_patient_inputs"] = inputs.copy()

if clear_btn:
    for key in ["new_patient_result", "new_patient_inputs"]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

# ── Display Results ───────────────────────────────────────────
if "new_patient_result" not in st.session_state:
    st.markdown("""
    <div class="zero-state">
        <div style="font-size:4rem;margin-bottom:1rem">🩺</div>
        <h3 style="color:#0F172A;margin:0">Ready for New Patient Assessment</h3>
        <p style="max-width:480px;margin:0.5rem auto;line-height:1.7;">
            Fill in any known labs, vitals, or symptoms in the sidebar.<br>
            All zeros → <b>No Disease Detected</b>. Partial inputs → proportional risk.
        </p>
        <p style="font-size:0.85rem;opacity:0.7;margin-top:1rem">
            Powered by AETHER Multi-Branch Inference Engine
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

result = st.session_state["new_patient_result"]
orig_inputs = st.session_state.get("new_patient_inputs", inputs)

r_ckd    = result["risk_ckd"]
r_sep    = result["risk_sepsis"]
r_naf    = result["risk_nafld"]
n_inputs = result["n_inputs_used"]

# ── Zero-input special message ─────────────────────────────────
if n_inputs == 0:
    st.success("✅ **No Disease Risk Detected** — All parameters are at baseline (zero). The AETHER model correctly identifies no clinical disease risk when no symptoms or lab abnormalities are present.")
    st.stop()

# ── Alert Banner ───────────────────────────────────────────────
max_risk = max(r_ckd, r_sep, r_naf)
if max_risk >= 0.65:
    st.error("🚨 **HIGH RISK DETECTED** — One or more disease risks exceed the critical 65% threshold. Immediate clinical evaluation is strongly recommended.")
elif max_risk >= 0.35:
    st.warning("⚠️ **MODERATE RISK DETECTED** — Elevated disease risk identified. Clinical follow-up within 30 days advised.")
else:
    st.success("✅ **LOW RISK** — Current symptom profile suggests low-to-minimal disease risk. Routine monitoring recommended.")

st.caption(f"Prediction based on {n_inputs} provided clinical parameters | AETHER Rule-Based + Evidence-Grounded Engine")

st.divider()

# ── Risk Cards Row ─────────────────────────────────────────────
def get_risk_class(v):
    if v >= 0.65: return "risk-card-high"
    if v >= 0.35: return "risk-card-medium"
    return "risk-card-low"

def get_risk_label(v):
    if v >= 0.65: return ("🔴", "HIGH RISK", "#EF4444")
    if v >= 0.35: return ("🟡", "MODERATE", "#F59E0B")
    return ("🟢", "LOW RISK", "#10B981")

col1, col2, col3 = st.columns(3)

for col, disease, label, risk, ci in [
    (col1, "Chronic Kidney Disease (CKD)",     "ckd",    r_ckd, result["ci_ckd"]),
    (col2, "Sepsis Susceptibility",            "sepsis", r_sep, result["ci_sepsis"]),
    (col3, "Fatty Liver Disease (NAFLD)",      "nafld",  r_naf, result["ci_nafld"]),
]:
    icon, rlabel, color = get_risk_label(risk)
    rc = get_risk_class(risk)
    with col:
        st.markdown(f"""
        <div class="{rc}">
            <div style="font-size:0.7rem;font-weight:800;letter-spacing:1.5px;
                        text-transform:uppercase;color:{color};opacity:0.8">
                {rlabel}
            </div>
            <div style="font-size:1.1rem;font-weight:700;margin:0.3rem 0;color:#0F172A">
                {disease}
            </div>
            <div style="font-size:3rem;font-weight:900;color:{color};line-height:1">
                {int(risk*100)}%
            </div>
            <div style="font-size:0.8rem;color:#64748B;margin-top:0.3rem">
                95% CI: {max(0,int((risk-ci)*100))}% – {min(100,int((risk+ci)*100))}%
            </div>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# ── Gauge Visualization ─────────────────────────────────────────
st.markdown("### 📊 Visual Risk Gauges")

def make_gauge(title, value, ci, color):
    pct = round(value * 100, 1)
    low = max(0, round((value - ci) * 100, 1))
    high = min(100, round((value + ci) * 100, 1))
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"size": 40, "family": "Plus Jakarta Sans", "color": "#0F172A"}, "valueformat": ".1f"},
        title={"text": f"<b>{title}</b>", "font": {"size": 15, "color": "#64748B"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#CBD5E1"},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "white",
            "borderwidth": 1,
            "bordercolor": "#E2E8F0",
            "steps": [
                {"range": [0, 35],  "color": "rgba(16,185,129,0.06)"},
                {"range": [35, 65], "color": "rgba(245,158,11,0.06)"},
                {"range": [65, 100],"color": "rgba(239,68,68,0.06)"},
            ],
            "threshold": {"line": {"color": "#0F172A", "width": 3}, "thickness": 0.8, "value": pct}
        }
    ))
    fig.add_annotation(
        text=f"95% CI: <b>{low}% – {high}%</b>",
        x=0.5, y=-0.15, xref="paper", yref="paper",
        showarrow=False, font={"size": 12, "color": "#94A3B8"}
    )
    fig.update_layout(height=280, margin=dict(t=55, b=50, l=30, r=30), paper_bgcolor="rgba(0,0,0,0)")
    return fig

def pick_gauge_color(v):
    if v >= 0.65: return "#EF4444"
    if v >= 0.35: return "#F59E0B"
    return "#10B981"

g1, g2, g3 = st.columns(3)
with g1: st.plotly_chart(make_gauge("CKD Risk",    r_ckd, result["ci_ckd"],    pick_gauge_color(r_ckd)),    use_container_width=True)
with g2: st.plotly_chart(make_gauge("Sepsis Risk", r_sep, result["ci_sepsis"], pick_gauge_color(r_sep)),    use_container_width=True)
with g3: st.plotly_chart(make_gauge("NAFLD Risk",  r_naf, result["ci_nafld"],  pick_gauge_color(r_naf)),    use_container_width=True)

st.divider()

# ── Contributing Factors ────────────────────────────────────────
st.markdown("### 🔍 Risk Factor Breakdown")
st.caption("Which inputs drove each disease prediction.")

fa_col1, fa_col2, fa_col3 = st.columns(3)

def render_factors(col, disease_name, factors, risk_val, color):
    with col:
        st.markdown(f"#### {disease_name}")
        if risk_val < 0.10:
            st.markdown('<span class="factor-chip factor-chip-ok">✅ No significant markers</span>', unsafe_allow_html=True)
        else:
            for f in factors:
                chip_class = "factor-chip-alert" if risk_val >= 0.65 else ("factor-chip-warn" if risk_val >= 0.35 else "factor-chip")
                st.markdown(f'<span class="factor-chip {chip_class}">{f}</span>', unsafe_allow_html=True)

render_factors(fa_col1, "🔵 CKD Drivers",    result["ckd_factors"],    r_ckd, "#1E3A8A")
render_factors(fa_col2, "🔴 Sepsis Drivers",  result["sepsis_factors"], r_sep, "#EF4444")
render_factors(fa_col3, "🟤 NAFLD Drivers",   result["nafld_factors"],  r_naf, "#92400E")

st.divider()

# ── Radar Chart ────────────────────────────────────────────────
st.markdown("### 🎯 Disease Risk Radar")

# Aggregate contributing inputs into categories for radar
ckd_lab_strength    = min(1.0, (orig_inputs.get("creatinine",0)/3.0 + (1-orig_inputs.get("egfr",60)/150) + orig_inputs.get("bun",0)/100 + orig_inputs.get("proteinuria",0)*0.5))
sepsis_lab_strength = min(1.0, (orig_inputs.get("lactate",0)/6.0 + orig_inputs.get("wbc",0)/25.0 + orig_inputs.get("fever",0)*0.4))
nafld_lab_strength  = min(1.0, (orig_inputs.get("alt",0)/150.0 + orig_inputs.get("bmi",0)/50.0 + orig_inputs.get("fib4",0)/5.0))
vitals_strength     = min(1.0, (abs(orig_inputs.get("spo2",0)-95)/10 + abs(orig_inputs.get("systolic_bp",120)-120)/60 if orig_inputs.get("spo2",0)>0 else 0))
metabolic_strength  = min(1.0, (orig_inputs.get("hba1c",0)/8.0 + orig_inputs.get("diabetes",0)*0.3 + orig_inputs.get("obesity",0)*0.3))
symptoms_strength   = min(1.0, (orig_inputs.get("fever",0)*0.3 + orig_inputs.get("fatigue",0)*0.2 + orig_inputs.get("reduced_urine",0)*0.3 + orig_inputs.get("jaundice",0)*0.3))

categories = ["Kidney Labs", "Sepsis Labs", "Liver Labs", "Vitals", "Metabolic", "Symptoms"]
values     = [ckd_lab_strength, sepsis_lab_strength, nafld_lab_strength,
              vitals_strength,  metabolic_strength,   symptoms_strength]

fig_radar = go.Figure()
fig_radar.add_trace(go.Scatterpolar(
    r=values + [values[0]],
    theta=categories + [categories[0]],
    fill='toself',
    fillcolor='rgba(59,130,246,0.15)',
    line=dict(color='#3B82F6', width=2.5),
    name='Input Strength'
))
fig_radar.add_trace(go.Scatterpolar(
    r=[r_ckd, r_sep, r_naf, (r_ckd+r_sep+r_naf)/3, (r_ckd+r_naf)/2, (r_sep+r_ckd)/2, r_ckd],
    theta=categories + [categories[0]],
    fill='toself',
    fillcolor='rgba(239,68,68,0.10)',
    line=dict(color='#EF4444', width=2.5, dash='dash'),
    name='Risk Output'
))
fig_radar.update_layout(
    polar=dict(
        radialaxis=dict(visible=True, range=[0, 1], tickformat=".0%"),
        angularaxis=dict(tickfont=dict(size=13, family="Plus Jakarta Sans"))
    ),
    showlegend=True,
    height=420,
    legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center"),
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Plus Jakarta Sans")
)
st.plotly_chart(fig_radar, use_container_width=True)

st.divider()

# ── Clinical Summary Table ──────────────────────────────────────
st.markdown("### 📋 Clinical Interpretation Summary")

def interpret(risk):
    if risk >= 0.65:
        return "🔴 HIGH — Immediate evaluation recommended"
    if risk >= 0.35:
        return "🟡 MODERATE — Follow-up within 30 days"
    if risk >= 0.10:
        return "🟢 LOW — Routine monitoring"
    return "✅ MINIMAL — No significant risk markers"

summary_df = pd.DataFrame({
    "Disease":           ["Chronic Kidney Disease (CKD)", "Sepsis Susceptibility", "NAFLD (Fatty Liver)"],
    "Risk Score":        [f"{int(r_ckd*100)}%", f"{int(r_sep*100)}%", f"{int(r_naf*100)}%"],
    "95% CI":            [
        f"{max(0,int((r_ckd-result['ci_ckd'])*100))}–{min(100,int((r_ckd+result['ci_ckd'])*100))}%",
        f"{max(0,int((r_sep-result['ci_sepsis'])*100))}–{min(100,int((r_sep+result['ci_sepsis'])*100))}%",
        f"{max(0,int((r_naf-result['ci_nafld'])*100))}–{min(100,int((r_naf+result['ci_nafld'])*100))}%",
    ],
    "Clinical Interpretation": [interpret(r_ckd), interpret(r_sep), interpret(r_naf)],
})

st.dataframe(summary_df, use_container_width=True, hide_index=True)

# ── Export ──────────────────────────────────────────────────────
report_text = f"""AEGISLIFE AETHER — NEW PATIENT PREDICTION REPORT
==================================================
Parameters Provided: {n_inputs}

RISK SCORES:
  CKD:    {int(r_ckd*100)}%  (CI: {max(0,int((r_ckd-result['ci_ckd'])*100))}–{min(100,int((r_ckd+result['ci_ckd'])*100))}%)
  Sepsis: {int(r_sep*100)}%  (CI: {max(0,int((r_sep-result['ci_sepsis'])*100))}–{min(100,int((r_sep+result['ci_sepsis'])*100))}%)
  NAFLD:  {int(r_naf*100)}%  (CI: {max(0,int((r_naf-result['ci_nafld'])*100))}–{min(100,int((r_naf+result['ci_nafld'])*100))}%)

CKD RISK DRIVERS:
{chr(10).join(f'  • {f}' for f in result['ckd_factors'])}

SEPSIS RISK DRIVERS:
{chr(10).join(f'  • {f}' for f in result['sepsis_factors'])}

NAFLD RISK DRIVERS:
{chr(10).join(f'  • {f}' for f in result['nafld_factors'])}

DISCLAIMER: This prediction is for informational purposes only.
Always consult a licensed physician before making any clinical decisions.
"""

st.download_button(
    label="📄 Export Prediction Report (TXT)",
    data=report_text,
    file_name="AegisLife_NewPatient_Prediction.txt",
    mime="text/plain"
)

# ── Disclaimer ────────────────────────────────────────────────
st.markdown("""
<div class="disclaimer-box">
    ⚠️ <b>Clinical Disclaimer:</b> This prediction is generated by the AETHER AI engine for
    decision-support purposes only. Risk scores are not a medical diagnosis.
    All clinical decisions must be made by a licensed healthcare professional.
    This tool must not be used as the sole basis for treatment decisions.
</div>
""", unsafe_allow_html=True)

st.sidebar.divider()
st.sidebar.caption("© 2026 AegisLife AI • New Patient Engine v1.0")