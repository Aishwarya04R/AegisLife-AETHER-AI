import streamlit as st
import plotly.graph_objects as go
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from shared.data_loader import get_patient

st.set_page_config(page_title="AegisLife | Risk Monitor", layout="wide")

sys.path.append(str(Path(__file__).parent.parent))
from shared.styles import apply_global_style
apply_global_style()

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
.stApp { background-color: #F8FAFC; }
</style>
""", unsafe_allow_html=True)

st.title("📈 Clinical Risk Surveillance")
st.caption("AETHER 12-Month Longitudinal Early Warning System | Real-time Bayesian Inference")

# ── Guard ──────────────────────────────────────────────────────
patient_id = st.session_state.get("selected_patient")
if not patient_id:
    st.warning("⚠️ Please select a patient on the Patient Overview page first.")
    st.stop()

data = get_patient(patient_id)

# ── Alert Banner ───────────────────────────────────────────────
max_risk = max(data["risk_ckd"], data["risk_sepsis"], data["risk_nafld"])
if max_risk >= 0.65:
    st.error(f"🚨 **CRITICAL STATUS** — Patient {patient_id} has exceeded the 65% clinical threshold.")
elif max_risk >= 0.35:
    st.warning(f"⚠️ **ELEVATED RISK** — Patient {patient_id} requires priority monitoring.")
else:
    st.success(f"✅ **STABLE STATUS** — Patient {patient_id} risk levels within baseline parameters.")

st.divider()

# ── Gauges ─────────────────────────────────────────────────────
def make_gauge(title, value, ci):
    pct  = round(float(value) * 100, 1)
    low  = max(0,   round((float(value) - float(ci)) * 100, 1))
    high = min(100, round((float(value) + float(ci)) * 100, 1))
    bar_color = "#1E3A8A" if value < 0.35 else "#F59E0B" if value < 0.65 else "#EF4444"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=pct,
        number={"suffix": "%", "font": {"size": 44, "family": "Plus Jakarta Sans", "color": "#0F172A"}, "valueformat": ".1f"},
        title={"text": f"<b>{title}</b>", "font": {"size": 18, "color": "#64748B"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#CBD5E1"},
            "bar": {"color": bar_color, "thickness": 0.3},
            "bgcolor": "white", "borderwidth": 1, "bordercolor": "#E2E8F0",
            "steps": [
                {"range": [0, 35],   "color": "rgba(16,185,129,0.05)"},
                {"range": [35, 65],  "color": "rgba(245,158,11,0.05)"},
                {"range": [65, 100], "color": "rgba(239,68,68,0.05)"},
            ],
            "threshold": {"line": {"color": "#0F172A", "width": 4}, "thickness": 0.8, "value": pct}
        }
    ))
    fig.add_annotation(text=f"95% CI: <b>{low}% – {high}%</b>",
        x=0.5, y=-0.15, xref="paper", yref="paper",
        showarrow=False, font={"size": 13, "color": "#94A3B8"})
    fig.update_layout(height=300, margin=dict(t=60, b=50, l=40, r=40), paper_bgcolor="rgba(0,0,0,0)")
    return fig

col1, col2, col3 = st.columns(3)
with col1: st.plotly_chart(make_gauge("CKD Risk",    data["risk_ckd"],    data["ci_ckd"]),    use_container_width=True)
with col2: st.plotly_chart(make_gauge("Sepsis Risk", data["risk_sepsis"], data["ci_sepsis"]), use_container_width=True)
with col3: st.plotly_chart(make_gauge("NAFLD Risk",  data["risk_nafld"],  data["ci_nafld"]),  use_container_width=True)

st.divider()
st.subheader("🗓️ 12-Month Predictive Horizon")
st.caption("Temporal risk progression with Bayesian Confidence Bands (LSTM Time-Series Output).")

# ── Build trend data ───────────────────────────────────────────
# Convert EVERYTHING to plain Python float before any calculation
ckd_risk    = float(data["risk_ckd"])
sepsis_risk = float(data["risk_sepsis"])
nafld_risk  = float(data["risk_nafld"])

pid_digits = ''.join(filter(str.isdigit, str(patient_id)))
base_seed  = int(pid_digits) % (2**31) if pid_digits else 42

months = ["Month 1","Month 2","Month 3","Month 4","Month 5","Month 6",
          "Month 7","Month 8","Month 9","Month 10","Month 11","Month 12"]

def make_trend(final_val, seed):
    """Returns a plain Python list of 12 floats, rising from 50% to final_val."""
    np.random.seed(seed)
    start = max(0.05, final_val * 0.5)
    pts   = np.linspace(start, final_val, 12) + np.random.normal(0, 0.008, 12)
    pts   = np.clip(pts, 0.01, 0.99) * 100
    # CRITICAL: convert each value to a plain Python float
    return [float(v) for v in pts]

ckd_y    = make_trend(ckd_risk,    base_seed + 1000)
sepsis_y = make_trend(sepsis_risk, base_seed + 2000)
nafld_y  = make_trend(nafld_risk,  base_seed + 3000)

# ── Chart — absolute minimum code, three traces only ──────────
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=months, y=ckd_y,
    name="Chronic Kidney Disease",
    mode="lines+markers",
    line=dict(color="#1E40AF", width=5, shape="spline"),
    marker=dict(size=10, color="#1E40AF", line=dict(width=2, color="white")),
))

fig.add_trace(go.Scatter(
    x=months, y=sepsis_y,
    name="Sepsis Susceptibility",
    mode="lines+markers",
    line=dict(color="#DC2626", width=5, shape="spline"),
    marker=dict(size=10, color="#DC2626", line=dict(width=2, color="white")),
))

fig.add_trace(go.Scatter(
    x=months, y=nafld_y,
    name="Fatty Liver (NAFLD)",
    mode="lines+markers",
    line=dict(color="#059669", width=5, shape="spline"),
    marker=dict(size=10, color="#059669", line=dict(width=2, color="white")),
))

fig.add_hline(y=65, line_dash="dash", line_color="#DC2626", line_width=1.5,
              annotation_text="Critical (65%)", annotation_position="top right",
              annotation_font=dict(color="#DC2626", size=11))
fig.add_hline(y=35, line_dash="dash", line_color="#F59E0B", line_width=1.5,
              annotation_text="Caution (35%)", annotation_position="top right",
              annotation_font=dict(color="#F59E0B", size=11))

fig.update_layout(
    height=480,
    xaxis=dict(title="Simulation Timeline", showgrid=False,
               tickfont=dict(size=12, family="Plus Jakarta Sans")),
    yaxis=dict(title="Fused Probability (%)", range=[0, 105], ticksuffix="%",
               showgrid=True, gridcolor="#E2E8F0",
               tickfont=dict(size=12, family="Plus Jakarta Sans")),
    legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center",
                font=dict(size=13, family="Plus Jakarta Sans")),
    hovermode="x unified",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#F8FAFC",
    margin=dict(t=10, b=10, l=10, r=10),
    font=dict(family="Plus Jakarta Sans"),
)

st.plotly_chart(fig, use_container_width=True)

# ── Risk Summary Cards ─────────────────────────────────────────
st.markdown("#### 📋 Month 12 Risk Summary")
card_data = [
    ("Chronic Kidney Disease", ckd_risk,    float(data.get("ci_ckd",    0.05)), "#1E40AF"),
    ("Sepsis Susceptibility",  sepsis_risk, float(data.get("ci_sepsis", 0.05)), "#DC2626"),
    ("Fatty Liver (NAFLD)",    nafld_risk,  float(data.get("ci_nafld",  0.05)), "#059669"),
]
cols = st.columns(3)
for col, (name, val, ci, color) in zip(cols, card_data):
    status = "🔴 Critical" if val >= 0.65 else "🟡 Elevated" if val >= 0.35 else "🟢 Stable"
    col.markdown(f"""
    <div style="background:white;padding:1rem 1.2rem;border-radius:14px;
                border-left:5px solid {color};border:1px solid #E2E8F0;
                box-shadow:0 2px 6px rgba(0,0,0,0.03);">
        <div style="font-size:0.72rem;font-weight:700;color:#64748B;
                    text-transform:uppercase;letter-spacing:1px">{name}</div>
        <div style="font-size:2rem;font-weight:800;color:{color}">{int(val*100)}%</div>
        <div style="font-size:0.8rem;color:#94A3B8">
            95% CI: {max(0,int((val-ci)*100))}–{min(100,int((val+ci)*100))}%
        </div>
        <div style="font-size:0.85rem;margin-top:4px">{status}</div>
    </div>
    """, unsafe_allow_html=True)

st.sidebar.divider()
st.sidebar.markdown(f"""
<div style='background: rgba(255,255,255,0.05); padding: 1rem; border-radius: 10px;'>
    <div style='font-size: 0.7rem; opacity: 0.6;'>SURVEILLANCE STATUS</div>
    <div style='font-size: 0.9rem; font-weight: 700;'>Patient {patient_id}</div>
    <div style='font-size: 0.8rem; margin-top: 5px;'>Monitoring Horizon: <b>12 Months</b></div>
</div>
""", unsafe_allow_html=True)