"""
AegisLife AETHER — data_loader.py  (FINAL v4 — uses aether_final_dashboard_data)
=================================================================================

KEY CHANGE:
    The final fused risk scores are now loaded directly from:
        aether_final_dashboard_data__1_.csv  (or aether_final_dashboard_data.csv)

    Columns in that file:
        patient_id, label, final_risk_ckd, final_risk_sep, final_risk_naf

    These pre-computed scores replace the in-code weighted fusion that was
    giving incorrectly high CKD and low NAFLD values for all patients.

    The branch CSVs (branch1–4) are still loaded to populate the per-branch
    breakdown bars shown on the Patient Profile page, but they no longer
    drive the headline risk numbers.

All other helpers (get_patient, load_shap_values, load_temporal_values,
load_bert_attention, predict_new_patient) are unchanged.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

ROOT        = Path(__file__).parent.parent
OUTPUTS_DIR = ROOT / "data" / "outputs"


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _get_path(filename: str) -> "Path | None":
    """Resolve file: data/outputs/ first, then project root."""
    p1 = OUTPUTS_DIR / filename
    p2 = ROOT / filename
    if p1.exists():
        return p1
    if p2.exists():
        return p2
    return None


def _make_name(pid: str) -> str:
    FIRST = ["Arjun", "Sriya", "Rahul", "Ananya", "Vikram",
             "Kavya", "Rohan", "Ishani", "Sanjay", "Meera"]
    LAST  = ["S.", "V.", "K.", "M.", "R.", "N.", "P.", "D.", "G.", "A."]
    digits = [c for c in str(pid) if c.isdigit()]
    i1 = int(digits[-1])   % len(FIRST) if digits else 0
    i2 = int(digits[-2])   % len(LAST)  if len(digits) > 1 else 0
    return f"{FIRST[i1]} {LAST[i2]}"


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_predictions(split: str = "test") -> pd.DataFrame:
    """
    Load final risk scores from aether_final_dashboard_data CSV and attach
    per-branch scores (for the branch breakdown chart) from branch1–4 CSVs.

    Parameters
    ----------
    split : str
        "test"  — returns the held-out test cohort (default, all rows since
                   aether_final_dashboard_data has no split column)
        "train" — same file, returns all rows
        "all"   — same file, returns all rows

    Returns
    -------
    pd.DataFrame with columns:
        patient_id,
        risk_ckd, risk_sepsis, risk_nafld,   ← from aether_final_dashboard_data
        ci_ckd,   ci_sepsis,   ci_nafld,
        b1_ckd … b4_nafld,                   ← per-branch breakdown (optional)
        name, age, gender
    """

    # ── 1. Load the final pre-fused scores ───────────────────────────────────
    final_path = (
        _get_path("aether_final_dashboard_data.csv")
        or _get_path("aether_final_dashboard_data__1_.csv")
        or _get_path("aether_final_dashboard_data (1).csv")
    )

    if final_path is None:
        # Hard fallback: try branch1 + old fusion
        return _legacy_load(split)

    df = pd.read_csv(final_path)
    df["patient_id"] = df["patient_id"].astype(str).str.strip()

    # Rename the final-score columns to the standard names used everywhere
    df = df.rename(columns={
        "final_risk_ckd": "risk_ckd",
        "final_risk_sep": "risk_sepsis",
        "final_risk_naf": "risk_nafld",
    })

    # Ensure numeric
    for col in ["risk_ckd", "risk_sepsis", "risk_nafld"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").clip(0.001, 0.999)

    # ── 2. Attach per-branch scores (branch breakdown chart) ─────────────────
    # These are purely for the multi-modal evidence chart on the Profile page.
    # They do NOT influence the headline risk numbers anymore.
    branch_map = {
        "branch1_prediction.csv": ("b1", {"p_ckd": "b1_ckd", "p_sepsis": "b1_sepsis", "p_nafld": "b1_nafld"}),
        "branch2_prediction.csv": ("b2", {"p_ckd": "b2_ckd", "p_sepsis": "b2_sepsis", "p_nafld": "b2_nafld"}),
        "branch3_prediction.csv": ("b3", {"p_ckd": "b3_ckd", "p_sepsis": "b3_sepsis", "p_nafld": "b3_nafld"}),
        "branch4_prediction.csv": ("b4", {"p_ckd": "b4_ckd", "p_sepsis": "b4_sepsis", "p_nafld": "b4_nafld"}),
    }

    for filename, (prefix, rename) in branch_map.items():
        path = _get_path(filename)
        if path:
            b = pd.read_csv(path)
            b["patient_id"] = b["patient_id"].astype(str).str.strip()
            b = b.rename(columns=rename)
            keep = ["patient_id"] + list(rename.values())
            keep = [c for c in keep if c in b.columns]
            df = df.merge(b[keep], on="patient_id", how="left")
        else:
            for new_col in rename.values():
                df[new_col] = np.nan

    # Fill missing branch scores with the fused risk (so the chart still renders)
    for disease in ["ckd", "sepsis", "nafld"]:
        for i in range(1, 5):
            col = f"b{i}_{disease}"
            if col not in df.columns:
                df[col] = np.nan
            df[col] = df[col].fillna(df[f"risk_{disease}"])

    # ── 3. Confidence intervals from branch spread ────────────────────────────
    for disease in ["ckd", "sepsis", "nafld"]:
        branch_cols = [f"b{i}_{disease}" for i in range(1, 5)]
        df[f"ci_{disease}"] = (
            df[branch_cols].std(axis=1).fillna(0.05).clip(0.02, 0.20)
        )

    # ── 4. Patient metadata ───────────────────────────────────────────────────
    meta_path = _get_path("patient_metadata.csv")
    if meta_path:
        meta = pd.read_csv(meta_path)
        meta["patient_id"] = meta["patient_id"].astype(str).str.strip()
        df = df.merge(meta, on="patient_id", how="left")

    # ── 5. Deterministic display names ────────────────────────────────────────
    df["name"] = df["patient_id"].apply(_make_name)

    # ── 6. Split filter ───────────────────────────────────────────────────────
    # aether_final_dashboard_data has no split column; treat all rows as "test"
    if split != "all" and "split" in df.columns:
        df = df[df["split"].str.lower() == split.lower()]

    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE ACCESSORS (unchanged API)
# ─────────────────────────────────────────────────────────────────────────────

def get_patient(patient_id: str) -> dict:
    df  = load_predictions(split="all")
    row = df[df["patient_id"] == str(patient_id).strip()]
    return row.iloc[0].to_dict() if not row.empty else {}


def get_patient_list(split: str = "test") -> list:
    df = load_predictions(split=split)
    return df["patient_id"].unique().tolist() if not df.empty else []


# ─────────────────────────────────────────────────────────────────────────────
# EXPLAINABILITY HELPERS (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def load_shap_values(patient_id: str = None) -> pd.DataFrame:
    path = _get_path("shap_value.csv")
    if path is None:
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    if patient_id:
        df = df[df["patient_id"] == str(patient_id).strip()]
    return df


def load_temporal_values(patient_id: str = None) -> pd.DataFrame:
    path = _get_path("temporal_value.csv")
    if path is None:
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    if patient_id:
        df = df[df["patient_id"] == str(patient_id).strip()]
    return df


def load_bert_attention(patient_id: str = None) -> pd.DataFrame:
    path = _get_path("bert_attention.csv")
    if path is None:
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    if patient_id:
        df = df[df["patient_id"] == str(patient_id).strip()]
    return df


# ─────────────────────────────────────────────────────────────────────────────
# NEW PATIENT PREDICTOR (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def predict_new_patient(inputs: dict) -> dict:
    def _get(key, default=0.0):
        v = inputs.get(key, default)
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    ckd_score, ckd_factors   = 0.0, []
    sep_score, sep_factors   = 0.0, []
    naf_score, naf_factors   = 0.0, []

    creatinine = _get("creatinine")
    if creatinine > 0:
        if creatinine > 3.0:   ckd_score += 0.45; ckd_factors.append(f"Creatinine severely elevated ({creatinine:.1f} mg/dL)")
        elif creatinine > 2.0: ckd_score += 0.30; ckd_factors.append(f"Creatinine elevated ({creatinine:.1f} mg/dL)")
        elif creatinine > 1.4: ckd_score += 0.15; ckd_factors.append(f"Creatinine borderline ({creatinine:.1f} mg/dL)")

    egfr = _get("egfr")
    if egfr > 0:
        if egfr < 15:   ckd_score += 0.45; ckd_factors.append(f"eGFR critically low ({egfr:.0f})")
        elif egfr < 30: ckd_score += 0.35; ckd_factors.append(f"eGFR severely reduced ({egfr:.0f})")
        elif egfr < 45: ckd_score += 0.20; ckd_factors.append(f"eGFR moderately reduced ({egfr:.0f})")
        elif egfr < 60: ckd_score += 0.10; ckd_factors.append(f"eGFR mildly reduced ({egfr:.0f})")

    bun = _get("bun")
    if bun > 25: ckd_score += 0.10; ckd_factors.append(f"BUN elevated ({bun:.0f})")
    if bun > 50: ckd_score += 0.10

    sbp = _get("systolic_bp")
    if sbp > 160:    ckd_score += 0.12; ckd_factors.append(f"Severe hypertension ({sbp:.0f})")
    elif sbp > 140:  ckd_score += 0.06; ckd_factors.append(f"Hypertension ({sbp:.0f})")

    if _get("proteinuria"):   ckd_score += 0.15; ckd_factors.append("Proteinuria present")
    if _get("diabetes"):      ckd_score += 0.08; ckd_factors.append("Diabetes (CKD risk factor)")
    if _get("hypertension"):  ckd_score += 0.06; ckd_factors.append("Hypertension history")
    if _get("reduced_urine"): ckd_score += 0.10; ckd_factors.append("Reduced urine output")

    wbc = _get("wbc")
    if wbc > 0:
        if wbc > 20 or wbc < 2:    sep_score += 0.30; sep_factors.append(f"WBC critically abnormal ({wbc:.1f})")
        elif wbc > 12 or wbc < 4:  sep_score += 0.15; sep_factors.append(f"WBC abnormal ({wbc:.1f})")

    lactate = _get("lactate")
    if lactate > 4.0:   sep_score += 0.40; sep_factors.append(f"Lactate severely elevated ({lactate:.1f})")
    elif lactate > 2.0: sep_score += 0.20; sep_factors.append(f"Lactate elevated ({lactate:.1f})")

    temp = _get("temperature")
    if temp > 0:
        if temp > 39.0 or temp < 35.5: sep_score += 0.20; sep_factors.append(f"Abnormal temperature ({temp:.1f}°C)")
        elif temp > 38.3:               sep_score += 0.10; sep_factors.append(f"Fever ({temp:.1f}°C)")

    spo2 = _get("spo2")
    if 0 < spo2 < 90:  sep_score += 0.25; sep_factors.append(f"SpO2 critically low ({spo2:.0f}%)")
    elif 0 < spo2 < 95: sep_score += 0.10; sep_factors.append(f"SpO2 low ({spo2:.0f}%)")

    if sbp > 0 and sbp < 90: sep_score += 0.35; sep_factors.append(f"Hypotension ({sbp:.0f})")
    if _get("fever"):   sep_score += 0.15; sep_factors.append("Fever reported")
    if _get("fatigue"): sep_score += 0.05; sep_factors.append("Fatigue present")

    alt = _get("alt"); ast = _get("ast")
    if alt > 40:
        naf_score += 0.25 if alt > 80 else 0.12
        naf_factors.append(f"ALT {'significantly ' if alt > 80 else ''}elevated ({alt:.0f})")
    if alt > 0 and ast > 0 and (ast / alt) < 1.0 and alt > 40:
        naf_score += 0.10; naf_factors.append("AST/ALT ratio <1 (NAFLD pattern)")
    if ast > 80: naf_score += 0.10; naf_factors.append(f"AST elevated ({ast:.0f})")

    bmi = _get("bmi")
    if bmi > 30:    naf_score += 0.20; naf_factors.append(f"Obesity (BMI {bmi:.1f})")
    elif bmi > 25:  naf_score += 0.08; naf_factors.append(f"Overweight (BMI {bmi:.1f})")

    hba1c = _get("hba1c")
    if hba1c > 6.5:  naf_score += 0.15; naf_factors.append(f"HbA1c elevated ({hba1c:.1f}%)")
    elif hba1c > 5.7: naf_score += 0.07; naf_factors.append(f"HbA1c borderline ({hba1c:.1f}%)")

    fib4 = _get("fib4")
    if fib4 > 2.67:  naf_score += 0.35; naf_factors.append(f"FIB-4 high ({fib4:.2f})")
    elif fib4 > 1.3: naf_score += 0.15; naf_factors.append(f"FIB-4 intermediate ({fib4:.2f})")

    if _get("diabetes"):    naf_score += 0.10; naf_factors.append("Diabetes (metabolic overlap)")
    if _get("obesity"):     naf_score += 0.12; naf_factors.append("Obesity reported")
    if _get("jaundice"):    naf_score += 0.15; naf_factors.append("Jaundice")
    if _get("hypertension"): naf_score += 0.05; naf_factors.append("Hypertension")

    n_inputs = sum(1 for v in inputs.values() if v and float(v) != 0.0)
    ci_base  = max(0.05, 0.20 - n_inputs * 0.008)

    return {
        "risk_ckd":       float(np.clip(ckd_score, 0.01, 0.99)),
        "risk_sepsis":    float(np.clip(sep_score, 0.01, 0.99)),
        "risk_nafld":     float(np.clip(naf_score, 0.01, 0.99)),
        "ci_ckd":         round(ci_base, 3),
        "ci_sepsis":      round(ci_base, 3),
        "ci_nafld":       round(ci_base, 3),
        "ckd_factors":    ckd_factors    or ["No significant CKD risk markers detected"],
        "sepsis_factors": sep_factors    or ["No significant Sepsis risk markers detected"],
        "nafld_factors":  naf_factors    or ["No significant NAFLD risk markers detected"],
        "n_inputs_used":  n_inputs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# LEGACY FALLBACK (only used when aether_final_dashboard_data is missing)
# ─────────────────────────────────────────────────────────────────────────────

_WEIGHTS = {
    "ckd":    {"b1": 0.20, "b2": 0.30, "b3": 0.40, "b4": 0.10},
    "sepsis": {"b1": 0.20, "b2": 0.30, "b3": 0.40, "b4": 0.10},
    "nafld":  {"b1": 0.20, "b2": 0.35, "b3": 0.45, "b4": 0.10},
}

def _fuse_row(row, disease):
    w = _WEIGHTS[disease]
    vals, avail_w = {}, {}
    for i in range(1, 5):
        col = f"b{i}_{disease}"
        v = row.get(col)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            vals[f"b{i}"] = float(v)
            avail_w[f"b{i}"] = w[f"b{i}"]
    if not vals:
        return 0.30
    total_w = sum(avail_w.values())
    return sum(vals[k] * avail_w[k] / total_w for k in vals)

def _legacy_load(split: str = "test") -> pd.DataFrame:
    b1_path = _get_path("branch1_prediction.csv")
    if b1_path is None:
        return _dummy_predictions(split)
    b1 = pd.read_csv(b1_path)
    b1["patient_id"] = b1["patient_id"].astype(str).str.strip()
    b1 = b1.rename(columns={"p_ckd": "b1_ckd", "p_sepsis": "b1_sepsis", "p_nafld": "b1_nafld"})
    merged = b1.copy()
    for i, fname in [(2, "branch2_prediction.csv"), (3, "branch3_prediction.csv"), (4, "branch4_prediction.csv")]:
        path = _get_path(fname)
        prefix = f"b{i}"
        if path:
            b = pd.read_csv(path)
            b["patient_id"] = b["patient_id"].astype(str).str.strip()
            b = b.rename(columns={"p_ckd": f"{prefix}_ckd", "p_sepsis": f"{prefix}_sepsis", "p_nafld": f"{prefix}_nafld"})
            merged = merged.merge(b[["patient_id", f"{prefix}_ckd", f"{prefix}_sepsis", f"{prefix}_nafld"]], on="patient_id", how="left")
        else:
            for d in ["ckd", "sepsis", "nafld"]:
                merged[f"{prefix}_{d}"] = np.nan
    for disease in ["ckd", "sepsis", "nafld"]:
        b1_col = f"b1_{disease}"
        for i in range(2, 5):
            col = f"b{i}_{disease}"
            if col in merged.columns:
                merged[col] = merged[col].fillna(merged[b1_col])
    for disease in ["ckd", "sepsis", "nafld"]:
        merged[f"risk_{disease}"] = merged.apply(lambda r: _fuse_row(r, disease), axis=1).clip(0.001, 0.999)
        cols = [f"b{i}_{disease}" for i in range(1, 5) if f"b{i}_{disease}" in merged.columns]
        merged[f"ci_{disease}"] = merged[cols].std(axis=1).fillna(0.05).clip(0.02, 0.20)
    meta_path = _get_path("patient_metadata.csv")
    if meta_path:
        meta = pd.read_csv(meta_path)
        meta["patient_id"] = meta["patient_id"].astype(str).str.strip()
        merged = merged.merge(meta, on="patient_id", how="left")
    merged["name"] = merged["patient_id"].apply(_make_name)
    if split != "all" and "split" in merged.columns:
        merged = merged[merged["split"].str.lower() == split.lower()]
    return merged.reset_index(drop=True)

def _dummy_predictions(split: str = "test") -> pd.DataFrame:
    np.random.seed(42)
    n = 15
    df = pd.DataFrame({
        "patient_id":  [f"P{str(i).zfill(4)}" for i in range(n)],
        "b1_ckd":    np.random.beta(2.5, 2.5, n),
        "b1_sepsis": np.random.beta(2.5, 2.5, n),
        "b1_nafld":  np.random.beta(1.5, 5.0, n),
        "b2_ckd":    np.random.beta(2.5, 2.5, n),
        "b2_sepsis": np.random.beta(2.5, 2.5, n),
        "b2_nafld":  np.random.beta(2.0, 2.0, n),
        "b3_ckd":    np.random.beta(2.0, 3.0, n),
        "b3_sepsis": np.random.beta(2.0, 3.0, n),
        "b3_nafld":  np.random.beta(1.5, 4.0, n),
        "b4_ckd":    np.random.beta(2.5, 2.5, n),
        "b4_sepsis": np.random.beta(2.5, 2.5, n),
        "b4_nafld":  np.random.beta(1.5, 5.0, n),
        "split": "test",
        "age":   np.random.randint(40, 85, n),
        "gender": np.random.choice(["M", "F"], n),
    })
    for d in ["ckd", "sepsis", "nafld"]:
        df[f"risk_{d}"] = df.apply(lambda r: _fuse_row(r, d), axis=1).clip(0.001, 0.999)
        cols = [f"b{i}_{d}" for i in range(1, 5)]
        df[f"ci_{d}"] = df[cols].std(axis=1).fillna(0.05).clip(0.02, 0.20)
    df["name"] = df["patient_id"].apply(_make_name)
    return df