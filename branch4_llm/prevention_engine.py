import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from branch4_llm.rag.retriever import build_context
from branch4_llm.llm.gemini_client import call_gemini
from branch4_llm.llm.prompts import prevention_plan_prompt, chatbot_prompt


def generate_prevention_plan(patient_id: str, patient_data: dict) -> dict:
    risks = {
        "ckd":    float(patient_data.get("risk_ckd", 0.5)),
        "sepsis": float(patient_data.get("risk_sepsis", 0.3)),
        "nafld":  float(patient_data.get("risk_nafld", 0.4)),
    }
    ci = {
        "ckd":    float(patient_data.get("ci_ckd", 0.05)),
        "sepsis": float(patient_data.get("ci_sepsis", 0.05)),
        "nafld":  float(patient_data.get("ci_nafld", 0.05)),
    }

    chunks, context = build_context(risks)
    prompt   = prevention_plan_prompt(patient_id, risks, ci, context)
    plan     = call_gemini(prompt)

    return {
        "patient_id":  patient_id,
        "risks":       risks,
        "plan_text":   plan,
        "sources":     list(set(c["source"] for c in chunks)),
        "chunks_used": len(chunks),
    }


def chat_response(question: str, patient_data: dict, history: list) -> str:
    risks = {
        "ckd":    float(patient_data.get("risk_ckd", 0.5)),
        "sepsis": float(patient_data.get("risk_sepsis", 0.3)),
        "nafld":  float(patient_data.get("risk_nafld", 0.4)),
    }

    patient_ctx = (
        f"CKD risk: {int(risks['ckd']*100)}%  |  "
        f"Sepsis risk: {int(risks['sepsis']*100)}%  |  "
        f"NAFLD risk: {int(risks['nafld']*100)}%"
    )

    _, context = build_context(risks)
    prompt     = chatbot_prompt(question, patient_ctx, context, history)
    return call_gemini(prompt)