from pathlib import Path
 
ROOT           = Path(__file__).parent.parent
DATA_DIR       = ROOT / "data"
OUTPUTS_DIR    = DATA_DIR / "outputs"
RAG_DIR        = ROOT / "branch4_llm" / "rag"
GUIDELINES_DIR = RAG_DIR / "guidelines"
CHROMA_DIR     = RAG_DIR / "chroma_db"
 
# Branch prediction files  (new naming from real outputs)
B1_PREDS = OUTPUTS_DIR / "branch1_prediction.csv"
B2_PREDS = OUTPUTS_DIR / "branch2_prediction.csv"
B3_PREDS = OUTPUTS_DIR / "branch3_prediction.csv"
B4_PREDS = OUTPUTS_DIR / "branch4_prediction.csv"
 
# Explainability files
SHAP_VALUES    = OUTPUTS_DIR / "shap_value.csv"
TEMPORAL_VALS  = OUTPUTS_DIR / "temporal_value.csv"
BERT_ATTENTION = OUTPUTS_DIR / "bert_attention.csv"
PATIENT_META   = OUTPUTS_DIR / "patient_metadata.csv"
 
#GEMINI_MODEL   = "gemini-2.5-flash"
GEMINI_MODEL    = "models/gemini-2.0-flash"