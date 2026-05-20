import chromadb
from chromadb.utils import embedding_functions
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from shared.config import CHROMA_DIR

EMBED_FN = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(
        name="medical_guidelines",
        embedding_function=EMBED_FN
    )

def retrieve(query: str, diseases: list, n_results: int = 5) -> list:
    collection = get_collection()

    # Only retrieve chunks relevant to the patient's high-risk diseases
    where = {"disease": {"$in": diseases}} if diseases else None

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where
    )

    chunks = []
    for i, doc in enumerate(results["documents"][0]):
        chunks.append({
            "text":    doc,
            "source":  results["metadatas"][0][i].get("source", ""),
            "disease": results["metadatas"][0][i].get("disease", ""),
            "topic":   results["metadatas"][0][i].get("topic", ""),
        })
    return chunks


def build_context(patient_risks: dict) -> tuple:
    """
    patient_risks = {"ckd": 0.78, "sepsis": 0.21, "nafld": 0.61}
    Returns (chunks list, formatted context string)
    """
    # Find which diseases are high risk (above 50%)
    high_risk_diseases = [
        d.upper() for d, v in patient_risks.items() if v > 0.5
    ]

    # Always include at least the highest risk disease
    if not high_risk_diseases:
        top = max(patient_risks, key=patient_risks.get)
        high_risk_diseases = [top.upper()]

    query = (
        f"diet nutrition exercise prevention monitoring recommendations "
        f"for {' and '.join(high_risk_diseases)} patients"
    )

    chunks = retrieve(query, diseases=high_risk_diseases, n_results=6)

    # Format into a clean context string for the LLM
    context = "\n\n".join([
        f"[Source: {c['source']} | Disease: {c['disease']} | Topic: {c['topic']}]\n{c['text']}"
        for c in chunks
    ])

    return chunks, context