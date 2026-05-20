import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from branch4_llm.rag.retriever import build_context

# Simulate a high-risk CKD + NAFLD patient
risks = {"ckd": 0.78, "sepsis": 0.21, "nafld": 0.65}
chunks, context = build_context(risks)

print(f"Retrieved {len(chunks)} chunks\n")
print("=" * 60)
for c in chunks:
    print(f"[{c['disease']} — {c['topic']} — {c['source']}]")
    print(c['text'][:100] + "...")
    print()