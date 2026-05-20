import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from shared.data_loader import get_patient, get_patient_list
from branch4_llm.prevention_engine import generate_prevention_plan

# Pick the first test patient
patients = get_patient_list(split="test")
patient_id = patients[0]
patient_data = get_patient(patient_id)

print(f"Testing with patient: {patient_id}")
print(f"CKD risk:    {patient_data['risk_ckd']:.2f}")
print(f"Sepsis risk: {patient_data['risk_sepsis']:.2f}")
print(f"NAFLD risk:  {patient_data['risk_nafld']:.2f}")
print("\nGenerating prevention plan...\n")

result = generate_prevention_plan(patient_id, patient_data)

print("=" * 60)
print(result["plan_text"])
print("=" * 60)
print(f"\nSources used: {result['sources']}")
print(f"Chunks retrieved: {result['chunks_used']}")