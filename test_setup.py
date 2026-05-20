from shared.data_loader import load_predictions, get_patient_list

df = load_predictions(split="test")
patients = get_patient_list()

print(f"Patients loaded: {len(df)}")
print(f"First patient: {patients[0]}")
print(f"Columns: {list(df.columns)}")
print("Setup working correctly!")