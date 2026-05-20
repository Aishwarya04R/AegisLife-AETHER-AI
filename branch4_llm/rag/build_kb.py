import chromadb
from chromadb.utils import embedding_functions
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from shared.config import CHROMA_DIR

EMBED_FN = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

GUIDELINES = [
    # ── CKD ──────────────────────────────────────────────────
    {
        "id": "ckd_diet_1", "disease": "CKD", "topic": "diet", "source": "KDOQI 2020",
        "text": "For CKD stages 3-5, dietary protein intake should be restricted to 0.6-0.8g per kg body weight per day to slow disease progression. Plant-based protein sources are preferred as they generate fewer uremic toxins than animal protein."
    },
    {
        "id": "ckd_diet_2", "disease": "CKD", "topic": "diet", "source": "KDOQI 2020",
        "text": "Sodium intake should be restricted to less than 2g per day in CKD patients to control blood pressure and reduce proteinuria. Processed foods, canned goods, and fast food are primary sources of excess sodium and should be avoided."
    },
    {
        "id": "ckd_diet_3", "disease": "CKD", "topic": "diet", "source": "KDOQI 2020",
        "text": "Potassium restriction to 2000-3000mg per day is recommended for CKD patients with hyperkalemia. High potassium foods to avoid include bananas, oranges, potatoes, tomatoes, and dairy products."
    },
    {
        "id": "ckd_diet_4", "disease": "CKD", "topic": "diet", "source": "KDOQI 2020",
        "text": "Phosphorus intake should be limited to 800-1000mg per day in CKD patients to prevent secondary hyperparathyroidism. Avoid processed foods with phosphate additives, cola drinks, and dairy products high in phosphorus."
    },
    {
        "id": "ckd_hydration_1", "disease": "CKD", "topic": "hydration", "source": "KDOQI 2020",
        "text": "Patients with CKD and preserved urine output should maintain adequate hydration of 1.5-2L per day. Adequate fluid intake prevents dehydration-related AKI. Patients approaching dialysis may require individualized fluid restriction based on urine output."
    },
    {
        "id": "ckd_exercise_1", "disease": "CKD", "topic": "exercise", "source": "ACC/AHA 2018",
        "text": "CKD patients should engage in moderate-intensity aerobic exercise for at least 150 minutes per week. Walking, cycling, and swimming are ideal. High-intensity exercise should be avoided in patients with eGFR below 30 ml/min/1.73m2."
    },
    {
        "id": "ckd_bp_1", "disease": "CKD", "topic": "blood_pressure", "source": "KDIGO 2022",
        "text": "Target blood pressure in CKD patients is below 120/80 mmHg when tolerated. ACE inhibitors or ARBs are first-line antihypertensive therapy in CKD patients with proteinuria due to their nephroprotective effects beyond blood pressure lowering."
    },
    {
        "id": "ckd_monitoring_1", "disease": "CKD", "topic": "monitoring", "source": "KDIGO 2022",
        "text": "eGFR and urine albumin-to-creatinine ratio should be measured every 3 months in high-risk CKD patients with eGFR below 30 or UACR above 300. Blood pressure should be monitored at every clinical encounter."
    },

    # ── Sepsis ───────────────────────────────────────────────
    {
        "id": "sepsis_bundle_1", "disease": "Sepsis", "topic": "treatment", "source": "Surviving Sepsis Campaign 2021",
        "text": "The Sepsis Hour-1 Bundle includes: measure lactate level, obtain blood cultures before antibiotics, administer broad-spectrum antibiotics, begin 30mL/kg crystalloid fluid for hypotension or lactate above 4mmol/L, and apply vasopressors for hypotension during or after fluid resuscitation."
    },
    {
        "id": "sepsis_prevention_1", "disease": "Sepsis", "topic": "prevention", "source": "Surviving Sepsis Campaign 2021",
        "text": "Sepsis prevention in high-risk patients includes early identification of infection, prompt antibiotic administration within 1 hour of recognition, adequate fluid resuscitation, and close monitoring of organ function. Hand hygiene and infection control reduce hospital-acquired sepsis significantly."
    },
    {
        "id": "sepsis_nutrition_1", "disease": "Sepsis", "topic": "nutrition", "source": "SCCM/ASPEN 2016",
        "text": "Early enteral nutrition should be initiated within 24-48 hours of ICU admission in hemodynamically stable sepsis patients. Target caloric intake is 25-30 kcal/kg/day with protein 1.2-2.0g/kg/day to prevent muscle wasting and support immune function."
    },
    {
        "id": "sepsis_monitoring_1", "disease": "Sepsis", "topic": "monitoring", "source": "Surviving Sepsis Campaign 2021",
        "text": "Continuous monitoring of lactate, MAP, urine output, and mental status is essential in sepsis management. Repeat lactate measurement within 2 hours if initial lactate is above 2 mmol/L. Target MAP above 65 mmHg with vasopressors if needed."
    },

    # ── NAFLD ────────────────────────────────────────────────
    {
        "id": "nafld_diet_1", "disease": "NAFLD", "topic": "diet", "source": "EASL 2016",
        "text": "A calorie-restricted diet targeting 7-10% body weight loss is the most effective dietary intervention for NAFLD. Weight loss of this magnitude significantly reduces hepatic steatosis, inflammation, and fibrosis in the majority of patients."
    },
    {
        "id": "nafld_diet_2", "disease": "NAFLD", "topic": "diet", "source": "EASL 2016",
        "text": "The Mediterranean diet is the most recommended dietary pattern for NAFLD: high in vegetables, fruits, whole grains, legumes, nuts, and olive oil; moderate in fish and poultry; low in red meat and processed foods. Fructose and sugar-sweetened beverages must be eliminated."
    },
    {
        "id": "nafld_diet_3", "disease": "NAFLD", "topic": "diet", "source": "AASLD 2023",
        "text": "Fructose consumption must be strictly limited in NAFLD patients as it directly promotes hepatic de novo lipogenesis. Patients should avoid all sugar-sweetened beverages, fruit juices, and foods with added high-fructose corn syrup completely."
    },
    {
        "id": "nafld_diet_4", "disease": "NAFLD", "topic": "diet", "source": "AASLD 2023",
        "text": "Coffee consumption of 2-3 cups per day has been associated with reduced liver fibrosis progression in NAFLD patients. Antioxidants in coffee may protect hepatocytes from oxidative stress. Unsweetened black coffee or coffee with minimal sugar is preferred."
    },
    {
        "id": "nafld_exercise_1", "disease": "NAFLD", "topic": "exercise", "source": "EASL 2016",
        "text": "Aerobic exercise of moderate intensity for 150-200 minutes per week significantly reduces hepatic fat content independent of weight loss. Resistance training 2-3 times per week improves insulin sensitivity and further reduces liver fat."
    },
    {
        "id": "nafld_exercise_2", "disease": "NAFLD", "topic": "exercise", "source": "AASLD 2023",
        "text": "Even without significant weight loss, regular physical activity of 200 minutes per week reduces liver fat by 20-30%. Patients should be counseled that exercise benefits liver health independently of body weight changes, to maintain long-term motivation."
    },
    {
        "id": "nafld_monitoring_1", "disease": "NAFLD", "topic": "monitoring", "source": "AASLD 2023",
        "text": "FIB-4 index should be calculated at every visit to monitor fibrosis progression in NAFLD patients. FIB-4 below 1.3 indicates low fibrosis risk, 1.3-2.67 requires further evaluation, and above 2.67 indicates advanced fibrosis requiring specialist referral."
    },
]


def build_knowledge_base():
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Delete and rebuild if already exists
    try:
        client.delete_collection("medical_guidelines")
        print("Deleted existing collection, rebuilding...")
    except:
        pass

    collection = client.create_collection(
        name="medical_guidelines",
        embedding_function=EMBED_FN,
        metadata={"hnsw:space": "cosine"}
    )

    collection.add(
        ids=[g["id"] for g in GUIDELINES],
        documents=[g["text"] for g in GUIDELINES],
        metadatas=[{k: v for k, v in g.items() if k not in ("text", "id")}
                   for g in GUIDELINES]
    )

    print(f"Knowledge base ready — {len(GUIDELINES)} guideline chunks stored.")
    return collection


if __name__ == "__main__":
    build_knowledge_base()