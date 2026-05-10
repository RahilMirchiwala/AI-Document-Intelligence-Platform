import joblib
import os
from extractor import extract_entities
from chat import generate_document_info

# Absolute path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_dir = os.path.join(base_dir, "models")

model = joblib.load(os.path.join(models_dir, "document_classifier.pkl"))
tfidf = joblib.load(os.path.join(models_dir, "tfidf_vectorizer.pkl"))
le = joblib.load(os.path.join(models_dir, "label_encoder.pkl"))

def predict_document(text):
    X = tfidf.transform([text])
    prediction = model.predict(X)[0]
    return le.inverse_transform([prediction])[0]

# Test
medical_text = """PREOPERATIVE DIAGNOSIS: Chest pain and possible coronary artery disease.
HISTORY OF PRESENT ILLNESS:
The patient is a 58-year-old male with complaints of intermittent chest pain radiating to the left arm for the past 2 weeks. He reports associated shortness of breath and dizziness.
PAST MEDICAL HISTORY: Hypertension, diabetes mellitus type 2.
PHYSICAL EXAMINATION: Blood pressure 148/92. Heart sounds regular. No edema noted.
ASSESSMENT: Likely angina secondary to coronary artery disease.
PLAN: Recommend stress test, ECG, and cardiology consultation. Prescribe aspirin and metformin."""

legal_text = """THIS SERVICE AGREEMENT is entered into effective April 10, 2026 by and between Nexa Solutions Pvt Ltd and Daniel Roberts.
The parties hereby agree to the terms and conditions set forth in this contract.
COMPENSATION: Client agrees to pay a total amount of $120,000 payable in monthly installments.
CONFIDENTIALITY: All proprietary information shall remain protected.
TERMINATION: Either party may terminate with thirty days written notice.
JURISDICTION: This agreement shall be governed by laws of State of California."""

financial_text = """QUARTERLY FINANCIAL STATEMENT - Q1 2026
Company: TechVentures Global Inc.
Total revenue for Q1 2026 was reported at $6.2 million representing a growth of 14%.
Operating expenses increased to $2.1 million due to infrastructure investments.
Net income after taxes amounted to $1.4 million.
Reported from New York headquarters on April 15, 2026."""

# print("Medical prediction:", predict_document(medical_text))
# print("Legal prediction:", predict_document(legal_text))
# print("Financial prediction:", predict_document(financial_text))

# # Medical
# print("MEDICAL:")
# print(extract_entities(medical_text, "Medical"))

# # Legal
# print("\nLEGAL:")
# print(extract_entities(legal_text, "Legal"))

# # Financial
# print("\nFINANCIAL:")
# print(extract_entities(financial_text, "Financial"))

# Medical
category = predict_document(medical_text)
entities = extract_entities(medical_text,"Medical")

summary = generate_document_info(medical_text, category, entities)
print("\nMEDICAL SUMMARY:")
print(summary)

# Legal
category = predict_document(legal_text)
entities = extract_entities(legal_text, "Legal")
summary = generate_document_info(legal_text, category, entities)
print("\nLEGAL SUMMARY:")
print(summary)

# Financial
category = predict_document(financial_text)
entities = extract_entities(financial_text, "Financial")
summary = generate_document_info(financial_text, category, entities)
print("\nFINANCIAL SUMMARY:")
print(summary)