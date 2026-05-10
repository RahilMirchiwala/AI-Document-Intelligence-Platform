import spacy
import re

nlp = spacy.load("en_core_web_sm")

def extract_medical_custom(text: str) -> dict:
    # Age — "58-year-old" or "age 45"
    age = re.findall(r'\b(\d+)[-\s]year[-\s]old|\bage[:\s]+(\d+)', 
                     text, re.IGNORECASE)
    ages = [a[0] or a[1] for a in age]

    # Diagnosis — after keywords
    diagnosis = re.findall(
        r'(?:diagnosis|assessment|impression)[:\s]+([^\n.]+)', 
        text, re.IGNORECASE)

    # Medicines — common patterns
    medicines = re.findall(
        r'\b(aspirin|metformin|insulin|paracetamol|ibuprofen|amoxicillin|ECG|stress test)\b',
        text, re.IGNORECASE)

    return {
        "ages": ages,
        "diagnosis": diagnosis,
        "medicines": medicines
    }

def extract_legal_custom(text: str) -> dict:
    # Jurisdiction — "under jurisdiction of X" or "governed by laws of X"
    jurisdiction = re.findall(
        r'(?:jurisdiction|governed by|laws of)[:\s]+([^\n.,]+)',
        text, re.IGNORECASE)

    return {
        "jurisdiction": jurisdiction
    }

def extract_entities(text: str, category: str) -> dict:
    doc = nlp(text)

    persons, orgs, dates, money, locations = [], [], [], [], []

    for ent in doc.ents:
        if ent.label_ == "PERSON":
            persons.append(ent.text)
        elif ent.label_ == "ORG":
            orgs.append(ent.text)
        elif ent.label_ == "DATE":
            dates.append(ent.text)
        elif ent.label_ == "MONEY":
            money.append(ent.text)
        elif ent.label_ in ["GPE", "LOC"]:
            locations.append(ent.text)

    if category == "Medical":
        custom = extract_medical_custom(text)
        # JURISDICTION jaise words remove karo
        clean_persons = [p for p in persons if p not in 
                        ["JURISDICTION", "CONFIDENTIALITY", "TERMINATION"]]
        return {
            "patient_names": clean_persons,
            "ages": custom["ages"],
            "diagnosis": custom["diagnosis"],
            "medicines": custom["medicines"],
            "dates": dates,
            "medical_orgs": orgs
        }

    elif category == "Legal":
        custom = extract_legal_custom(text)
        clean_persons = [p for p in persons if p not in 
                        ["JURISDICTION", "CONFIDENTIALITY", "TERMINATION"]]
        return {
            "parties": clean_persons,
            "organizations": orgs,
            "dates": dates,
            "amounts": money,
            "jurisdiction": custom["jurisdiction"]
        }

    elif category == "Financial":
        return {
            "companies": orgs,
            "dates": dates,
            "revenue_amounts": money,
            "locations": locations
        }

    else:
        return {
            "persons": persons,
            "organizations": orgs,
            "dates": dates,
            "money": money
        }