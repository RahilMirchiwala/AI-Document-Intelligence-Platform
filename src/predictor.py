# predictor.py
import os
import joblib

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_dir = os.path.join(base_dir, "models")

model = joblib.load(os.path.join(models_dir, "document_classifier.pkl"))
tfidf = joblib.load(os.path.join(models_dir, "tfidf_vectorizer.pkl"))
le = joblib.load(os.path.join(models_dir, "label_encoder.pkl"))

def predict_document(text: str) -> str:
    X = tfidf.transform([text])
    prediction = model.predict(X)[0]
    return le.inverse_transform([prediction])[0]