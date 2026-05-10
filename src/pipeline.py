import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# 1 — Load data
df = pd.read_csv("data/combined_dataset.csv")
print("Shape:", df.shape)

# 2 — TF-IDF
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['text'])
print("TF-IDF Shape:", X.shape)

# 3 — Label encode
le = LabelEncoder()
y = le.fit_transform(df['label'])

# 4 — Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 5 — XGBoost
model = XGBClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6 — Evaluate
y_pred = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, 
      target_names=le.classes_))

# 7 — Save model & vectorizer
import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_dir = os.path.join(base_dir, "models")
os.makedirs(models_dir, exist_ok=True)

joblib.dump(model, os.path.join(models_dir, "document_classifier.pkl"))
joblib.dump(tfidf, os.path.join(models_dir, "tfidf_vectorizer.pkl"))
joblib.dump(le, os.path.join(models_dir, "label_encoder.pkl"))
print("Models saved to:", models_dir)