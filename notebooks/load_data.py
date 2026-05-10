import pandas as pd

# 1 — Medical
medical = pd.read_csv("data/mtsamples.csv")
medical = medical[['transcription', 'medical_specialty']].dropna()
medical.columns = ['text', 'label']
medical['label'] = 'Medical'

# 2 — Legal
legal = pd.read_csv("data/legal_text_classification.csv")
legal = legal[['case_text', 'case_outcome']].dropna()
legal.columns = ['text', 'label']
legal['label'] = 'Legal'

# 3 — Financial
financial = pd.read_csv("data/all-data.csv",
                        header=None,
                        names=['label', 'text'],
                        encoding='latin-1')
financial['label'] = 'Financial'

# Balance — 4000 each
medical_s = medical.sample(4000, random_state=42)
legal_s = legal.sample(4000, random_state=42)
financial_s = financial.sample(4000, random_state=42)

# Combine
df = pd.concat([medical_s, legal_s, financial_s], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("Final Shape:", df.shape)
print("\nLabel distribution:")
print(df['label'].value_counts())

# Save karo
df.to_csv("data/combined_dataset.csv", index=False)
print("\nSaved!")