# AI Document Intelligence Platform

Classify documents, extract entities, and generate AI summaries.

## Live Demo
https://ai-document-intelligence-platform-02mb.onrender.com

## What it does
- Classifies documents: Medical / Legal / Financial
- Extracts entities: Names, Dates, Amounts, Diagnosis
- Generates AI summary using Groq + Llama 3.3
- 99.45% classification accuracy

## Tech Stack
- Python, FastAPI
- XGBoost + TF-IDF (99.45% accuracy)
- spaCy NER (Entity Extraction)
- Groq API (Llama 3.3 70B)
- Pydantic, Joblib

## Run Locally
pip install -r src/requirements.txt
python -m spacy download en_core_web_sm
cd src && uvicorn main:app --reload

## Environment Variables
GROQ_API_KEY=your_key_here
