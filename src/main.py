from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# imports
from models import DocumentInput, DocumentResponse
from predictor import predict_document
from extractor import extract_entities
from chat import generate_document_info

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# UI serve karo
@app.get("/")
async def serve_ui():
    return FileResponse("../ui/index.html")

@app.post("/analyze", response_model=DocumentResponse)
async def analyze_document(input: DocumentInput):
    category = predict_document(input.text)        # Step 1
    entities = extract_entities(input.text, category)  # Step 2
    summary = generate_document_info(input.text, category, entities)  # Step 3
    
    return DocumentResponse(
        category=category,
        entities=entities,
        summary=summary
    )