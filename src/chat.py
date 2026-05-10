import os
import sys
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
sys.stdout.reconfigure(encoding='utf-8')  

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

def generate_document_info(text: str, category: str, entities: dict) -> str:
    
    system_prompt = f"""You are a Document Intelligence Assistant.
    You analyze {category} documents.
    Be professional and concise."""

    user_prompt = f"""Document Category: {category}
    Extracted Entities: {entities}
    
    Document Text: {text[:500]}
    
    Provide a professional summary."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    return response.choices[0].message.content