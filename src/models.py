from pydantic import BaseModel

class DocumentInput(BaseModel):
    text: str

class DocumentResponse(BaseModel):
    category: str
    entities: dict
    summary: str