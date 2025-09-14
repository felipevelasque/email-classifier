# app/schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class AnalyzeMeta(BaseModel):
    language: str = "pt"
    signals: List[str] = []
    used_hf: bool = False
    used_openai: bool = False
    fallbacks: List[str] = []
    overrides: Optional[Dict[str, Any]] = None
    elapsed_ms: Optional[int] = None
    output_size: Optional[int] = None

class AnalyzeResponse(BaseModel):
    category: str = Field(pattern="^(Produtivo|Improdutivo)$")
    confidence: float
    reply: str
    meta: AnalyzeMeta
