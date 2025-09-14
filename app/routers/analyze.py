# app/routers/analyze.py
import time, math
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path

from app.services.classifier import read_txt_pdf, classify_email, clean_text
from app.services.replier import ai_reply, reply_template
from app.schemas import AnalyzeResponse

router = APIRouter()

ROOT = Path(__file__).resolve().parents[2]
TEMPLATES = ROOT / "templates"

@router.get("/")
async def index():
    return FileResponse(str(TEMPLATES / "index.html"))

@router.get("/healthz")
async def healthz():
    return {"ok": True}

@router.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(
    email_file: UploadFile | None = File(None),
    email_text: str | None = Form(None),
):
    start = time.perf_counter()

    if not email_file and not email_text:
        raise HTTPException(400, detail="Envie um arquivo .txt/.pdf ou cole o texto do email.")

    # conteúdo (combina texto + arquivo, se ambos vierem)
    raw_text = (email_text or "").strip()

    file_text = ""
    # Alguns navegadores podem mandar o campo mesmo sem arquivo; trate isso:
    if isinstance(email_file, UploadFile) and getattr(email_file, "filename", None):
        # Evita tentar ler quando o input veio, mas sem arquivo real
        if email_file.filename.strip():
            file_text = read_txt_pdf(email_file)  # sua função já lida com txt/pdf

    if raw_text and file_text:
        content = f"{raw_text}\n\n---\n[CONTEÚDO DO ANEXO]\n{file_text}"
    elif raw_text:
        content = raw_text
    elif file_text:
        content = file_text
    else:
        raise HTTPException(400, detail="Envie um arquivo .txt/.pdf ou cole o texto do email.")

    text_clean = clean_text(content)
    snippet = text_clean[:1000]

    # classificar
    category, confidence, signals, info = classify_email(content)

    # resposta
    fallbacks = []
    used_openai = False

    ai_text = ai_reply(category, snippet, signals)
    if ai_text:
        reply_text = ai_text
        used_openai = True
    else:
        reply_text = reply_template(category, signals)
        fallbacks.append("templates")

    # meta
    elapsed_ms = max(1, math.ceil((time.perf_counter() - start) * 1000))  
    meta = {
        "language": "pt",
        "signals": signals,
        "used_hf": info.get("used_hf", False),
        "used_openai": used_openai,
        "fallbacks": fallbacks,
        "overrides": info.get("overrides"),
        "elapsed_ms": elapsed_ms,
        "output_size": len(reply_text or ""),
    }

    return AnalyzeResponse(
        category=category,
        confidence=confidence,
        reply=reply_text,
        meta=meta,
    )
