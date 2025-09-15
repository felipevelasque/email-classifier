# app/routers/analyze.py
import time, math
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path

from app.services.classifier import read_txt_pdf, classify_email, clean_text, detect_language
from app.services.replier import ai_reply, reply_template
from app.schemas import AnalyzeResponse
import logging
logger = logging.getLogger(__name__)


router = APIRouter(prefix="/api")

ROOT = Path(__file__).resolve().parents[2]
TEMPLATES = ROOT / "templates"
MAX_SIZE = 2 * 1024 * 1024  # 2 MB

@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    email_file: UploadFile | None = File(None),
    email_text: str | None = Form(None),
):
    start = time.perf_counter()

    # --- normaliza texto colado ---
    raw_text = (email_text or "").strip()

    # --- validação de arquivo: pode vir o campo sem arquivo real ---
    has_real_file = bool(
    email_file
    and getattr(email_file, "filename", "")  # pode vir "" em alguns clientes
    and email_file.filename.strip()
    )


    file_text = ""
    if has_real_file:
        # mede tamanho
        email_file.file.seek(0, 2)  # fim
        size = email_file.file.tell()
        email_file.file.seek(0)     # volta pro início
        if size > MAX_SIZE:
            raise HTTPException(413, detail="Arquivo muito grande. Limite: 2MB.")

        # lê conteúdo (422 se vazio/ilegível)
        file_text = read_txt_pdf(email_file)



    # --- decide conteúdo final (combina texto+arquivo se ambos vierem) ---
    if raw_text and file_text:
        content = f"{raw_text}\n\n---\n[CONTEÚDO DO ANEXO]\n{file_text}"
    elif raw_text:
        content = raw_text
    elif file_text:
        content = file_text
    else:
        # aqui sim: nem texto útil, nem arquivo válido
        raise HTTPException(400, detail="Envie um arquivo .txt/.pdf ou cole o texto do email.")

    # guarda erro específico se por algum motivo ficou vazio
    if not (content or "").strip():
        raise HTTPException(422, detail="Entrada vazia. Envie texto e/ou arquivo com conteúdo legível.")


    # --- pipeline de classificação ---
    text_clean = clean_text(content)
    snippet = text_clean[:1000]

    # 🔹 detectar idioma do e-mail
    lang = detect_language(snippet, default="pt")

    #classificar
    category, confidence, signals, info = classify_email(content)

    # --- resposta ---
    fallbacks = []
    used_openai = False
    ai_text = ai_reply(category, snippet, signals, lang=lang)
    if ai_text:
        reply_text = ai_text
        used_openai = True
    else:
        reply_text = reply_template(category, signals, lang=lang)
        fallbacks.append("templates")

    logger.info(
    "analyze_result",
    extra={
        "category": category,
        "confidence": confidence,
        "signals": signals[:6],  # limita o tamanho do log
        "used_hf": info.get("used_hf", False),
        "used_openai": used_openai,
        },
    )


    # --- meta ---
    elapsed_ms = max(1, math.ceil((time.perf_counter() - start) * 1000))
    meta = {
        "language": lang,
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
