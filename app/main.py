from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import io, re
from typing import List, Tuple

try:
    from pdfminer.high_level import extract_text as pdf_extract_text 
    PDF_OK = True 
except Exception:
    PDF_OK = False

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "templates"
STATIC = ROOT / "static"

app = FastAPI(title="Email Auto Classifier (mínimo)")
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


HTML_TAG_RE = re.compile(r"<[^>]+>")
SIGN_RE = re.compile(r"(?is)(atenciosamente,.*$|kind regards,.*$|--+\s*\n.*$)")


# Sinais com pesos (bem simples; ajuste à vontade)
POS_SIGNALS = {
"anexo": 1.4, "arquivo": 1.2, "solicitacao": 1.3, "pedido": 1.0, "status": 1.4,
"andamento": 1.1, "atualizacao": 1.1, "erro": 1.5, "suporte": 1.2, "sistema": 1.0,
"prazo": 1.2, "urgente": 1.4, "nota fiscal": 1.2, "nf": 1.1, "contrato": 1.2,
"chamado": 1.3, "protocolo": 1.0, "boleto": 1.0, "fatura": 1.0
}
NEG_SIGNALS = {
"feliz natal": 1.5, "feliz ano": 1.3, "parabens": 1.2, "obrigado": 1.0, "agradeco": 1.0,
"convite": 1.0, "newsletter": 1.3, "divulgacao": 1.2, "marketing": 1.2,
"bom dia": 0.8, "boa tarde": 0.8, "comunicado": 0.9
}

def read_txt_pdf(file: UploadFile) -> str:
    name = (file.filename or "").lower()
    blob = file.file.read()
    if name.endswith(".txt"):
        return blob.decode(errors="ignore")
    if name.endswith(".pdf"):
        if not PDF_OK:
            raise HTTPException(415, detail="PDF não suportado (pdfminer não instalado)")
    with io.BytesIO(blob) as f:
        return pdf_extract_text(f)
    raise HTTPException(415, detail="Formato não suportado. Use .txt ou .pdf.")

def clean_text(text: str) -> str:
    text = text or ""
    text = HTML_TAG_RE.sub(" ", text)
    text = SIGN_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize(text: str) -> str:
    try:
        from unidecode import unidecode
        return unidecode(text.lower())
    except Exception:
        return text.lower()

def detect_signals(text_norm: str) -> Tuple[List[str], List[str], float]:
    pos_hits, neg_hits = [], []
    score = 0.0
    # checa expressões compostas primeiro
    for key, w in POS_SIGNALS.items():
        if key in text_norm:
            pos_hits.append(key)
            score += w
    for key, w in NEG_SIGNALS.items():
        if key in text_norm:
            neg_hits.append(key)
            score -= w
    return pos_hits, neg_hits, score

def rule_classifier(text: str) -> Tuple[str, float, List[str]]:
    text_clean = clean_text(text)
    text_norm = normalize(text_clean)
    pos_hits, neg_hits, score = detect_signals(text_norm)
    if score > 0.5:
        category = "Produtivo"
    elif score < -0.5:
        category = "Improdutivo"
    else:
        category = "Produtivo" # fail-safe


    # confiança aproximada (0.55–0.9)
    raw = abs(score)
    conf = 0.55 + min(raw / 6.0, 0.35)


    # sinais (mostramos os principais)
    signals = list(dict.fromkeys(pos_hits + neg_hits))[:8]
    return category, round(conf, 2), signals




def reply_template(category: str, signals: List[str]) -> str:
    has_attach = any(s in signals for s in ("anexo", "arquivo"))
    if category == "Produtivo":
        extra = " Se possível, anexe prints/arquivos." if not has_attach else ""
        return (
            "Olá, tudo bem? Recebemos sua mensagem e vamos dar andamento. "
            "Para agilizar, poderia confirmar o **ID do chamado** (ou dados do cliente) e a **data/horário** do ocorrido?"
            f"{extra} Nossa previsão para a primeira atualização é de **até 1 dia útil**. Ficamos à disposição."
        )
    else:
        return (
            "Olá! Obrigado pela mensagem. No momento **não é necessária nenhuma ação** da nossa equipe. "
            "Se precisar de algo, é só nos chamar. Abraços!"
        )



@app.get("/")
async def index():
    return FileResponse(str(TEMPLATES / "index.html"))




@app.post("/api/analyze")
async def analyze(email_file: UploadFile | None = File(None), email_text: str | None = Form(None)):
    if not email_file and not email_text:
        raise HTTPException(400, detail="Envie um arquivo .txt/.pdf ou cole o texto do email.")

    try:
        if email_file:
            content = read_txt_pdf(email_file)
        else:
            content = email_text or ""


        category, confidence, signals = rule_classifier(content)
        reply = reply_template(category, signals)


        return JSONResponse({
            "category": category,
            "confidence": confidence,
            "reply": reply,
            "meta": {
                "language": "pt", # simples (sem detecção neste mínimo)
                "signals": signals,
                "used_hf": False,
                "used_openai": False,
                "fallbacks": ["rules", "templates"]
            }
        })


    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=f"Erro ao processar: {e}")