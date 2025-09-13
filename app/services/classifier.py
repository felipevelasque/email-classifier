# app/services/classifier.py
from typing import List, Tuple
from pathlib import Path
from fastapi import UploadFile, HTTPException
import io, re, requests
from pdfminer.high_level import extract_text as pdf_extract_text

from app.core.settings import HF_TOKEN, HF_MODEL

# --- regex e utilitários
HTML_TAG_RE = re.compile(r"<[^>]+>")
SIGN_RE = re.compile(r"(?is)(atenciosamente,.*$|kind regards,.*$|--+\s*\n.*$)")

def clean_text(text: str) -> str:
    text = text or ""
    text = HTML_TAG_RE.sub(" ", text)
    text = SIGN_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()

def normalize(text: str) -> str:
    try:
        from unidecode import unidecode
        return unidecode(text.lower())
    except Exception:
        return text.lower()

# --- leitura de arquivos
def read_txt_pdf(file: UploadFile) -> str:
    name = (file.filename or "").lower()
    blob = file.file.read()
    if name.endswith(".txt"):
        return blob.decode(errors="ignore")
    if name.endswith(".pdf"):
        try:
            with io.BytesIO(blob) as f:
                return pdf_extract_text(f)
        except Exception:
            raise HTTPException(415, detail="PDF não suportado (envie PDF pesquisável)")

    raise HTTPException(415, detail="Formato não suportado. Use .txt ou .pdf.")

# --- sinais
POS_SIGNALS = {
    "anexo": 1.4, "arquivo": 1.2, "solicitacao": 1.3, "pedido": 1.0, "status": 1.4,
    "andamento": 1.1, "atualizacao": 1.1, "erro": 1.5, "sistema": 1.0,
    "prazo": 1.2, "urgente": 1.4, "nota fiscal": 1.2, "nf": 1.1, "contrato": 1.2,
    "chamado": 1.3, "protocolo": 1.0, "boleto": 1.0, "fatura": 1.0
}
NEG_SIGNALS = {
    "obrigado": 2.2, "muito obrigado": 2.5, "agradeco": 2.2, "agradecimento": 2.5,
    "feliz natal": 2.5, "feliz ano": 2.2, "ano novo": 2.0, "parabens": 2.0,
    "newsletter": 1.6, "divulgacao": 1.5, "marketing": 1.5, "convite": 1.4,
    "bom dia": 0.6, "boa tarde": 0.6, "boa noite": 0.6,
}

def detect_signals(text_norm: str) -> Tuple[List[str], List[str], float]:
    pos_hits, neg_hits = [], []
    score = 0.0
    for key, w in POS_SIGNALS.items():
        if key in text_norm:
            pos_hits.append(key); score += w
    for key, w in NEG_SIGNALS.items():
        if key in text_norm:
            neg_hits.append(key); score -= w
    return pos_hits, neg_hits, score

# --- regras contextuais
ACTION_HINTS = {
    "status","andamento","prazo","erro","atualizacao","protocolo",
    "chamado","contrato","boleto","fatura","nota fiscal","nf","anexo","arquivo",
    "verificar","verifiquem","poderiam verificar","podem verificar","processado",
    "emitir","emissao","resolucao","correcao","resolver","eta"
}
GRATITUDE_HINTS = {
    "obrigado","muito obrigado","agradeco","agradecimento",
    "feliz natal","feliz ano","ano novo","parabens"
}
FUNCTIONING_PHRASES = {
    "tudo funcionando","funcionando perfeitamente","problema resolvido",
    "issue resolvida","resolvido"
}

def _has_any(text_norm: str, vocab: set[str]) -> bool:
    return any(term in text_norm for term in vocab)

def rule_classifier(text: str) -> Tuple[str, float, List[str]]:
    text_clean = clean_text(text)
    text_norm = normalize(text_clean)

    pos_hits, neg_hits, base_score = detect_signals(text_norm)

    pos_bonus = 0.0
    neg_bonus = 0.0
    if _has_any(text_norm, ACTION_HINTS): pos_bonus += 0.6
    if _has_any(text_norm, FUNCTIONING_PHRASES): neg_bonus += 0.8

    score = base_score + pos_bonus - neg_bonus

    if score > 0.6: category = "Produtivo"
    elif score < -0.6: category = "Improdutivo"
    else: category = "Produtivo"

    if _has_any(text_norm, GRATITUDE_HINTS) and not _has_any(text_norm, ACTION_HINTS):
        category = "Improdutivo"; conf_val = 0.80
    else:
        raw = abs(score)
        conf_val = 0.55 + min(raw / 6.0, 0.35)

    signals = list(dict.fromkeys(pos_hits + neg_hits))[:8]
    return category, round(conf_val, 2), signals

# --- HF zero-shot
def hf_zero_shot(text: str) -> tuple[str, float] | None:
    if not HF_TOKEN:
        return None
    try:
        url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        payload = {
            "inputs": text,
            "parameters": {
                "candidate_labels": ["Produtivo", "Improdutivo"],
                "hypothesis_template": "Este email requer uma ação imediata da equipe: {}."
            }
        }
        r = requests.post(url, headers=headers, json=payload, timeout=15)
        r.raise_for_status()
        data = r.json()
        labels = data.get("labels") or []
        scores = data.get("scores") or []
        if not labels or not scores:
            return None
        return labels[0], float(scores[0])
    except Exception:
        return None

# --- pipeline único chamado pela rota
def classify_email(content: str) -> tuple[str, float, list, dict]:
    """
    Retorna: category, confidence, signals, meta_overrides
    meta_overrides: ex.: {"gratitude_no_action": True, "action_over_prod_low_conf": False}
    """
    text_clean = clean_text(content)
    norm = normalize(text_clean)

    meta_over = {"gratitude_no_action": False, "action_over_prod_low_conf": False}

    hf_result = hf_zero_shot(text_clean)
    if hf_result:
        category, confidence = hf_result
        used_hf = True
    else:
        category, confidence, _ = rule_classifier(content)
        used_hf = False

    pos_hits, neg_hits, _ = detect_signals(norm)
    signals = list(dict.fromkeys(pos_hits + neg_hits))[:8]

    # filtro 'nf' ruído
    import re
    def looks_like_nf(txt: str) -> bool:
        return ("nota fiscal" in txt) or bool(re.search(r"\bnf\b", txt))
    if "nf" in signals and not looks_like_nf(norm):
        signals = [s for s in signals if s != "nf"]

    # override #1: gratidão/felicitação sem pedido
    gratitude_terms = GRATITUDE_HINTS
    action_terms = {
        "status","andamento","prazo","erro","atualiza","protocolo",
        "chamado","contrato","boleto","fatura","nota fiscal","nf","anexo","arquivo",
        "verificar","processado","emitir","emissao","correcao","resolver","eta"
    }
    has_gratitude = any(t in norm for t in gratitude_terms)
    has_action = any(t in norm for t in action_terms)
    if has_gratitude and not has_action:
        category = "Improdutivo"
        confidence = max(float(confidence or 0.0), 0.80)
        meta_over["gratitude_no_action"] = True
        if "obrigado" not in signals and any(t in norm for t in ["obrigado","muito obrigado"]):
            signals = ["obrigado"] + signals

    # override #2: ação detectada, modelo disse improdutivo com baixa confiança
    if has_action and category == "Improdutivo" and float(confidence or 0.0) < 0.80:
        category = "Produtivo"
        confidence = max(float(confidence or 0.0), 0.75)
        meta_over["action_over_prod_low_conf"] = True

    return category, round(float(confidence), 2), signals, {"used_hf": used_hf, "overrides": meta_over}
