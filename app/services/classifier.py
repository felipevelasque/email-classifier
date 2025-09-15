# app/services/classifier.py
from typing import List, Tuple
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
        return unidecode(text.lower().strip())
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

# --- sinais base
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

# termos que NÃO implicam ação sozinhos (contexto)
CONTEXT_ONLY_TERMS = {
    "chamado","contrato","protocolo","anexo","arquivo","nota fiscal","nf","fatura","boleto","cliente","conta"
}
# verbos/pedidos claros (pt/en/es)
REQUEST_TERMS = {
    # pt
    "verificar","poderiam verificar","podem verificar","informar","enviar","mandar",
    "emitir","atualizar","abrir","analisar","corrigir","resolver","processar","gerar",
    # en
    "check","could you","can you","please send","share","provide","issue","update","open","fix","resolve","process","generate",
    # es
    "verificar","podrian","pueden","enviar","mandar","emitir","actualizar","abrir","analizar","corregir","resolver","procesar","generar"
}
# termos informativos (status/prazo) — contam como ação se vierem em pergunta
INFO_TERMS = {"status","prazo","andamento","atualizacao","update","eta","estado","plazo"}


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

WELL_WISHES_TERMS = {
    "espero que estejam bem", "espero que esteja bem",
    "ótima semana", "otima semana",
    "boa semana", "boa jornada", "bom trabalho",
    "tenha um bom dia", "tenha uma boa semana",
    "desejo uma ótima semana", "desejo uma otima semana",
}


# saudações simples (pt) – não indicam ação por si só
GREETING_TERMS = {
    "ola", "olá", "oi", "bom dia", "boa tarde", "boa noite",
    "tudo bem", "como vai", "como está", "como esta"
}


# --- léxicos adicionais (pt/en/es) ---
ACTION_TERMS_EXTRA = {
    # pt
    "acompanhar","retorno","retornar","atualizacao","andamento","prazo","status","chamado","protocolo",
    "contrato","boleto","fatura","nota fiscal","anexo","arquivo","correcao","resolver","emissao","emitir","processado",
    # en
    "status","update","follow up","follow-up","ticket","eta","invoice","contract","attachment","attached",
    # es
    "estado","actualizacion","seguimiento","soporte","plazo","ticket","adjunto","factura","contrato",
}
MARKETING_TERMS = {
    "newsletter","divulgacao","marketing","convite","evento","webinar","lancamento","release","oferta","promocao"
}
GRATITUDE_TERMS = {
    "obrigado","muito obrigado","agradeco","agradecimento",
    "feliz natal","feliz ano","ano novo","parabens","gracias","thank you","thanks"
}
RESOLVED_TERMS = {
    "tudo funcionando","funcionando perfeitamente","problema resolvido","issue resolvida","resolvido",
    "nao preciso","não preciso","pode desconsiderar","pode cancelar","cancelar solicitacao","cancelada","cancelado"
}
URGENCY_TERMS = {"urgente","urgencia","asap","o mais rapido possivel","priority","prioridade"}

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

# --- overrides centralizados
def apply_overrides(norm: str, category: str, confidence: float, signals: list[str]) -> tuple[str, float, list[str], dict]:
    """
    Aplica regras finais de bom senso. Retorna (category, confidence, signals, meta_overrides).
    """
    meta = {
        "gratitude_no_action": False,
        "action_over_low_conf": False,
        "marketing_newsletter": False,
        "resolved_or_cancelled": False,
        "urgency_boost": False,
        "short_question_hint": False,
        "neutral-short": False,
        "noise_filter": [],
    }

    import re

    # --- filtro anti-ruído: 'nf' só vale se "nota fiscal" ou token isolado ---
    def _looks_like_nf(txt: str) -> bool:
        return ("nota fiscal" in txt) or bool(re.search(r"\bnf\b", txt))
    if "nf" in signals and not _looks_like_nf(norm):
        signals = [s for s in signals if s != "nf"]
        meta["noise_filter"].append("nf")

    # --- intenção de ação: verbo OU (termo de status + pergunta). 
    # Termos de contexto isolados NÃO contam como ação. ---
    has_request_verb = any(t in norm for t in REQUEST_TERMS)
    has_info_term   = any(t in norm for t in INFO_TERMS)
    has_question    = "?" in norm
    has_action = has_request_verb or (has_info_term and has_question)

    # (1) Gratidão/Felicitação sem pedido -> Improdutivo
    has_gratitude = any(t in norm for t in GRATITUDE_TERMS)
    if has_gratitude and not has_action:
        category = "Improdutivo"
        confidence = max(float(confidence or 0.0), 0.80)
        meta["gratitude_no_action"] = True
        # garante 'obrigado' uma única vez no topo
        if not any((s or "").strip().lower().startswith("obrigado") for s in signals):
            signals = ["obrigado"] + signals
    
    # (1.1) Saudação/boas-vindas sem pedido -> Improdutivo
    has_greeting = any(t in norm for t in GREETING_TERMS)
    has_well_wishes = any(t in norm for t in WELL_WISHES_TERMS)

    # sinal de pergunta real
    has_question = "?" in norm

    # Considera mensagens de saudação mesmo que um pouco maiores (até ~20 tokens),
    # desde que não haja pedido de ação e não seja pergunta.
    token_count = len(norm.split())

    if (has_greeting or has_well_wishes) and not has_action and not has_question and token_count <= 20:
        category = "Improdutivo"
        confidence = max(float(confidence or 0.0), 0.80)
        meta["greeting_only"] = True
        if "saudacao" not in signals:
            signals = ["saudacao"] + signals


    # (2) Marketing/Newsletter/Convite sem pedido -> Improdutivo
    if any(t in norm for t in MARKETING_TERMS) and not has_action:
        if category != "Improdutivo":
            category = "Improdutivo"
            confidence = max(float(confidence or 0.0), 0.75)
        meta["marketing_newsletter"] = True

    # (3) Resolvido/Cancelado/Desconsiderar -> SEMPRE Improdutivo (incondicional)
    if any(t in norm for t in RESOLVED_TERMS):
        category = "Improdutivo"
        confidence = max(float(confidence or 0.0), 0.85)
        meta["resolved_or_cancelled"] = True  # <<< agora dentro do if, com confiança maior

    # (4) Ação detectada, modelo disse Improdutivo com baixa confiança -> força Produtivo
    if has_action and category == "Improdutivo" and float(confidence or 0.0) < 0.80:
        category = "Produtivo"
        confidence = max(float(confidence or 0.0), 0.75)
        meta["action_over_low_conf"] = True

    # (5) Urgência -> boost na confiança se Produtivo
    if any(t in norm for t in URGENCY_TERMS) and category == "Produtivo":
        confidence = max(float(confidence or 0.0), 0.78)
        meta["urgency_boost"] = True
        if "urgente" in norm and "urgente" not in signals:
            signals = ["urgente"] + signals

    # (6) Pergunta/solicitação curta sobre status/prazo -> Produtivo com piso de confiança
    # Funciona COM ou SEM "?" (ex.: "e o status", "status por favor", "prazo do ticket 123")
    import re

    status_terms = {"status", "prazo", "andamento", "update", "eta", "ticket"}
    short_len = len(norm) <= 40
    # poucas palavras? ajuda a pegar frases curtinhas sem pontuação
    short_tokens = len(norm.split()) <= 6

    # padrões comuns de pergunta/solicitação sem interrogação
    looks_like_question = (
        "?" in norm
        or norm.strip() in {
            "status", "qual o status", "e o status", "como está o status",
            "status do chamado", "status do ticket", "e o prazo", "qual o prazo"
        }
        or any(norm.strip().endswith(t) for t in status_terms)
        or any(f"{t} por favor" in norm for t in status_terms)
        or bool(re.search(r"\b(qual|sobre|e\s*o|e\s*quanto)\b", norm))
    )

    has_status_term = any(t in norm for t in status_terms)

    if (short_len or short_tokens) and has_status_term and looks_like_question:
        category = "Produtivo"
        confidence = max(float(confidence or 0.0), 0.70)
        meta["short_question_hint"] = True

    # --- normalização de sinais ('obrigado' colapsado, sem duplicatas) ---
    def _normalize_signals(items: list[str]) -> list[str]:
        out = []
        for s in items:
            s2 = re.sub(r"\s+", " ", (s or "").strip().lower())
            if s2 in {"muito obrigado", "obrigado!", "obrigado.", "obrigado,"}:
                s2 = "obrigado"
            out.append(s2)
        dedup = list(dict.fromkeys(out))
        if "obrigado" in dedup:
            dedup = ["obrigado"] + [x for x in dedup if x != "obrigado"]
        return dedup

    signals = _normalize_signals(signals)

    # (7) Mensagem muito curta e neutra (ex.: "feliz", "olá", "ok") -> Improdutivo
    # Evita cair no fail-safe Produtivo quando não há intenção de ação.
    neutral_short = (len(norm.split()) <= 2 and len(norm) <= 12)

    # já calculados acima / disponíveis aqui:
    # - has_gratitude
    # - has_action (via request verb ou info+pergunta)
    has_status_term = any(t in norm for t in {"status","prazo","andamento","update","eta","ticket"})
    has_marketing   = any(t in norm for t in MARKETING_TERMS)
    has_resolved    = any(t in norm for t in RESOLVED_TERMS)

    if neutral_short and not (has_action or has_status_term or has_gratitude or has_marketing or has_resolved):
        category = "Improdutivo"
        confidence = max(float(confidence or 0.0), 0.65)
        meta["neutral_short"] = True

    return category, round(float(confidence), 2), signals, meta


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
    
def _normalize_signals(signals: list[str]) -> list[str]:
    """Padroniza sinais (lower/strip), consolida variações e remove duplicatas preservando a ordem."""
    normed = []
    for s in signals:
        s2 = re.sub(r"\s+", " ", (s or "").strip().lower())
        # consolidação de sinônimos/variações
        if s2 in {"muito obrigado", "obrigado!", "obrigado.", "obrigado,"}:
            s2 = "obrigado"
        normed.append(s2)

    # remove duplicatas mantendo ordem
    deduped = list(dict.fromkeys(normed))
    # regra extra: se "obrigado" estiver presente, remova qualquer variação redundante (já mapeamos acima, mas fica de segurança)
    if "obrigado" in deduped:
        deduped = ["obrigado"] + [x for x in deduped if x != "obrigado"][1:]
    return deduped


# --- pipeline único chamado pela rota
def classify_email(content: str) -> tuple[str, float, list, dict]:
    """
    Retorna: category, confidence, signals, meta_info
    meta_info: {"used_hf": bool, "overrides": {...}}
    """
    text_clean = clean_text(content)
    norm = normalize(text_clean)

    hf_result = hf_zero_shot(text_clean)
    if hf_result:
        category, confidence = hf_result
        used_hf = True
    else:
        category, confidence, _ = rule_classifier(content)
        used_hf = False

    pos_hits, neg_hits, _ = detect_signals(norm)
    signals = list(dict.fromkeys(pos_hits + neg_hits))[:8]

    # filtro 'nf' ruído (pré) — também aplicado dentro do override para garantir
    import re
    def looks_like_nf(txt: str) -> bool:
        return ("nota fiscal" in txt) or bool(re.search(r"\bnf\b", txt))
    if "nf" in signals and not looks_like_nf(norm):
        signals = [s for s in signals if s != "nf"]

    # aplicar overrides consolidados
    category, confidence, signals, over_meta = apply_overrides(norm, category, confidence, signals)

    signals = _normalize_signals(signals)

    return category, round(float(confidence), 2), signals, {"used_hf": used_hf, "overrides": over_meta}


