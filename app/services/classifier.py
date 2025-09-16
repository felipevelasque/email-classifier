# app/services/classifier.py
from typing import List, Tuple, Dict, Iterable, Pattern
from fastapi import UploadFile, HTTPException
import io, re, requests, time
from pdfminer.high_level import extract_text as pdf_extract_text
from langdetect import detect_langs, DetectorFactory
from app.core.settings import HF_TOKEN, HF_MODEL

DetectorFactory.seed = 0

SUPPORTED = {"pt", "en", "es"}

# ============================================================================
# Helpers de regex padronizados
# ============================================================================

FLAGS = re.IGNORECASE

def normalize(text: str) -> str:
    try:
        from unidecode import unidecode
        return unidecode(text.lower().strip())
    except Exception:
        return text.lower()

def _literal_to_regex(term: str) -> Pattern:
    """
    Converte um termo literal para regex robusto:
    - Escapa metacaracteres do termo
    - Substitui espaços por \\s+ para tolerar múltiplos espaços/pontuações
    - Adiciona bordas de palavra \\b...\\b
    """
    t = normalize(term)
    escaped = re.escape(t)
    escaped = re.sub(r"\s+", r"\\s+", escaped)
    pat = rf"\b{escaped}\b"
    return re.compile(pat, FLAGS)

def _compile_patterns(terms: Iterable[str]) -> List[Pattern]:
    return [_literal_to_regex(t) for t in terms]

def _compile_map_weights(weight_map: Dict[str, float]) -> Dict[str, Pattern]:
    """Mantém as chaves originais (para exibir nos chips), valor vira regex compilada."""
    return {k: _literal_to_regex(k) for k in weight_map.keys()}

def any_match(patterns: Iterable[Pattern], text: str) -> bool:
    return any(rx.search(text) for rx in patterns)

# ============================================================================
# Detecção de idioma (PT / EN / ES) com heurísticas
# ============================================================================

_EN_WHITELIST = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening", "thanks", "thank you"}
_ES_WHITELIST = {"hola", "buenas", "buenos dias", "buenas tardes", "gracias"}
_PT_WHITELIST = {"oi", "ola", "bom dia", "boa tarde", "boa noite", "obrigado", "obrigada"}

def detect_language(text: str, default: str = "pt") -> str:
    """
    Heurísticas diretas + langdetect como fallback.
    - Viés para PT quando há 'ola' (sem 'h') ou pistas típicas ('está', 'não', 'funcionando', etc.).
    - Suporta textos curtos e médios.
    """
    t = (text or "").strip()
    if not t:
        return default

    norm = normalize(t)

    # Heurísticas diretas PT/ES
    if "hola" in norm:
        return "es"
    if re.search(r"\bola\b", norm) and "hola" not in norm:
        return "pt"

    # Pistas explícitas de PT (sem acento por causa de normalize)
    PT_HINTS = ["esta", "nao", "funcionando", "obrigado", "prazo", "voces", "atualizacao"]
    if any(h in norm for h in PT_HINTS):
        return "pt"

    # Whitelists
    if any(kw in norm for kw in _PT_WHITELIST): return "pt"
    if any(kw in norm for kw in _ES_WHITELIST): return "es"
    if any(kw in norm for kw in _EN_WHITELIST): return "en"

    # Fallback: langdetect
    try:
        langs = detect_langs(t)
        top = max(langs, key=lambda x: x.prob)
        lang, prob = top.lang, float(top.prob)

        # Se langdetect apontar ES mas houver indícios fortes de PT, favorece PT
        if lang == "es":
            if re.search(r"\bola\b", norm) and "hola" not in norm:
                return "pt"
            if any(kw in norm for kw in _PT_WHITELIST):
                return "pt"

        # Baixa confiança + frase muito curta → viés a EN se ascii
        if prob < 0.75 and len(t) <= 25 and t.isascii() and any(c.isalpha() for c in t):
            if any(kw in norm for kw in _EN_WHITELIST):
                return "en"
            return "en"

        return lang if lang in SUPPORTED else default
    except Exception:
        if "hola" in norm: return "es"
        if re.search(r"\bola\b", norm): return "pt"
        if t.isascii() and any(c.isalpha() for c in t):
            if any(kw in norm for kw in _EN_WHITELIST): return "en"
        return default

# ============================================================================
# Limpeza e leitura de arquivos
# ============================================================================

HTML_TAG_RE = re.compile(r"<[^>]+>")
SIGN_RE = re.compile(r"(?is)(atenciosamente,.*$|kind regards,.*$|--+\s*\n.*$)")

def clean_text(text: str) -> str:
    text = text or ""
    text = HTML_TAG_RE.sub(" ", text)
    text = SIGN_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()

def read_txt_pdf(file: UploadFile) -> str:
    name = (file.filename or "").lower()
    blob = file.file.read()

    if name.endswith(".txt"):
        text = blob.decode(errors="ignore")
        if not text.strip():
            raise HTTPException(422, detail="TXT vazio ou ilegível.")
        return text

    if name.endswith(".pdf"):
        try:
            with io.BytesIO(blob) as f:
                text = pdf_extract_text(f)
            if not text.strip():
                raise HTTPException(422, detail="PDF sem texto (escaneado/sem OCR) ou vazio.")
            return text
        except Exception:
            raise HTTPException(415, detail="PDF não suportado (envie PDF pesquisável).")

    raise HTTPException(415, detail="Formato não suportado. Use .txt ou .pdf.")

# ============================================================================
# Vocabulários → regex
# ============================================================================

POS_SIGNALS = {
    "anexo": 1.4, "arquivo": 1.2, "solicitacao": 1.3, "pedido": 1.0, "status": 1.4,
    "andamento": 1.1, "atualizacao": 1.1, "erro": 1.5, "sistema": 1.0,
    "prazo": 1.2, "urgente": 1.4, "nota fiscal": 1.2, "nf": 1.1, "contrato": 1.2,
    "chamado": 1.3, "protocolo": 1.0, "boleto": 1.0, "fatura": 1.0,
}
NEG_SIGNALS = {
    "obrigado": 2.2, "muito obrigado": 2.5, "agradeco": 2.2, "agradecimento": 2.5,
    "feliz natal": 2.5, "feliz ano": 2.2, "ano novo": 2.0, "parabens": 2.0,
    "newsletter": 1.6, "divulgacao": 1.5, "marketing": 1.5, "convite": 1.4,
    "bom dia": 0.6, "boa tarde": 0.6, "boa noite": 0.6,
}

RX_POS = _compile_map_weights(POS_SIGNALS)
RX_NEG = _compile_map_weights(NEG_SIGNALS)

REQUEST_TERMS_RX = _compile_patterns([
    # pt
    "verificar","poderiam verificar","podem verificar","informar","enviar","mandar",
    "emitir","atualizar","abrir","analisar","corrigir","resolver","processar","gerar",
    # en
    "check","could you","can you","please send","share","provide","issue","update","open","fix","resolve","process","generate",
    # es
    "verificar","podrian","pueden","enviar","mandar","emitir","actualizar","abrir","analizar","corregir","resolver","procesar","generar",
])
INFO_TERMS_RX = _compile_patterns(["status","prazo","andamento","atualizacao","update","eta","estado","plazo"])

ACTION_HINTS_RX = _compile_patterns([
    "status","andamento","prazo","erro","atualizacao","protocolo","chamado","contrato",
    "boleto","fatura","nota fiscal","nf","anexo","arquivo",
    "verificar","verifiquem","poderiam verificar","podem verificar","processado",
    "emitir","emissao","resolucao","correcao","resolver","eta",
])
GRATITUDE_HINTS_RX = _compile_patterns([
    "obrigado","muito obrigado","agradeco","agradecimento","feliz natal","feliz ano","ano novo","parabens",
])
FUNCTIONING_PHRASES_RX = _compile_patterns([
    "tudo funcionando","funcionando perfeitamente","problema resolvido","issue resolvida","resolvido",
])
WELL_WISHES_TERMS_RX = _compile_patterns([
    "espero que estejam bem","espero que esteja bem","otima semana","boa semana",
    "boa jornada","bom trabalho","tenha um bom dia","tenha uma boa semana",
    "desejo uma otima semana",
])
GREETING_TERMS_RX = _compile_patterns([
    "ola","oi","bom dia","boa tarde","boa noite","tudo bem","como vai","como esta",
])

MARKETING_TERMS_RX = _compile_patterns([
    "newsletter","divulgacao","marketing","convite","evento","webinar","lancamento","release","oferta","promocao",
])
GRATITUDE_TERMS_RX = _compile_patterns([
    "obrigado","muito obrigado","agradeco","agradecimento","feliz natal","feliz ano","ano novo","parabens",
    "gracias","thank you","thanks",
])
RESOLVED_TERMS_RX = _compile_patterns([
    "tudo funcionando","funcionando perfeitamente","problema resolvido","issue resolvida","resolvido",
    "nao preciso","pode desconsiderar","pode cancelar","cancelar solicitacao","cancelada","cancelado",
])
URGENCY_TERMS_RX = _compile_patterns([
    "urgente","urgencia","asap","o mais rapido possivel","priority","prioridade",
])

# Padrões de erro/issue/acesso (já como regex "livre")
ERROR_PATTERNS_RX = [
    re.compile(p, FLAGS) for p in [
        r"\b(erro|error|bug|falha|falhou|trava|travou|crash)\b",
        r"\b(problema|issue|incidente)\b",
        r"\b(acessar|acesso|login|logar|autenticacao|senha|usuario)\b",
        r"\bnao\s+funciona\b",
        r"\bnao\s+esta\s+funcionando\b",
        r"\bno\s+funciona\b",
        r"\bnot\s+working\b",
        r"\bfora\s+do\s+ar\b",
    ]
]

def _has_issue(text_norm: str) -> bool:
    return any(rx.search(text_norm) for rx in ERROR_PATTERNS_RX)

# ============================================================================
# Sinais (agora com regex)
# ============================================================================

def detect_signals(text_norm: str) -> Tuple[List[str], List[str], float]:
    pos_hits, neg_hits = [], []
    score = 0.0
    for key, rx in RX_POS.items():
        if rx.search(text_norm):
            pos_hits.append(key); score += POS_SIGNALS[key]
    for key, rx in RX_NEG.items():
        if rx.search(text_norm):
            neg_hits.append(key); score -= NEG_SIGNALS[key]
    return pos_hits, neg_hits, score

# ============================================================================
# Classificador por regras
# ============================================================================

def _has_any_rx(text_norm: str, patterns: Iterable[Pattern]) -> bool:
    return any(rx.search(text_norm) for rx in patterns)

def rule_classifier(text: str) -> Tuple[str, float, List[str]]:
    text_clean = clean_text(text)
    text_norm = normalize(text_clean)

    pos_hits, neg_hits, base_score = detect_signals(text_norm)

    pos_bonus = 0.0
    neg_bonus = 0.0
    if _has_any_rx(text_norm, ACTION_HINTS_RX): pos_bonus += 0.6
    if _has_any_rx(text_norm, FUNCTIONING_PHRASES_RX): neg_bonus += 0.8

    score = base_score + pos_bonus - neg_bonus

    if score > 0.6: category = "Produtivo"
    elif score < -0.6: category = "Improdutivo"
    else: category = "Produtivo"

    if _has_any_rx(text_norm, GRATITUDE_HINTS_RX) and not _has_any_rx(text_norm, ACTION_HINTS_RX):
        category = "Improdutivo"; conf_val = 0.80
    else:
        raw = abs(score)
        conf_val = 0.55 + min(raw / 6.0, 0.35)

    signals = list(dict.fromkeys(pos_hits + neg_hits))[:8]
    return category, round(conf_val, 2), signals

# ============================================================================
# Overrides finais (regex everywhere)
# ============================================================================

def apply_overrides(norm: str, category: str, confidence: float, signals: list[str]) -> tuple[str, float, list[str], dict]:
    meta = {
        "gratitude_no_action": False,
        "acao_baixa_conf": False,
        "marketing_newsletter": False,
        "resolved_or_cancelled": False,
        "urgency_boost": False,
        "short_question_hint": False,
        "neutral-short": False,
        "noise_filter": [],
    }

    # 'nf' só vale se "nota fiscal" ou token isolado 'nf'
    def _looks_like_nf(txt: str) -> bool:
        return bool(_literal_to_regex("nota fiscal").search(txt) or re.search(r"\bnf\b", txt))
    if "nf" in signals and not _looks_like_nf(norm):
        signals = [s for s in signals if s != "nf"]
        meta["noise_filter"].append("nf")

    # intenção de ação
    has_request_verb = any_match(REQUEST_TERMS_RX, norm)
    has_info_term    = any_match(INFO_TERMS_RX, norm)
    has_question     = "?" in norm
    has_issue        = _has_issue(norm)

    has_action = has_issue or has_request_verb or (has_info_term and has_question)
    if has_issue:
        meta["issue_detected"] = True

    # (1) Gratidão/felicitações sem pedido -> Improdutivo
    has_gratitude = any_match(GRATITUDE_TERMS_RX, norm)
    if has_gratitude and not has_action:
        category = "Improdutivo"
        confidence = max(float(confidence or 0.0), 0.80)
        meta["gratitude_no_action"] = True
        if not any((s or "").strip().lower().startswith("obrigado") for s in signals):
            signals = ["obrigado"] + signals

    # (1.1) Saudação/well-wishes puro (curto) -> Improdutivo
    has_greeting = any_match(GREETING_TERMS_RX, norm)
    has_well_wishes = any_match(WELL_WISHES_TERMS_RX, norm)
    token_count = len(norm.split())

    if (has_greeting or has_well_wishes) and not has_action and not has_question and not has_issue:
        if token_count <= 6 and len(norm) <= 40:
            category = "Improdutivo"
            confidence = max(float(confidence or 0.0), 0.80)
            meta["greeting_only"] = True
            if "saudacao" not in signals:
                signals = ["saudacao"] + signals

    # (2) Marketing/newsletter/convite sem pedido -> Improdutivo
    if any_match(MARKETING_TERMS_RX, norm) and not has_action:
        if category != "Improdutivo":
            category = "Improdutivo"
            confidence = max(float(confidence or 0.0), 0.75)
        meta["marketing_newsletter"] = True

    # (3) Resolvido/cancelado -> sempre Improdutivo
    if any_match(RESOLVED_TERMS_RX, norm):
        category = "Improdutivo"
        confidence = max(float(confidence or 0.0), 0.85)
        meta["resolved_or_cancelled"] = True

    # (4) Ação detectada mas modelo veio Improdutivo com baixa confiança → força Produtivo
    if has_action and category == "Improdutivo" and float(confidence or 0.0) < 0.80:
        category = "Produtivo"
        confidence = max(float(confidence or 0.0), 0.75)
        meta["action_over_low_conf"] = True

    # (5) Urgência -> boost em Produtivo
    if any_match(URGENCY_TERMS_RX, norm) and category == "Produtivo":
        confidence = max(float(confidence or 0.0), 0.78)
        meta["urgency_boost"] = True
        if "urgente" in norm and "urgente" not in signals:
            signals = ["urgente"] + signals

    # (6) Pergunta/solicitação curta sobre status/prazo
    status_terms_rx = _compile_patterns(["status","prazo","andamento","update","eta","ticket"])
    short_len = len(norm) <= 40
    short_tokens = len(norm.split()) <= 6
    looks_like_question = (
        "?" in norm
        or norm.strip() in {"status","qual o status","e o status","como esta o status","status do chamado","status do ticket","e o prazo","qual o prazo"}
        or any(rx.search(norm.strip()) for rx in status_terms_rx)
        or any(re.search(rf"{rx.pattern}\s+por\s+favor", norm) for rx in status_terms_rx)
        or bool(re.search(r"\b(qual|sobre|e\s*o|e\s*quanto)\b", norm))
    )
    has_status_term = any_match(status_terms_rx, norm)

    if (short_len or short_tokens) and has_status_term and looks_like_question:
        category = "Produtivo"
        confidence = max(float(confidence or 0.0), 0.70)
        meta["short_question_hint"] = True

    # normalização de sinais
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

    # (7) Muito curta & neutra -> Improdutivo
    neutral_short = (len(norm.split()) <= 2 and len(norm) <= 12)
    has_marketing = any_match(MARKETING_TERMS_RX, norm)
    has_resolved  = any_match(RESOLVED_TERMS_RX, norm)

    if neutral_short and not (has_action or has_status_term or has_gratitude or has_marketing or has_resolved):
        category = "Improdutivo"
        confidence = max(float(confidence or 0.0), 0.65)
        meta["neutral_short"] = True

    # Piso de confiança para issue produtiva
    if meta.get("problema_detectado") and category == "Produtivo":
        confidence = max(float(confidence or 0.0), 0.80)

    return category, round(float(confidence), 2), signals, meta

# ============================================================================
# HF zero-shot
# ============================================================================

def hf_zero_shot(text: str) -> tuple[str, float] | None:
    if not HF_TOKEN:
        return None
    url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": text,
        "parameters": {
            "candidate_labels": ["Produtivo", "Improdutivo"],
            "hypothesis_template": "Este email requer uma ação imediata da equipe: {}."
        }
    }
    retries, backoff = 3, 2
    for attempt in range(1, retries + 1):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=10)
            r.raise_for_status()
            data = r.json()
            labels = data.get("labels") or []
            scores = data.get("scores") or []
            if not labels or not scores:
                return None
            return labels[0], float(scores[0])
        except Exception as e:
            if attempt == retries:
                print(f"[HF] Falha após {retries} tentativas: {e}")
                return None
            time.sleep(backoff ** attempt)

# ============================================================================
# Pipeline principal
# ============================================================================

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

    # filtro 'nf' ruído (pré)
    if "nf" in signals and not (_literal_to_regex("nota fiscal").search(norm) or re.search(r"\bnf\b", norm)):
        signals = [s for s in signals if s != "nf"]

    # overrides finais
    category, confidence, signals, over_meta = apply_overrides(norm, category, confidence, signals)

    # normaliza sinais final
    def _normalize_signals_final(items: list[str]) -> list[str]:
        normed = []
        for s in items:
            s2 = re.sub(r"\s+", " ", (s or "").strip().lower())
            if s2 in {"muito obrigado","obrigado!","obrigado.","obrigado,"}:
                s2 = "obrigado"
            normed.append(s2)
        dedup = list(dict.fromkeys(normed))
        if "obrigado" in dedup:
            dedup = ["obrigado"] + [x for x in dedup if x != "obrigado"][1:]
        return dedup

    signals = _normalize_signals_final(signals)

    return category, round(float(confidence), 2), signals, {"used_hf": used_hf, "overrides": over_meta}
