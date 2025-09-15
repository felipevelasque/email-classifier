# app/services/replier.py
from typing import List
from app.core.settings import OPENAI_KEY, OPENAI_MODEL, TEMP, MAX_TOKENS
import time

# --- Prompts do sistema por idioma ---
SYS_PROMPTS = {
    "pt": (
        "Você é um assistente de atendimento ao cliente em português do Brasil. "
        "Responda com tom profissional, cordial e objetivo em 3–6 linhas. "
        "Se for PRODUTIVO: confirme recebimento, peça apenas o essencial (ID do chamado/prints/datas) "
        "e prometa somente a primeira atualização (até 1 dia útil). "
        "Se for IMPRODUTIVO: agradeça e informe que não há ação. "
        "Nunca invente dados ou prazos. Responda **em português**."
    ),
    "en": (
        "You are a customer support assistant. "
        "Reply professionally, warmly, and concisely in 3–6 lines. "
        "If PRODUCTIVE: acknowledge receipt, ask only for essentials (ticket ID/screenshots/dates), "
        "and promise only the first update (within 1 business day). "
        "If UNPRODUCTIVE: thank them and note no action is required. "
        "Do not fabricate details or deadlines. Reply **in English**."
    ),
    "es": (
        "Eres un asistente de soporte al cliente. "
        "Responde de manera profesional, cordial y concisa en 3–6 líneas. "
        "Si es PRODUCTIVO: confirma recepción, solicita solo lo esencial (ID del ticket/capturas/fechas) "
        "y promete solo la primera actualización (en 1 día hábil). "
        "Si es IMPRODUCTIVO: agradece e indica que no se requiere acción. "
        "No inventes datos ni plazos. Responde **en español**."
    ),
}

def _sys_prompt(lang: str) -> str:
    return SYS_PROMPTS.get(lang, SYS_PROMPTS["pt"])

# --- Templates localizados (fallback quando a LLM não responde) ---
def reply_template(category: str, signals: List[str], lang: str = "pt") -> str:
    # considera variações de “anexo”
    has_attach = any(s in signals for s in (
        "anexo", "arquivo", "attachment", "attached", "adjunto"
    ))

    if lang == "en":
        if category == "Produtivo":
            extra = "" if has_attach else " If possible, please attach screenshots/files."
            return (
                "Hi! We’ve received your message and will proceed. "
                "To speed things up, could you confirm the **ticket ID** (or client data) and the **date/time** of the issue?"
                f"{extra} Our first update is due **within 1 business day**."
            )
        else:
            return (
                "Hi! Thanks for your message. At the moment **no action is required** from our team. "
                "If you need anything else, just let us know."
            )

    if lang == "es":
        if category == "Produtivo":
            extra = "" if has_attach else " Si es posible, adjunta capturas/archivos."
            return (
                "¡Hola! Recibimos tu mensaje y daremos seguimiento. "
                "Para agilizar, ¿podrías confirmar el **ID del ticket** (o datos del cliente) y la **fecha/hora** del incidente?"
                f"{extra} Nuestra primera actualización será **dentro de 1 día hábil**."
            )
        else:
            return (
                "¡Hola! Gracias por tu mensaje. Por el momento **no se requiere ninguna acción** de nuestro equipo. "
                "Si necesitas algo más, avísanos."
            )

    # default: pt-BR
    if category == "Produtivo":
        extra = " Se possível, anexe prints/arquivos." if not has_attach else ""
        return (
            "Olá, tudo bem? Recebemos sua mensagem e vamos dar andamento. "
            "Para agilizar, poderia confirmar o **ID do chamado** (ou dados do cliente) e a **data/horário** do ocorrido?"
            f"{extra} Nossa previsão para a primeira atualização é de **até 1 dia útil**."
        )
    else:
        return (
            "Olá! Obrigado pela mensagem. No momento **não é necessária nenhuma ação** da nossa equipe. "
            "Se precisar de algo, é só nos chamar."
        )

# --- Geração com OpenAI (responde no idioma detectado) ---
def ai_reply(
    category: str,
    snippet: str,
    signals: list[str],
    lang: str = "pt",
    temperature: float | None = None
) -> str | None:
    if not OPENAI_KEY:
        return None

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_KEY)

        # ajuste leve de temperatura por classe (opcional)
        if temperature is not None:
            temp = float(temperature)
        else:
            temp = float(TEMP)
            if category == "Improdutivo":
                temp = max(temp, 0.55)  # um pouco mais “humano” em agradecimentos

        sys_msg = _sys_prompt(lang)
        user_msg = (
            f"Category: {category}\n"
            f"Signals: {', '.join(signals) if signals else 'none'}\n"
            f"Email snippet (clean, up to 900 chars): {snippet[:900]}"
        )

        # retries simples com backoff exponencial
        retries = 3
        backoff = 2
        for attempt in range(1, retries + 1):
            try:
                resp = client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=temp,
                    max_tokens=MAX_TOKENS,
                    timeout=15,
                )
                content = (resp.choices[0].message.content or "").strip()
                if content:
                    return content
            except Exception as e:
                if attempt == retries:
                    print(f"[OpenAI] Falha após {retries} tentativas: {e}")
                    return None
                wait = backoff ** attempt
                print(f"[OpenAI] Tentativa {attempt} falhou, aguardando {wait}s...")
                time.sleep(wait)

        return None
    except Exception:
        return None
