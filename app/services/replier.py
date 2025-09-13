from typing import List
from app.core.settings import OPENAI_KEY, OPENAI_MODEL, TEMP, MAX_TOKENS

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

def ai_reply(category: str, snippet: str, signals: list[str]) -> str | None:
    if not OPENAI_KEY:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_KEY)
        sys_msg = (
            "Você é um assistente de atendimento ao cliente em português (Brasil). "
            "Responda com tom profissional, cordial e objetivo, em até 6–8 linhas. "
            "Se for PRODUTIVO: confirme recebimento, peça dados faltantes (ID do chamado, prints, datas) "
            "e prometa apenas a primeira atualização (até 1 dia útil). "
            "Se for IMPRODUTIVO: agradeça e informe que não há ação. "
            "Nunca exponha dados sensíveis nem invente números/protocolos."
        )
        user_msg = (
            f"Categoria: {category}\n"
            f"Sinais: {', '.join(signals) if signals else 'nenhum'}\n"
            f"Trecho do email (limpo): {snippet[:900]}"
        )
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "system", "content": sys_msg},
                      {"role": "user", "content": user_msg}],
            temperature=TEMP,
            max_tokens=MAX_TOKENS,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return None
