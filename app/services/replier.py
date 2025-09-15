from typing import List
from app.core.settings import OPENAI_KEY, OPENAI_MODEL, TEMP, MAX_TOKENS
import time


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
            "Obrigado pela mensagem! Que bom receber seu retorno. "
            "No momento, não é necessária nenhuma ação da nossa equipe. "
            "Se surgir qualquer coisa, estamos por aqui. 😊"
        )

def ai_reply(category: str, snippet: str, signals: list[str], temperature: float | None = None) -> str | None:
    if not OPENAI_KEY:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_KEY)

        # tom mais humano e curto; regras diferentes por classe
        sys_msg = (
            "Você é um assistente de atendimento ao cliente em português do Brasil. "
            "Escreva como uma pessoa: natural, claro e cordial, sem jargões e sem soar automático. "
            "Use 3 a 6 linhas. Evite repetir a mesma ideia. Não invente números, ID ou prazos exatos que não foram dados. "
            "Política: "
            "- Se o email for PRODUTIVO: confirme recebimento, peça somente o essencial (ID do chamado, prints, datas) "
            "  e prometa APENAS a primeira atualização (até 1 dia útil). "
            "- Se o email for IMPRODUTIVO: agradeça, reconheça o contexto e encerre gentilmente sem pedir ação adicional."
        )
        user_msg = f"Categoria: {category}\nSinais: {signals}\nTrecho: {snippet[:800]}"

        retries = 3
        backoff = 2
        for attempt in range(1, retries + 1):
            try:
                resp = client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[{"role":"system","content":sys_msg},
                            {"role":"user","content":user_msg}],
                    temperature=TEMP,
                    max_tokens=MAX_TOKENS,
                    timeout=15
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                if attempt == retries:
                    print(f"[OpenAI] Falha após {retries} tentativas: {e}")
                    return None
                wait = backoff ** attempt
                print(f"[OpenAI] Tentativa {attempt} falhou, aguardando {wait}s...")
                time.sleep(wait)


        # pequena dica de estilo específica por classe
        if category == "Improdutivo":
            style_hint = "Tom leve e simpático; personalize 1 detalhe do contexto e finalize de forma positiva."
            temp = temperature if temperature is not None else 0.6  # um pouco mais criativo
        else:
            style_hint = "Tom objetivo e profissional; peça só o mínimo necessário."
            temp = temperature if temperature is not None else 0.4

        user_msg = (
            f"Categoria: {category}\n"
            f"Sinais: {', '.join(signals) if signals else 'nenhum'}\n"
            f"Contexto do email (limpo, até 900 chars):\n{snippet[:900]}\n\n"
            f"Estilo: {style_hint}"
        )

        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role":"system","content":sys_msg},
                      {"role":"user","content":user_msg}],
            temperature=temp,
            max_tokens=MAX_TOKENS,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return None
