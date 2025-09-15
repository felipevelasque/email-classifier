## Email Auto Classifier

Aplicação web que classifica e-mails automaticamente em **Produtivo** ou **Improdutivo**, sugerindo respostas automáticas baseadas no conteúdo.  
O projeto combina **NLP + regras heurísticas + LLMs (OpenAI / Hugging Face)** para oferecer alta precisão e explicabilidade.

---

## Objetivo do Projeto

- **Classificar** e-mails recebidos em duas categorias:
- **Produtivo**: requer ação/resposta específica (ex.: erro no sistema, solicitação de prazo, status de chamado).
- **Improdutivo**: não requer ação (ex.: agradecimentos, felicitações, newsletters).
- **Gerar respostas automáticas** contextualizadas, de forma natural, utilizando **LLMs** com fallback para **templates prontos**.
- **Exibir resultados na interface** com categoria, sinais detectados, nível de confiança e resposta sugerida.

---

## Funcionalidades

- Upload de arquivos `.txt` ou `.pdf` (drag & drop ou seleção manual)  
- Campo de texto para colar conteúdo de e-mail  
- Classificação híbrida (Regras + Hugging Face + OpenAI)  
- Resposta automática gerada pela IA  
- Visualização de **categoria (badge)**, **confiança (progress bar)** e **sinais detectados (chips)**  
- Overrides inteligentes:
    - Agradecimento sem pedido → Improdutivo  
    - Marketing/newsletter → Improdutivo  
    - Resolvido/cancelado → Improdutivo  
    - Urgência → Produtivo com boost  
    - Perguntas curtas de status/prazo → Produtivo mesmo sem `?`
- Detecção automática de idioma do e-mail (PT / EN / ES) com geração de resposta no idioma detectado  
- Analisar resposta (com atalho `Ctrl+Enter` / `⌘+Enter`)  
- Tratamento de erros com mensagens claras:  
    - Arquivo vazio → `400 Bad Request`  
    - Arquivo muito grande (>2MB) → `413 Arquivo muito grande`  

---

## Tecnologias

**Backend**: Python 3.11 + [FastAPI]
**Frontend**: HTML, CSS, Javascript
**NLP/AI**:
  - Hugging Face Zero-Shot Classification
  - OpenAI GPT para respostas naturais
  - Regras heurísticas customizadas (palavras-chave e contexto)  
**Outros**:
  - pdfminer.six (leitura de PDFs)
  - Unidecode (normalização de texto)
  - Python JSON Logger (logs estruturados)
  - Uvicorn (servidor ASGI)

---

## Como rodar localmente

1. **Clone o repositório**
```bash 
git clone https://github.com/seu-usuario/email-auto-classifier.git
cd email-auto-classifier 
```

2. Crie e ative um ambiente virtual
```bash 
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows
```

3. Instale as dependências
```bash 
pip3 install -r requirements.txt
```

4. Configure variáveis de ambiente no arquivo .env (raiz do projeto):
```
OPENAI_KEY=sk-...
HF_TOKEN=hf_...
HF_MODEL=facebook/bart-large-mnli
```

5. Execute o servidor
```bash 
uvicorn app.main:app --reload
```

6. Acesse no navegador
```bash 
http://localhost:8000
```

---

Estrutura do projeto
```
email-classifier/
├── app
│   ├── core
│   │   ├── logging.py
│   │   └── settings.py
│   ├── main.py
│   ├── routers
│   │   └── analyze.py
│   ├── schemas.py
│   └── services
│       ├── classifier.py
│       └── replier.py
├── examples
│   ├── bigfile.txt
│   ├── empty.txt
│   ├── status.txt
│   └── thanks.txt
├── requirements.txt
├── README.md
├── static
│   ├── app.js
│   └── style.css
└── templates
    └── index.html
```
---

Testando via curl
recomendado instalar jq (se ainda não tiver)
```bash 
brew install jq #macOS (Homebrew)
sudo apt-get install jq -y  #Ubuntu/Debian
```


Casos normais:
Pergunta de status -> Produtivo HTTP 200 OK
```bash  
curl -s -X POST -F "email_file=@examples/status.txt" http://localhost:8000/api/analyze | jq . 
```

Agradecimento -> Improdutivo HTTP 200 OK
```bash 
curl -s -X POST -F "email_file=@examples/thanks.txt" http://localhost:8000/api/analyze | jq . 
```

Casos de erro:
Arquivo vazio -> HTTP 400 Bad Request
```bash 
curl -s -X POST -F "email_file=@examples/empty.txt" http://localhost:8000/api/analyze | jq .
```
Arquivo muito grande -> HTTP 413 Payload Too Large
```bash 
curl -s -X POST -F "email_file=@examples/bigfile.txt" http://localhost:8000/api/analyze | jq .
```
Requisição inválida (campo errado) → HTTP 422 Unprocessable Entity
```bash 
curl -s -X POST -F "email_file=so-um-texto" http://localhost:8000/api/analyze | jq .
```

Observação para Windows.
No PowerShell, use:
```bash 
curl -s -X POST -F "email_file=@examples/status.txt;type=text/plain" http://localhost:8000/api/analyze | jq .
```