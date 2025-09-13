from dotenv import load_dotenv
import os

load_dotenv()

HF_TOKEN = os.getenv("HF_API_TOKEN")
HF_MODEL = os.getenv("HF_ZEROSHOT_MODEL", "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TEMP = float(os.getenv("TEMPERATURE", "0.4"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS_REPLY", "220"))

# limites e flags opcionais
MAX_TEXT_CHARS = int(os.getenv("MAX_TEXT_CHARS", "50000"))  # 50k
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "5"))        # 5 MB
