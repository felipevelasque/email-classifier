# app/main.py
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from app.routers.analyze import router as analyze_router

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "static"

app = FastAPI(title="Email Auto Classifier")
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
app.include_router(analyze_router)
