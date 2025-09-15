import time
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.core.logging import setup_logger
from app.routers.analyze import router as analyze_router

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "static"
TEMPLATES = ROOT / "templates"

logger = setup_logger()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("app_startup")
    try:
        yield
    finally:
        logger.info("app_shutdown")

app = FastAPI(title="Email Auto Classifier", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

# Home (UI)
@app.get("/")
async def index():
    return FileResponse(str(TEMPLATES / "index.html"))

# API
app.include_router(analyze_router)

# (opcional) access log
@app.middleware("http")
async def access_log(request: Request, call_next):
    start = time.perf_counter()
    resp = await call_next(request)
    dur_ms = int((time.perf_counter() - start) * 1000)
    logger.info("request",
        extra={"method": request.method, "path": request.url.path,
               "status": resp.status_code, "duration_ms": dur_ms})
    return resp
