"""FastAPI application: Caption Reliability Inspector backend.

Run: ``uvicorn api.main:app --host 0.0.0.0 --port 8000`` (or ``bash scripts/launch_api.sh``). Importing this module loads no model.
"""
import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.dependencies import get_app_config
from api.routes import caption, comparison, examples, health, models
from src.utils.logging import setup_logging

setup_logging()
logger = logging.getLogger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start-up/shutdown hooks. Models are lazy: start-up only reads config; shutdown drops loaded models."""
    cfg = get_app_config()
    logger.info("API starting (demo_mode=%s)", cfg.get("demo_mode", False))
    yield
    from api.dependencies import get_registry

    reg = get_registry()
    reg.predictors.clear()
    logger.info("API stopped; models released")


app = FastAPI(title="Caption Reliability Inspector", version="0.2.0", lifespan=lifespan,
              description="Captions from B1/B2/A with attention overlays and object-level evidence flags (heuristic, not probabilities).")
# CORS: restrictive by default (explicit origins from configs/application.yaml; no wildcard unless configured on purpose).
app.add_middleware(CORSMiddleware, allow_origins=get_app_config().get("cors_origins", []), allow_methods=["GET", "POST"], allow_headers=["*"])
for r in (caption.router, comparison.router, health.router, models.router, examples.router):
    app.include_router(r)


@app.exception_handler(Exception)
async def unhandled(_: Request, exc: Exception) -> JSONResponse:
    """JSON error without the exception text or any local path; the id correlates with the server log."""
    rid = uuid.uuid4().hex[:8]
    logger.exception("Unhandled error [%s]", rid)
    return JSONResponse(status_code=500, content={"detail": f"Internal error (reference {rid})"})
