"""FastAPI server of the demo (one process, serves the API and the single-page interface).

Run from the repository root:  ``python -m demo_app`` (or ``uvicorn demo_app.server:app --host 127.0.0.1 --port 8000``)
then open http://127.0.0.1:8000. Models load in a background thread at start-up; ``GNJA_DEMO_REPLAY_ONLY=1`` skips loading
(cached replay and the laboratory on saved cases still work offline).
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import uuid
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from api.dependencies import read_image
from demo_app.engine import DemoEngine, DemoError
from src.config.loader import load_config
from src.utils.paths import resolve

logger = logging.getLogger("demo")
STATIC = Path(__file__).parent / "static"
_CASE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


@lru_cache(maxsize=1)
def get_engine() -> DemoEngine:
    return DemoEngine(load_config("configs/demo.yaml"), load_config("configs/model.yaml"), load_config("configs/detector.yaml"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    eng = app.dependency_overrides.get(get_engine, get_engine)()
    if os.environ.get("GNJA_DEMO_REPLAY_ONLY", "0") == "1":
        eng.status = "replay_only"
    elif eng.status == "not_loaded":
        threading.Thread(target=eng.load, name="model-loader", daemon=True).start()
    yield


app = FastAPI(title="Grounded, Not Just Attending - Caption Reliability Inspector (demo)", version="1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.exception_handler(DemoError)
async def demo_error(_: Request, exc: DemoError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def unhandled(_: Request, exc: Exception) -> JSONResponse:
    rid = uuid.uuid4().hex[:8]
    logger.exception("Unhandled error [%s]", rid)
    return JSONResponse(status_code=500, content={"detail": f"Internal error (reference {rid}); see the server console."})


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


# ------------------------------------------------------------------------------------------------------------------ status
def _cases_dir() -> Path:
    return resolve(load_config("configs/demo.yaml").get("cases_dir", "demo_app/cases"))


def _list_cases():
    out = []
    for p in sorted(_cases_dir().glob("*.json")):
        try:
            c = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        out.append({"case_id": p.stem, "caption": c.get("caption"), "purpose": c.get("image", {}).get("purpose"),
                    "source": c.get("image", {}).get("source"), "license": c.get("image", {}).get("license"),
                    "created_utc": c.get("created_utc")})
    return out


@app.get("/api/health")
def health(eng: DemoEngine = Depends(get_engine)):
    return {**eng.info(), "cases_available": len(_list_cases())}


@app.get("/api/models")
def models(eng: DemoEngine = Depends(get_engine)):
    """B1 is the pretrained model the demo runs. B2 / A need trained checkpoints; their presence is reported, never faked."""
    app_cfg = load_config("configs/application.yaml")
    out = []
    for name, desc in (("B1", "Frozen pretrained BLIP"), ("B2", "LoRA fine-tune, caption loss only"), ("A", "LoRA + consistency objective (f_s)")):
        path = app_cfg.get("model_paths", {}).get(name)
        exists = path is None or resolve(path).is_file()
        out.append({"name": name, "description": desc, "checkpoint": path, "checkpoint_present": exists,
                    "used_by_demo": name == "B1", "status": ("loaded" if eng.ready else eng.status) if name == "B1" else
                    ("checkpoint present - compare with the full Inspector (api/ + frontend/)" if exists else "not trained yet: no checkpoint")})
    return out


@app.get("/api/evaluation/runs")
def evaluation_runs():
    runs = []
    for p in sorted(resolve("results").glob("eval_*/metric_results.json")):
        try:
            runs.append({"run_id": p.parent.name, "results": json.loads(p.read_text(encoding="utf-8"))})
        except (OSError, json.JSONDecodeError):
            runs.append({"run_id": p.parent.name, "error": "unreadable metric_results.json"})
    return {"runs": runs, "how_to_run": "python scripts/evaluate_all.py --config configs/evaluation.yaml --split dev "
                                        "(needs COCO and trained B2/A checkpoints; see README.md, Execution Workflow)"}


# ------------------------------------------------------------------------------------------------------------------ cases
@app.get("/api/demo/cases")
def cases():
    return {"cases": _list_cases(), "how_to_add": "put images in demo_app/images, list them in demo_app/images/manifest.json, "
                                                  "then run: python -m demo_app.prepare_cases"}


@app.get("/api/demo/cases/{case_id}")
def case(case_id: str, eng: DemoEngine = Depends(get_engine)):
    if not _CASE_ID.match(case_id):
        raise HTTPException(status_code=400, detail="invalid case id")
    p = _cases_dir() / f"{case_id}.json"
    if not p.is_file():
        raise HTTPException(status_code=404, detail="case not found")
    return eng.register_case(json.loads(p.read_text(encoding="utf-8"))).public


# ------------------------------------------------------------------------------------------------------------------ analysis
@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...), eng: DemoEngine = Depends(get_engine)):
    image = await read_image(file)
    from starlette.concurrency import run_in_threadpool

    st = await run_in_threadpool(eng.analyse, image, None, None, "live", {"source": "uploaded by the user"})
    return st.public


class CounterfactualRequest(BaseModel):
    analysis_id: str = Field(..., max_length=64)
    word_index: int = Field(..., ge=0, le=200)
    replacement: str = Field(..., min_length=1, max_length=30)


@app.post("/api/analyze/counterfactual")
def counterfactual(req: CounterfactualRequest, eng: DemoEngine = Depends(get_engine)):
    return eng.counterfactual(req.analysis_id, req.word_index, req.replacement)


class ObjectiveRequest(BaseModel):
    analysis_id: str = Field(..., max_length=64)
    word_index: int = Field(..., ge=0, le=200)
    steps: int = Field(100, ge=0, le=500)
    learning_rate: float = Field(1e-3, ge=1e-5, le=1.0)
    temperature: float = Field(0.07, ge=0.01, le=2.0)
    lambda_: float = Field(0.5, gt=0.0, le=10.0, alias="lambda")
    num_negatives: Optional[int] = Field(None, ge=1, le=20)


@app.post("/api/objective/preview")
def objective_preview(req: ObjectiveRequest, eng: DemoEngine = Depends(get_engine)):
    return eng.preview(req.analysis_id, req.word_index, req.temperature, req.num_negatives)


@app.post("/api/objective/optimize")
def objective_optimize(req: ObjectiveRequest, eng: DemoEngine = Depends(get_engine)):
    return eng.optimise(req.analysis_id, req.word_index, req.steps, req.learning_rate, req.temperature, req.lambda_, req.num_negatives)
