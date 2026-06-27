"""FastAPI service for the Invoice & Receipt Processing system.

Endpoints: GET /health, POST /extract, POST /classify, POST /batch,
GET /batch/{job_id}, GET /review-queue, GET /metrics.
Models load once into a singleton agent; every response echoes model versions.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Dict, List

from ..logging_utils import get_logger
from .dependencies import get_config, get_invoice_agent
from .schemas import (BatchStatusResponse, BatchSubmitResponse, ClassifyResponse,
                      ExtractResponse, HealthResponse, ReviewItem, ReviewQueueResponse)

logger = get_logger(__name__)

_BATCH_JOBS: Dict[str, Dict] = {}


def _state_to_extract(sd: Dict, filename: str, ms: float) -> ExtractResponse:
    return ExtractResponse(
        filename=filename, doc_type=sd.get("doc_type", "unknown"), doc_type_conf=sd.get("doc_type_conf", 0.0),
        page_count=sd.get("page_count", 1), status=sd.get("status", "failed"),
        needs_review=sd.get("needs_review", False), review_reasons=sd.get("review_reasons", []),
        overall_confidence=sd.get("overall_confidence", 0.0), currency=sd.get("currency"),
        fields=sd.get("fields", {}), line_items=sd.get("line_items", []), validation=sd.get("validation"),
        model_versions=sd.get("model_versions", {}), trace=sd.get("trace", []), processing_ms=round(ms, 2))


def create_app():
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import PlainTextResponse

    cfg = get_config()
    app = FastAPI(title=cfg.serving.api_title, version=cfg.serving.api_version)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    try:
        from prometheus_client import Counter, Histogram, generate_latest
        REQ = Counter("invoice_requests_total", "Requests", ["endpoint"])
        REVIEW = Counter("invoice_needs_review_total", "Docs flagged for review")
        LAT = Histogram("invoice_latency_seconds", "Latency", ["endpoint"])
        _PROM = True
    except Exception:
        REQ = REVIEW = LAT = None
        _PROM = False

    @app.get("/health", response_model=HealthResponse)
    def health():
        agent = get_invoice_agent()
        return HealthResponse(status="ok", version=cfg.serving.api_version, model_version=cfg.serving.model_version,
                              models_loaded={"layout_extractor": agent.layout_extractor is not None,
                                             "llm_fallback": agent.llm.available()})

    @app.post("/extract", response_model=ExtractResponse)
    async def extract(file: UploadFile = File(...)):
        t0 = time.perf_counter()
        raw = await file.read()
        if len(raw) > cfg.serving.max_file_mb * 1024 * 1024:
            raise HTTPException(status_code=413, detail="FILE_TOO_LARGE")
        agent = get_invoice_agent()
        state = agent.process(raw_bytes=raw, filename=file.filename or "document")
        dt = time.perf_counter() - t0
        if _PROM:
            REQ.labels("extract").inc(); LAT.labels("extract").observe(dt)
            if state.to_dict().get("needs_review"):
                REVIEW.inc()
        return _state_to_extract(state.to_dict(), file.filename or "document", dt * 1000)

    @app.post("/classify", response_model=ClassifyResponse)
    async def classify(file: UploadFile = File(...)):
        agent = get_invoice_agent()
        state = agent.process(raw_bytes=await file.read(), filename=file.filename or "document")
        sd = state.to_dict()
        return ClassifyResponse(filename=file.filename or "document", doc_type=sd["doc_type"],
                                confidence=sd["doc_type_conf"], needs_review=sd.get("needs_review", False))

    def _run_batch(job_id: str, files: List[Dict]):
        agent = get_invoice_agent()
        out = []
        for f in files:
            state = agent.process(raw_bytes=f["bytes"], filename=f["name"])
            out.append(_state_to_extract(state.to_dict(), f["name"], 0).model_dump())
            _BATCH_JOBS[job_id]["done"] += 1
        _BATCH_JOBS[job_id]["results"] = out
        _BATCH_JOBS[job_id]["status"] = "done"

    @app.post("/batch", response_model=BatchSubmitResponse, status_code=202)
    async def batch(files: List[UploadFile] = File(...)):
        from starlette.background import BackgroundTask
        from fastapi.responses import JSONResponse
        payload = [{"name": f.filename or "document", "bytes": await f.read()} for f in files]
        job_id = uuid.uuid4().hex[:12]
        _BATCH_JOBS[job_id] = {"status": "running", "done": 0, "total": len(payload), "results": None}
        resp = JSONResponse(status_code=202,
                            content=BatchSubmitResponse(job_id=job_id, status="running", total=len(payload)).model_dump())
        resp.background = BackgroundTask(_run_batch, job_id, payload)
        return resp

    @app.get("/batch/{job_id}", response_model=BatchStatusResponse)
    def batch_status(job_id: str):
        from fastapi import HTTPException
        job = _BATCH_JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job_id")
        return BatchStatusResponse(job_id=job_id, status=job["status"], done=job["done"],
                                   total=job["total"], results=job["results"])

    @app.get("/review-queue", response_model=ReviewQueueResponse)
    def review_queue(limit: int = 50):
        path = Path(cfg.serving.review_queue_path)
        items: List[ReviewItem] = []
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines()[-limit:]:
                try:
                    rec = json.loads(line)
                    items.append(ReviewItem(filename=rec.get("filename", "?"),
                                            reasons=rec.get("reasons", []), fields=rec.get("fields", {})))
                except Exception:
                    continue
        return ReviewQueueResponse(items=items, count=len(items))

    @app.get("/metrics")
    def metrics():
        if not _PROM:
            return PlainTextResponse("prometheus_client not installed", status_code=501)
        from prometheus_client import generate_latest
        return PlainTextResponse(generate_latest().decode("utf-8"))

    return app


app = create_app()

__all__ = ["app", "create_app"]
