"""Pydantic response schemas for the Invoice AI REST API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ExtractResponse(BaseModel):
    filename: str
    doc_type: str
    doc_type_conf: float = 0.0
    page_count: int = 1
    status: str
    needs_review: bool = False
    review_reasons: List[str] = Field(default_factory=list)
    overall_confidence: float = 0.0
    currency: Optional[str] = None
    fields: Dict[str, Any] = Field(default_factory=dict)
    line_items: List[Dict[str, Any]] = Field(default_factory=list)
    validation: Optional[Dict[str, Any]] = None
    model_versions: Dict[str, str] = Field(default_factory=dict)
    trace: List[Dict[str, Any]] = Field(default_factory=list)
    processing_ms: float = 0.0


class ClassifyResponse(BaseModel):
    filename: str
    doc_type: str
    confidence: float = 0.0
    needs_review: bool = False


class BatchSubmitResponse(BaseModel):
    job_id: str
    status: str
    total: int


class BatchStatusResponse(BaseModel):
    job_id: str
    status: str
    done: int
    total: int
    results: Optional[List[Dict[str, Any]]] = None


class ReviewItem(BaseModel):
    filename: str
    reasons: List[str] = Field(default_factory=list)
    fields: Dict[str, Any] = Field(default_factory=dict)


class ReviewQueueResponse(BaseModel):
    items: List[ReviewItem] = Field(default_factory=list)
    count: int = 0


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    model_version: str = "v1"
    models_loaded: Dict[str, bool] = Field(default_factory=dict)


__all__ = ["ExtractResponse", "ClassifyResponse", "BatchSubmitResponse", "BatchStatusResponse",
           "ReviewItem", "ReviewQueueResponse", "HealthResponse"]
