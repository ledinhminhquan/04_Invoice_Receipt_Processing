"""Agent state + value objects (the shared blackboard).

Every tool reads/writes one :class:`AgentState`, and the full ``trace`` is
returned to the caller as an audit log. Field values carry a **confidence** and a
**bbox** (provenance) so the review UI can highlight exactly where each value came
from — the transparency a human reviewer (and an auditor) needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class DocType(str, Enum):
    INVOICE = "invoice"
    RECEIPT = "receipt"
    OTHER = "other"
    UNKNOWN = "unknown"


class Status(str, Enum):
    NEW = "new"
    ROUTED = "routed"
    OCR_DONE = "ocr_done"
    EXTRACTED = "extracted"
    VALIDATED = "validated"
    NORMALIZED = "normalized"
    AUTO_APPROVED = "auto_approved"
    NEEDS_REVIEW = "needs_review"
    FAILED = "failed"


def _jsonable(v: Any) -> Any:
    if isinstance(v, Decimal):
        return float(v)
    return v


@dataclass
class FieldValue:
    value: Any
    confidence: float = 0.0
    source: str = "rule"                       # layoutlmv3 | donut | ocr | llm | rule
    bbox: Optional[Tuple[int, int, int, int]] = None
    page: int = 1
    raw: Optional[str] = None                  # original surface form before normalisation

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["value"] = _jsonable(self.value)
        return d


@dataclass
class LineItem:
    description: Optional[FieldValue] = None
    quantity: Optional[FieldValue] = None
    unit_price: Optional[FieldValue] = None
    amount: Optional[FieldValue] = None
    page: int = 1
    row: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {k: (v.to_dict() if isinstance(v, FieldValue) else v)
                for k, v in {"description": self.description, "quantity": self.quantity,
                             "unit_price": self.unit_price, "amount": self.amount,
                             "page": self.page, "row": self.row}.items()}


@dataclass
class ValidationReport:
    reconciles: bool = False
    reconcile_delta: float = 0.0
    checks: Dict[str, bool] = field(default_factory=dict)
    missing_required: List[str] = field(default_factory=list)
    low_confidence_fields: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ToolTrace:
    tool: str
    ok: bool
    latency_ms: float
    summary: str
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentState:
    doc_path: str = ""
    filename: str = "document"
    doc_type: DocType = DocType.UNKNOWN
    doc_type_conf: float = 0.0
    scan_quality: float = 1.0
    page_count: int = 1
    ocr_text: str = ""
    ocr_tokens: List[Dict[str, Any]] = field(default_factory=list)  # [{text,bbox,conf,page}]
    ocr_mean_conf: float = 0.0
    ocr_engine: str = "none"
    fields: Dict[str, FieldValue] = field(default_factory=dict)
    line_items: List[LineItem] = field(default_factory=list)
    validation: Optional[ValidationReport] = None
    normalized: Dict[str, Any] = field(default_factory=dict)
    currency: Optional[str] = None
    status: Status = Status.NEW
    overall_confidence: float = 0.0
    review_reasons: List[str] = field(default_factory=list)
    attempts: Dict[str, int] = field(default_factory=dict)
    used_llm_fallback: bool = False
    trace: List[ToolTrace] = field(default_factory=list)
    model_versions: Dict[str, str] = field(default_factory=dict)

    def add_trace(self, t: ToolTrace) -> None:
        self.trace.append(t)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filename": self.filename,
            "doc_type": self.doc_type.value if isinstance(self.doc_type, DocType) else self.doc_type,
            "doc_type_conf": round(self.doc_type_conf, 4),
            "scan_quality": round(self.scan_quality, 4),
            "page_count": self.page_count,
            "ocr_engine": self.ocr_engine,
            "ocr_mean_conf": round(self.ocr_mean_conf, 4),
            "currency": self.currency,
            "fields": {k: v.to_dict() for k, v in self.fields.items()},
            "line_items": [li.to_dict() for li in self.line_items],
            "validation": self.validation.to_dict() if self.validation else None,
            "normalized": {k: _jsonable(v) for k, v in self.normalized.items()},
            "status": self.status.value if isinstance(self.status, Status) else self.status,
            "needs_review": self.status == Status.NEEDS_REVIEW,
            "overall_confidence": round(self.overall_confidence, 4),
            "review_reasons": self.review_reasons,
            "used_llm_fallback": self.used_llm_fallback,
            "trace": [t.to_dict() for t in self.trace],
            "model_versions": self.model_versions,
        }


__all__ = ["DocType", "Status", "FieldValue", "LineItem", "ValidationReport", "ToolTrace", "AgentState"]
