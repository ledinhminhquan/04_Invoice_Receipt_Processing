"""The agent's tools — each mutates and returns the shared :class:`AgentState`.

All tools run fully offline (no paid API) except the optional, isolated
``llm_vision_fallback`` (in :mod:`.llm_orchestrator`). Tools never raise past the
agent; failures are recorded and the agent decides what to do next.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from ..config import AgentConfig, OcrConfig
from ..logging_utils import get_logger
from .state import AgentState, DocType, FieldValue, LineItem, Status

logger = get_logger(__name__)

_INVOICE_CUES = re.compile(r"\b(invoice|tax invoice|bill to|invoice no|invoice number|purchase order)\b", re.I)
_RECEIPT_CUES = re.compile(r"\b(receipt|change due|cash|card|pos|thank you for|store|cashier)\b", re.I)
_AMOUNT_TOK = re.compile(r"[$€£₫¥]?\s?\d{1,3}(?:[.,]\d{3})*[.,]\d{2}")


# ─────────────────────────────────────────────────────────────────────────────
# OCR
# ─────────────────────────────────────────────────────────────────────────────

def run_ocr(state: AgentState, page_images: List, native_tokens: Optional[List[dict]],
            ocr_cfg: OcrConfig, engine: Optional[str] = None) -> AgentState:
    """Fill ocr_tokens/text/mean_conf. Uses native PDF tokens when available."""
    if native_tokens:
        state.ocr_tokens = native_tokens
        state.ocr_engine = "native_pdf"
        state.ocr_mean_conf = 1.0
        state.scan_quality = 1.0
    else:
        from ..ocr.engine import ocr_image
        eng = engine or (ocr_cfg.fallback_engine if ocr_cfg.engine == "auto" else ocr_cfg.engine)
        all_tokens, confs = [], []
        for pi, img in enumerate(page_images, start=1):
            res = ocr_image(img, engine=eng, lang=ocr_cfg.lang, page=pi)
            all_tokens.extend(res.tokens)
            if res.tokens:
                confs.append(res.mean_conf)
                state.ocr_engine = res.engine
        state.ocr_tokens = all_tokens
        state.ocr_mean_conf = sum(confs) / len(confs) if confs else 0.0
        state.scan_quality = min(1.0, state.ocr_mean_conf * 1.1)
    state.ocr_text = "\n".join(_lines_from_tokens(state.ocr_tokens))
    state.page_count = max(state.page_count, len({t.get("page", 1) for t in state.ocr_tokens}) or 1)
    state.status = Status.OCR_DONE
    return state


def _lines_from_tokens(tokens: List[dict]) -> List[str]:
    """Group tokens into text lines by (page, y-band) so regex sees line structure."""
    if not tokens:
        return []
    rows = {}
    for t in tokens:
        y = t["bbox"][1]
        key = (t.get("page", 1), round(y / 12))   # ~12px band
        rows.setdefault(key, []).append(t)
    lines = []
    for key in sorted(rows):
        toks = sorted(rows[key], key=lambda t: t["bbox"][0])
        lines.append(" ".join(t["text"] for t in toks))
    return lines


# ─────────────────────────────────────────────────────────────────────────────
# Classify
# ─────────────────────────────────────────────────────────────────────────────

def classify_document(state: AgentState) -> AgentState:
    text = state.ocr_text or ""
    inv = len(_INVOICE_CUES.findall(text))
    rec = len(_RECEIPT_CUES.findall(text))
    if inv == 0 and rec == 0:
        # weak signal: if there's a total-like amount, assume invoice; else other
        state.doc_type = DocType.INVOICE if _AMOUNT_TOK.search(text) else DocType.OTHER
        state.doc_type_conf = 0.5 if state.doc_type == DocType.INVOICE else 0.4
    elif inv >= rec:
        state.doc_type = DocType.INVOICE
        state.doc_type_conf = round(min(0.99, 0.6 + 0.1 * inv), 3)
    else:
        state.doc_type = DocType.RECEIPT
        state.doc_type_conf = round(min(0.99, 0.6 + 0.1 * rec), 3)
    state.status = Status.ROUTED
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Field + line-item extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_fields(state: AgentState, layout_extractor=None, page_images: Optional[List] = None) -> AgentState:
    """LayoutLMv3 if available, else the regex baseline over OCR text."""
    fields: Dict[str, FieldValue] = {}
    if layout_extractor is not None and page_images:
        try:
            fields = layout_extractor.extract(page_images[0], [t for t in state.ocr_tokens if t.get("page", 1) == 1])
            state.model_versions["extractor"] = f"layoutlmv3-{getattr(layout_extractor, 'version', '1.0.0')}"
        except Exception as exc:
            logger.warning("LayoutLMv3 extract failed (%s); falling back to regex.", exc)
    if not fields:
        from ..models.baseline_regex import extract_fields as regex_extract
        fields = regex_extract(state.ocr_text, state.ocr_tokens)
        state.model_versions["extractor"] = "regex-1.0.0"
    state.fields.update(fields)
    state.status = Status.EXTRACTED
    return state


def extract_line_items(state: AgentState) -> AgentState:
    """Heuristic table rows: group tokens by y-band; a row with a trailing amount is a line."""
    tokens = [t for t in state.ocr_tokens if t.get("page", 1) == 1]
    rows = {}
    for t in tokens:
        key = round(t["bbox"][1] / 12)
        rows.setdefault(key, []).append(t)
    items: List[LineItem] = []
    for ri, key in enumerate(sorted(rows)):
        toks = sorted(rows[key], key=lambda t: t["bbox"][0])
        line_text = " ".join(t["text"] for t in toks)
        if any(w in line_text.lower() for w in ("total", "subtotal", "tax", "balance", "vat")):
            continue
        amounts = [t for t in toks if _AMOUNT_TOK.fullmatch(t["text"].strip()) or _AMOUNT_TOK.search(t["text"])]
        if not amounts:
            continue
        amt_tok = amounts[-1]
        from ..models.baseline_regex import _to_amount
        amt = _to_amount(amt_tok["text"])
        if amt is None or amt <= 0:
            continue
        desc_toks = [t for t in toks if t["bbox"][0] < amt_tok["bbox"][0] and not _AMOUNT_TOK.search(t["text"])]
        desc = " ".join(t["text"] for t in desc_toks).strip()
        if len(desc) < 2:
            continue
        items.append(LineItem(
            description=FieldValue(desc[:80], 0.6, state.ocr_engine),
            amount=FieldValue(amt, 0.6, state.ocr_engine, bbox=tuple(amt_tok["bbox"])),
            row=ri,
        ))
    state.line_items = items[:50]
    return state


__all__ = ["run_ocr", "classify_document", "extract_fields", "extract_line_items"]
