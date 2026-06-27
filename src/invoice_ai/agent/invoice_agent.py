"""The agentic invoice/receipt processing pipeline.

Wires the tools + decision points into the control loop:

    load → OCR (quality gate + retry = D1a) → classify (route = D1b)
         → extract fields → extract line items → normalize → validate (D2)
         → [optional LLM fallback] → final confidence gate (D3)
         → AUTO-APPROVE | NEEDS-REVIEW

Runs fully offline (regex extractor + OCR + rules). Loads a fine-tuned LayoutLMv3
extractor and an optional LLM-vision fallback when configured.
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional

from ..config import AppConfig
from ..logging_utils import JsonlLogger, get_logger
from ..ocr.pdf import load_pages
from . import tools
from .llm_orchestrator import LLMFallback
from .state import AgentState, DocType, Status, ToolTrace
from .validation import compute_overall_confidence, normalize, validate

logger = get_logger(__name__)


class InvoiceAgent:
    def __init__(self, cfg: Optional[AppConfig] = None, *, load_model: bool = True):
        self.cfg = cfg or AppConfig()
        self.layout_extractor = self._load_extractor(load_model)
        self.llm = LLMFallback(self.cfg.agent)
        self._log = JsonlLogger(self.cfg.serving.extraction_log_path) if self.cfg.serving.log_extractions else None
        self._review = JsonlLogger(self.cfg.serving.review_queue_path)

    def _load_extractor(self, load_model: bool):
        if not load_model:
            return None
        try:
            from ..models.layout_extractor import LayoutExtractor
            ext = LayoutExtractor.from_config(self.cfg.model)
            logger.info("Loaded fine-tuned LayoutLMv3 extractor (v%s)", ext.version)
            return ext
        except Exception as exc:
            logger.info("No fine-tuned layout model (%s); using regex baseline extractor.", exc)
            return None

    # ---- step timing/trace -------------------------------------------------
    def _step(self, state: AgentState, name: str, fn: Callable[[], AgentState], summary: str = "") -> AgentState:
        t0 = time.perf_counter()
        try:
            state = fn()
            ok, err = True, None
        except Exception as exc:
            logger.warning("tool %s failed: %s", name, exc)
            ok, err = False, str(exc)
        state.add_trace(ToolTrace(tool=name, ok=ok, latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                                  summary=summary or name, error=err))
        return state

    # ---- main entrypoint ---------------------------------------------------
    def process(self, doc_path: Optional[str] = None, raw_bytes: Optional[bytes] = None,
                filename: str = "document") -> AgentState:
        state = AgentState(filename=filename)
        tmp = None
        if doc_path is None and raw_bytes is not None:
            suffix = Path(filename).suffix or ".pdf"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(raw_bytes)
            tmp.close()
            doc_path = tmp.name
        state.doc_path = doc_path or ""

        try:
            loaded = load_pages(doc_path, dpi=self.cfg.ocr.dpi) if doc_path else None
            if not loaded or not loaded.page_images:
                state.review_reasons.append("could not load/parse the document")
                return self._finish(state, Status.FAILED)
            state.page_count = loaded.n_pages

            # D1a: OCR with quality gate + engine-switch retry
            engine = None if loaded.is_digital else (
                self.cfg.ocr.fallback_engine if self.cfg.ocr.engine == "auto" else self.cfg.ocr.engine)
            while True:
                state = self._step(state, "run_ocr",
                                   lambda e=engine: tools.run_ocr(state, loaded.page_images, loaded.native_tokens,
                                                                  self.cfg.ocr, engine=e),
                                   summary=f"engine={engine or 'native'} conf={state.ocr_mean_conf:.2f}")
                if (loaded.is_digital or state.ocr_mean_conf >= self.cfg.agent.ocr_min
                        or state.scan_quality >= self.cfg.agent.quality_min):
                    break
                state.attempts["ocr"] = state.attempts.get("ocr", 0) + 1
                if state.attempts["ocr"] > self.cfg.agent.max_ocr_attempts:
                    state.review_reasons.append(f"low OCR confidence ({state.ocr_mean_conf:.2f}); rescan recommended")
                    return self._finish(state, Status.NEEDS_REVIEW)
                engine = "paddle" if engine == "tesseract" else "tesseract"

            if not state.ocr_tokens:
                state.review_reasons.append("no text extracted (empty OCR)")
                return self._finish(state, Status.NEEDS_REVIEW)

            # D1b: classify + route
            state = self._step(state, "classify_document", lambda: tools.classify_document(state),
                               summary=f"{state.doc_type} ({state.doc_type_conf})")
            if state.doc_type == DocType.OTHER:
                state.review_reasons.append("not an invoice/receipt")
                return self._finish(state, Status.FAILED)

            # extract
            state = self._step(state, "extract_fields",
                               lambda: tools.extract_fields(state, self.layout_extractor, loaded.page_images),
                               summary=f"{len(state.fields)} fields")
            if state.doc_type == DocType.INVOICE:
                state = self._step(state, "extract_line_items", lambda: tools.extract_line_items(state),
                                   summary=f"{len(state.line_items)} line items")

            # normalize + validate (D2)
            state = self._step(state, "normalize", lambda: normalize(state, self.cfg.agent))
            state = self._step(state, "validate", lambda: validate(state, self.cfg.agent),
                               summary=f"reconciles={state.validation.reconciles if state.validation else '?'}")
            if self._needs_help(state) and self.llm.available() and not state.used_llm_fallback:
                state = self._step(state, "llm_vision_fallback", lambda: self.llm.extract(state), summary="escalated")
                state = self._step(state, "normalize", lambda: normalize(state, self.cfg.agent))
                state = self._step(state, "validate", lambda: validate(state, self.cfg.agent))

            # D3: final gate
            state.overall_confidence = compute_overall_confidence(state)
            v = state.validation
            if v and v.reconciles and not v.missing_required and state.overall_confidence >= self.cfg.agent.auto_approve_min:
                return self._finish(state, Status.AUTO_APPROVED)

            # collect review reasons
            if v and not v.reconciles:
                state.review_reasons.append(
                    f"totals don't reconcile (delta {v.reconcile_delta})" if v.reconcile_delta else "totals don't reconcile")
            if v:
                state.review_reasons += [f"missing: {m}" for m in v.missing_required]
                state.review_reasons += [f"low confidence: {f}" for f in v.low_confidence_fields]
            if state.overall_confidence < self.cfg.agent.auto_approve_min and not state.review_reasons:
                state.review_reasons.append(f"overall confidence {state.overall_confidence:.2f} below auto-approve threshold")
            return self._finish(state, Status.NEEDS_REVIEW)
        finally:
            if tmp is not None:
                try:
                    os.unlink(tmp.name)
                except OSError:
                    pass

    def process_tokens(self, tokens: list, filename: str = "document",
                       page_images: Optional[list] = None) -> AgentState:
        """Run the post-OCR pipeline on pre-supplied OCR tokens.

        Lets the agent run with no OCR binary / no document file (used by tests
        and the offline demo). ``tokens`` = ``[{text, bbox, conf, page}]``.
        """
        state = AgentState(filename=filename)
        state.ocr_tokens = list(tokens)
        state.ocr_engine = "supplied"
        state.ocr_mean_conf = (sum(t.get("conf", 1.0) for t in tokens) / len(tokens)) if tokens else 0.0
        state.scan_quality = 1.0
        state.ocr_text = "\n".join(tools._lines_from_tokens(state.ocr_tokens))
        state.page_count = len({t.get("page", 1) for t in tokens}) or 1
        state.status = Status.OCR_DONE

        state = self._step(state, "classify_document", lambda: tools.classify_document(state),
                           summary=f"{state.doc_type}")
        if state.doc_type == DocType.OTHER:
            state.review_reasons.append("not an invoice/receipt")
            return self._finish(state, Status.FAILED)
        state = self._step(state, "extract_fields",
                           lambda: tools.extract_fields(state, self.layout_extractor, page_images),
                           summary=f"{len(state.fields)} fields")
        if state.doc_type == DocType.INVOICE:
            state = self._step(state, "extract_line_items", lambda: tools.extract_line_items(state))
        state = self._step(state, "normalize", lambda: normalize(state, self.cfg.agent))
        state = self._step(state, "validate", lambda: validate(state, self.cfg.agent),
                           summary=f"reconciles={state.validation.reconciles if state.validation else '?'}")
        if self._needs_help(state) and self.llm.available() and not state.used_llm_fallback:
            state = self._step(state, "llm_vision_fallback", lambda: self.llm.extract(state))
            state = self._step(state, "normalize", lambda: normalize(state, self.cfg.agent))
            state = self._step(state, "validate", lambda: validate(state, self.cfg.agent))
        state.overall_confidence = compute_overall_confidence(state)
        v = state.validation
        if v and v.reconciles and not v.missing_required and state.overall_confidence >= self.cfg.agent.auto_approve_min:
            return self._finish(state, Status.AUTO_APPROVED)
        if v and not v.reconciles:
            state.review_reasons.append(
                f"totals don't reconcile (delta {v.reconcile_delta})" if v.reconcile_delta else "totals don't reconcile")
        if v:
            state.review_reasons += [f"missing: {m}" for m in v.missing_required]
            state.review_reasons += [f"low confidence: {f}" for f in v.low_confidence_fields]
        return self._finish(state, Status.NEEDS_REVIEW)

    def _needs_help(self, state: AgentState) -> bool:
        v = state.validation
        return bool(v and (not v.reconciles or v.missing_required or v.low_confidence_fields))

    def _finish(self, state: AgentState, status: Status) -> AgentState:
        state.status = status
        state.model_versions.setdefault("model_version", self.cfg.serving.model_version)
        if self._log is not None:
            try:
                self._log.log("extract", filename=state.filename, doc_type=str(state.doc_type.value),
                              status=str(status.value), reconciles=bool(state.validation.reconciles) if state.validation else None,
                              overall_confidence=state.overall_confidence, needs_review=status == Status.NEEDS_REVIEW)
            except Exception:
                pass
        if status == Status.NEEDS_REVIEW:
            try:
                self._review.log("review", filename=state.filename, reasons=state.review_reasons,
                                 fields={k: v.to_dict() for k, v in state.fields.items()})
            except Exception:
                pass
        return state


_AGENT: Optional[InvoiceAgent] = None


def get_agent(cfg: Optional[AppConfig] = None, **kwargs) -> InvoiceAgent:
    global _AGENT
    if _AGENT is None:
        _AGENT = InvoiceAgent(cfg, **kwargs)
    return _AGENT


__all__ = ["InvoiceAgent", "get_agent"]
