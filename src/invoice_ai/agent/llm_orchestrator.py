"""Optional LLM-vision fallback — the only cloud tool, isolated and feature-flagged.

This is exactly the reference implementation's GPT-4o-vision call, demoted to an
*optional escalation*. It is used only when validation fails and a key is
configured; if unavailable, the agent degrades gracefully to human review — the
system never depends on a paid API to function.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional

from ..config import AgentConfig
from ..logging_utils import get_logger
from .state import AgentState, FieldValue

logger = get_logger(__name__)

_FIELDS = ["vendor", "invoice_number", "invoice_date", "currency", "subtotal", "tax", "total"]


class LLMFallback:
    name = "llm_vision"

    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg
        self._client = None
        self._ready = False
        if not cfg.llm_fallback_enabled:
            return
        key = os.environ.get(cfg.llm_api_key_env)
        if not key:
            logger.info("LLM fallback enabled but no key in $%s.", cfg.llm_api_key_env)
            return
        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=key)
            self._ready = True
        except Exception as exc:
            logger.warning("anthropic SDK unavailable (%s).", exc)

    def available(self) -> bool:
        return self._ready

    def extract(self, state: AgentState) -> AgentState:
        """Re-extract fields from the OCR text via the LLM (JSON), filling gaps."""
        if not self._ready:
            return state
        try:
            system = ("You extract invoice/receipt fields. Return ONLY compact JSON with keys "
                      f"{_FIELDS}. Amounts as numbers, date as YYYY-MM-DD, currency as ISO-4217. "
                      "Use null if not present.")
            user = f"Document text:\n{state.ocr_text[:6000]}"
            resp = self._client.messages.create(
                model=self.cfg.llm_model, max_tokens=512, system=system,
                messages=[{"role": "user", "content": user}])
            raw = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
            data = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
            for k in _FIELDS:
                v = data.get(k)
                if v not in (None, "", "null"):
                    # LLM fills only missing or low-confidence fields.
                    cur = state.fields.get(k)
                    if cur is None or cur.confidence < self.cfg.field_conf_min:
                        state.fields[k] = FieldValue(v, 0.82, "llm")
            state.used_llm_fallback = True
            state.model_versions["llm"] = self.cfg.llm_model
        except Exception as exc:
            logger.warning("LLM fallback failed (%s); proceeding to human review.", exc)
        return state


__all__ = ["LLMFallback"]
