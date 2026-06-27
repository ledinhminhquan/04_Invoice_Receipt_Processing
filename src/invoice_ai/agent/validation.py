"""Validation + normalization — the heart of the agent (pure rules, no API).

This is the core value-add over a pure-LLM extractor: instead of trusting the
extracted numbers, the agent **checks the arithmetic**. A printed total that does
not equal ``subtotal + tax`` (within a penny-rounding epsilon) is caught and the
document is routed to a human — exactly the silent error a type-coercing pipeline
misses. Money is handled as :class:`~decimal.Decimal` to avoid float errors.
"""

from __future__ import annotations

import re
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Optional

from ..config import AgentConfig
from .state import AgentState, FieldValue, ValidationReport

REQUIRED_FIELDS = ["invoice_number", "invoice_date", "total", "vendor"]
_CURRENCY_SYMBOLS = {"$": "USD", "€": "EUR", "£": "GBP", "₫": "VND", "¥": "JPY"}


def _dec(v: Any) -> Optional[Decimal]:
    if v is None:
        return None
    try:
        return Decimal(str(v)).quantize(Decimal("0.01"))
    except (InvalidOperation, ValueError):
        return None


def _field_value(state: AgentState, key: str):
    fv = state.fields.get(key)
    return fv.value if isinstance(fv, FieldValue) else None


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation
# ─────────────────────────────────────────────────────────────────────────────

def normalize(state: AgentState, cfg: AgentConfig) -> AgentState:
    """Canonicalise dates (ISO-8601), amounts (Decimal 2dp) and currency (ISO-4217)."""
    norm: Dict[str, Any] = {}

    # amounts
    for key in ("subtotal", "tax", "total"):
        d = _dec(_field_value(state, key))
        if d is not None:
            norm[key] = d
            state.fields[key].value = float(d)

    # date
    raw_date = _field_value(state, "invoice_date")
    if raw_date:
        iso = _parse_date(str(raw_date))
        if iso:
            norm["invoice_date"] = iso
            state.fields["invoice_date"].raw = str(raw_date)
            state.fields["invoice_date"].value = iso

    # currency
    cur = _field_value(state, "currency")
    if cur:
        cur = _CURRENCY_SYMBOLS.get(str(cur), str(cur).upper())
        state.currency = cur
        norm["currency"] = cur

    for key in ("invoice_number", "vendor"):
        v = _field_value(state, key)
        if v is not None:
            norm[key] = v

    state.normalized = norm
    return state


def _parse_date(s: str) -> Optional[str]:
    s = s.strip()
    fmts = ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%d.%m.%Y",
            "%Y/%m/%d", "%d %b %Y", "%d %B %Y", "%d/%m/%y", "%m/%d/%y"]
    for fmt in fmts:
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    try:  # last resort: dateutil if available
        from dateutil import parser
        return parser.parse(s, dayfirst=True).strftime("%Y-%m-%d")
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate(state: AgentState, cfg: AgentConfig) -> AgentState:
    checks: Dict[str, bool] = {}
    errors = []

    subtotal = _dec(state.normalized.get("subtotal", _field_value(state, "subtotal")))
    tax = _dec(state.normalized.get("tax", _field_value(state, "tax")))
    total = _dec(state.normalized.get("total", _field_value(state, "total")))

    # line-item sum + per-line math
    line_sum = Decimal("0")
    per_line_ok = True
    for li in state.line_items:
        amt = _dec(li.amount.value) if li.amount else None
        if amt is not None:
            line_sum += amt
        qty = _dec(li.quantity.value) if li.quantity else None
        up = _dec(li.unit_price.value) if li.unit_price else None
        if qty is not None and up is not None and amt is not None:
            if abs(qty * up - amt) > Decimal("0.01"):
                per_line_ok = False
    if state.line_items:
        checks["per_line_math"] = per_line_ok

    # epsilon tolerance
    eps = Decimal(str(max(cfg.reconcile_eps_abs, cfg.reconcile_eps_rel * float(total or 0))))

    reconciles = True
    reconcile_delta = Decimal("0")
    if total is not None and subtotal is not None:
        expected = subtotal + (tax or Decimal("0"))
        reconcile_delta = (total - expected).quantize(Decimal("0.01"))
        checks["subtotal_plus_tax_equals_total"] = abs(reconcile_delta) <= eps
        reconciles = reconciles and checks["subtotal_plus_tax_equals_total"]
    if state.line_items and subtotal is not None:
        checks["lines_equal_subtotal"] = abs(line_sum - subtotal) <= eps
        reconciles = reconciles and checks["lines_equal_subtotal"]
    if state.line_items and total is not None and subtotal is None:
        checks["lines_equal_total"] = abs(line_sum + (tax or Decimal("0")) - total) <= eps
        reconciles = reconciles and checks["lines_equal_total"]

    # field-level checks
    inv_no = _field_value(state, "invoice_number")
    checks["invoice_no_plausible"] = bool(inv_no and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9\-\/]{2,20}", str(inv_no)))
    date_iso = state.normalized.get("invoice_date")
    checks["date_valid"] = _date_plausible(date_iso)
    checks["currency_detected"] = bool(state.currency)

    missing = [f for f in REQUIRED_FIELDS if _field_value(state, f) in (None, "")]
    low_conf = [k for k, fv in state.fields.items()
                if isinstance(fv, FieldValue) and k in REQUIRED_FIELDS and fv.confidence < cfg.field_conf_min]

    if total is None:
        errors.append("no total detected")
    if not reconciles and total is not None and subtotal is not None:
        errors.append(f"totals don't reconcile: total={total} vs subtotal+tax off by {reconcile_delta}")

    state.validation = ValidationReport(
        reconciles=reconciles and (total is not None and subtotal is not None or not state.line_items),
        reconcile_delta=float(reconcile_delta),
        checks=checks, missing_required=missing, low_confidence_fields=low_conf, errors=errors,
    )
    return state


def _date_plausible(iso: Optional[str]) -> bool:
    if not iso:
        return False
    try:
        d = datetime.strptime(iso, "%Y-%m-%d")
    except (ValueError, TypeError):
        return False
    year = d.year
    return 1990 <= year <= datetime.now().year + 1


def compute_overall_confidence(state: AgentState) -> float:
    """Conservative bottleneck: the weakest required signal blocks auto-approval."""
    signals = [state.doc_type_conf or 1.0, state.ocr_mean_conf or 1.0]
    for f in REQUIRED_FIELDS:
        fv = state.fields.get(f)
        if isinstance(fv, FieldValue):
            signals.append(fv.confidence)
    return round(min(signals), 4) if signals else 0.0


__all__ = ["normalize", "validate", "compute_overall_confidence", "REQUIRED_FIELDS"]
