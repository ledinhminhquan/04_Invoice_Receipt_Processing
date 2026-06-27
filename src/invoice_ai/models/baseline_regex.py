"""Regex / heuristic field extractor — the mandatory baseline + offline fallback.

This zero-training, fully-interpretable extractor is (a) the floor the trained
LayoutLMv3 must beat, and (b) the fallback the agent uses when no fine-tuned model
is present — so the system always extracts *something* and the demo runs offline.

It works over OCR/native text (and, when token boxes are available, attaches a
provenance ``bbox`` to each field for the review UI).
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from ..agent.state import FieldValue
from ..logging_utils import get_logger

logger = get_logger(__name__)

_CURRENCY = {"$": "USD", "US$": "USD", "USD": "USD", "€": "EUR", "EUR": "EUR",
             "£": "GBP", "GBP": "GBP", "₫": "VND", "VND": "VND", "¥": "JPY", "JPY": "JPY"}

# Capture a full money number (handles "1800.00", "1,800.00", "9.35"); not just
# thousands-grouped forms. \b...total\b avoids matching "total" inside "subtotal".
_AMOUNT = r"([$€£₫]?\s?\d[\d.,]*\d)"
_TOTAL_RE = re.compile(r"\b(?:grand\s+total|total\s+due|amount\s+due|balance\s+due|total)\b\s*[:\-]?\s*" + _AMOUNT, re.I)
_SUBTOTAL_RE = re.compile(r"(?:sub[\s\-]?total)\s*[:\-]?\s*" + _AMOUNT, re.I)
_TAX_RE = re.compile(r"(?:tax|vat|gst)\s*(?:\(?\d{1,2}%\)?)?\s*[:\-]?\s*" + _AMOUNT, re.I)
_INVNO_RE = re.compile(r"(?:invoice\s*(?:no|number|#)|inv\s*#?)\s*[:\-]?\s*([A-Za-z0-9][A-Za-z0-9\-\/]{2,20})", re.I)
_DATE_RE = re.compile(
    r"(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}|\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2}|"
    r"\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4})", re.I)
_CURR_RE = re.compile(r"([$€£₫¥]|USD|EUR|GBP|VND|JPY)")


def _to_amount(s: str) -> Optional[float]:
    s = re.sub(r"[^\d.,]", "", s).strip()
    if not s:
        return None
    # Heuristic: if both separators, the last one is the decimal point.
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif s.count(",") == 1 and len(s.split(",")[-1]) == 2:
        s = s.replace(",", ".")
    else:
        s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def _bbox_for(value_text: str, tokens: Optional[List[dict]]):
    if not tokens or not value_text:
        return None
    digits = re.sub(r"[^\d]", "", value_text)
    for t in tokens:
        if digits and digits in re.sub(r"[^\d]", "", t["text"]):
            return tuple(t["bbox"])
    return None


class RegexExtractor:
    name = "regex"
    version = "1.0.0"

    def extract_fields(self, text: str, tokens: Optional[List[dict]] = None) -> Dict[str, FieldValue]:
        fields: Dict[str, FieldValue] = {}

        m = _INVNO_RE.search(text)
        if m:
            fields["invoice_number"] = FieldValue(m.group(1).strip(), 0.7, "rule")

        m = _DATE_RE.search(text)
        if m:
            fields["invoice_date"] = FieldValue(m.group(1).strip(), 0.65, "rule", raw=m.group(1).strip())

        for key, rx in (("subtotal", _SUBTOTAL_RE), ("tax", _TAX_RE), ("total", _TOTAL_RE)):
            m = rx.search(text)
            if m:
                amt = _to_amount(m.group(1))
                if amt is not None:
                    fields[key] = FieldValue(amt, 0.7 if key == "total" else 0.6, "rule",
                                             bbox=_bbox_for(m.group(1), tokens))

        # Fallback for total: largest currency amount on the page.
        if "total" not in fields:
            amounts = [_to_amount(a) for a in re.findall(_AMOUNT, text)]
            amounts = [a for a in amounts if a is not None]
            if amounts:
                fields["total"] = FieldValue(max(amounts), 0.4, "rule")

        m = _CURR_RE.search(text)
        if m:
            fields["currency"] = FieldValue(_CURRENCY.get(m.group(1), m.group(1)), 0.6, "rule")

        # Vendor heuristic: the first non-empty, mostly-alphabetic line near the top.
        for line in [l.strip() for l in text.splitlines() if l.strip()][:6]:
            if len(line) >= 3 and sum(c.isalpha() for c in line) / len(line) > 0.5 and not _DATE_RE.search(line):
                fields["vendor"] = FieldValue(line[:80], 0.45, "rule")
                break

        return fields


def extract_fields(text: str, tokens: Optional[List[dict]] = None) -> Dict[str, FieldValue]:
    return RegexExtractor().extract_fields(text, tokens)


__all__ = ["RegexExtractor", "extract_fields"]
