"""Built-in synthetic invoices (as OCR tokens) for tests, the demo and CI.

Fully synthetic (no real data). Tokens carry pixel bboxes so the agent's
field/line-item extraction + validation run **with no OCR binary and no model**.
One invoice reconciles; one has a deliberately wrong printed total (the
``2,160`` vs ``1,860`` case from the design brief) to exercise the human-review
routing.
"""

from __future__ import annotations

from typing import Dict, List


def _tok(text: str, x: int, y: int, conf: float = 0.96, h: int = 18, page: int = 1) -> dict:
    return {"text": text, "bbox": (x, y, x + max(8, 9 * len(text)), y + h), "conf": conf, "page": page}


def _line(y: int, parts: List[tuple]) -> List[dict]:
    """parts = [(x, "text words"), ...]; multi-word strings are split into tokens."""
    out = []
    for x, text in parts:
        cx = x
        for w in text.split():
            out.append(_tok(w, cx, y))
            cx += 9 * len(w) + 8
    return out


def _invoice_tokens(total_str: str) -> List[dict]:
    """Build a 2-line-item invoice; pass the printed total to (mis)match."""
    toks: List[dict] = []
    toks += _line(30, [(50, "Acme Corp Ltd")])
    toks += _line(60, [(50, "123 Market Street, London")])
    toks += _line(110, [(50, "Invoice No: INV-2024-077"), (380, "Date: 2024-11-03")])
    toks += _line(150, [(50, "Bill To: Globex Inc")])
    # line items (description left, amount right)
    toks += _line(220, [(50, "Consulting services"), (300, "10"), (380, "150.00"), (480, "1500.00")])
    toks += _line(250, [(50, "Onboarding setup"), (300, "1"), (380, "300.00"), (480, "300.00")])
    # totals block
    toks += _line(320, [(360, "Subtotal"), (480, "1800.00")])
    toks += _line(345, [(360, "Tax 20%"), (480, "360.00")])
    toks += _line(370, [(360, "Total"), (480, total_str)])
    toks += _line(420, [(50, "GBP")])
    return toks


def _receipt_tokens() -> List[dict]:
    toks: List[dict] = []
    toks += _line(30, [(50, "QuickMart Store")])
    toks += _line(55, [(50, "Receipt #4471  Date: 2026-05-14")])
    toks += _line(110, [(50, "Coffee"), (200, "3.50")])
    toks += _line(135, [(50, "Sandwich"), (200, "5.00")])
    toks += _line(180, [(50, "Subtotal"), (200, "8.50")])
    toks += _line(205, [(50, "Tax"), (200, "0.85")])
    toks += _line(230, [(50, "Total"), (200, "9.35")])
    toks += _line(260, [(50, "Card  Thank you for shopping")])
    return toks


# name -> {tokens, note}
SAMPLE_DOCS: Dict[str, dict] = {
    "invoice_ok.png": {"tokens": _invoice_tokens("2160.00"),
                       "note": "reconciles: subtotal 1800 + tax 360 == total 2160"},
    "invoice_bad_total.png": {"tokens": _invoice_tokens("1860.00"),
                              "note": "printed total 1860 != subtotal+tax 2160 -> needs review"},
    "receipt_ok.png": {"tokens": _receipt_tokens(),
                       "note": "receipt: subtotal 8.50 + tax 0.85 == total 9.35"},
}

__all__ = ["SAMPLE_DOCS"]
