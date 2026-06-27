"""Invoice & Receipt Processing System (Document-AI / Key Information Extraction).

A production, agentic system that extracts structured data (vendor, date, invoice
number, currency, subtotal, tax, total, line items) from scanned receipt/invoice
images and PDFs: ingest -> classify -> OCR -> layout extraction (LayoutLMv3) ->
line-item extraction -> validation (totals reconcile?) -> normalize -> JSON +
confidence + needs-review routing.

Runs offline (no paid vision API) with open models + a deterministic agent;
upgrades to a fine-tuned LayoutLMv3/Donut model and an optional LLM-vision brain.
"""

from __future__ import annotations

__version__ = "1.0.0"
__all__ = ["__version__"]
