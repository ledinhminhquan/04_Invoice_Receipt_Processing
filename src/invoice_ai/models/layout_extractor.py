"""LayoutLMv3 token-classification inference for header-field extraction.

Loads a fine-tuned ``LayoutLMv3ForTokenClassification`` and turns OCR tokens
(words + 0-1000-normalised boxes) into header :class:`FieldValue` objects with a
confidence (mean softmax over the field's tokens) and a provenance ``bbox``.
Heavy deps are imported lazily; if no fine-tuned model exists the agent uses the
regex baseline instead.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from ..agent.state import FieldValue
from ..config import ModelConfig
from ..logging_utils import get_logger
from ..ocr.engine import normalize_boxes
from .model_registry import has_model, load_model_metadata, read_labels, resolve_latest

logger = get_logger(__name__)

# Map raw entity types (from the label scheme) to our canonical field names.
ENTITY_TO_FIELD = {
    "COMPANY": "vendor", "VENDOR": "vendor", "SELLER": "vendor",
    "DATE": "invoice_date", "INVOICE_DATE": "invoice_date",
    "ADDRESS": "address",
    "TOTAL": "total", "GRAND_TOTAL": "total", "AMOUNT": "total",
    "SUBTOTAL": "subtotal", "TAX": "tax", "VAT": "tax",
    "INVOICE_NO": "invoice_number", "INVOICE_NUMBER": "invoice_number",
}


def _entity_type(label: str) -> Optional[str]:
    if not label or label == "O":
        return None
    # strip BIO / S- prefixes
    for pre in ("B-", "I-", "S-", "E-"):
        if label.startswith(pre):
            return label[len(pre):].upper()
    return label.upper()


class LayoutExtractor:
    name = "layoutlmv3"

    def __init__(self, cfg: Optional[ModelConfig] = None, model_dir: Optional[str] = None, device: Optional[str] = None):
        import torch
        from transformers import AutoModelForTokenClassification, AutoProcessor

        self.cfg = cfg or ModelConfig()
        self.model_dir = resolve_latest(model_dir or self.cfg.output_dir)
        if not has_model(self.model_dir):
            raise FileNotFoundError(f"No fine-tuned layout model at {self.model_dir}.")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(str(self.model_dir), apply_ocr=False)
        self.model = AutoModelForTokenClassification.from_pretrained(str(self.model_dir)).to(self.device).eval()
        self.id2label = read_labels(self.model_dir) or {int(k): v for k, v in self.model.config.id2label.items()}
        meta = load_model_metadata(self.model_dir)
        self.version = meta.get("version", "1.0.0")

    @classmethod
    def from_config(cls, cfg: ModelConfig, device: Optional[str] = None) -> "LayoutExtractor":
        return cls(cfg, model_dir=str(cfg.output_dir), device=device)

    def extract(self, image, tokens: List[dict]) -> Dict[str, FieldValue]:
        import torch

        if not tokens:
            return {}
        words = [t["text"] for t in tokens]
        boxes = normalize_boxes(tokens, image.size)
        enc = self.processor(image.convert("RGB"), words, boxes=boxes, truncation=True,
                             max_length=self.cfg.max_length, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**enc).logits[0]
            probs = torch.softmax(logits.float(), dim=-1)
            conf, pred = probs.max(dim=-1)

        # Map model tokens back to words (first subword carries the label).
        word_ids = enc.word_ids(0)
        per_field = defaultdict(list)   # field -> [(word_idx, conf)]
        seen = set()
        for ti, wid in enumerate(word_ids):
            if wid is None or wid in seen:
                continue
            seen.add(wid)
            label = self.id2label.get(int(pred[ti]), "O")
            ent = _entity_type(label)
            if ent is None:
                continue
            field = ENTITY_TO_FIELD.get(ent, ent.lower())
            per_field[field].append((wid, float(conf[ti])))

        fields: Dict[str, FieldValue] = {}
        for field, hits in per_field.items():
            idxs = [w for w, _ in hits]
            text = " ".join(words[i] for i in idxs)
            mean_conf = sum(c for _, c in hits) / len(hits)
            bbox = tuple(tokens[idxs[0]]["bbox"]) if idxs else None
            value = _coerce(field, text)
            fields[field] = FieldValue(value, round(mean_conf, 4), "layoutlmv3", bbox=bbox, raw=text)
        return fields


def _coerce(field: str, text: str):
    if field in ("total", "subtotal", "tax"):
        from .baseline_regex import _to_amount
        amt = _to_amount(text)
        return amt if amt is not None else text
    return text


__all__ = ["LayoutExtractor", "ENTITY_TO_FIELD"]
