"""Evaluation: token-level entity-F1 (if a model is trained) + end-to-end agent.

Two complementary views:
  * **Layout F1** — entity-level seqeval P/R/F1 of the fine-tuned LayoutLMv3 on a
    held-out KIE test split (vs the model's own training; best-effort).
  * **Agent end-to-end** — runs the full agent on the built-in synthetic invoices
    and reports field-extraction accuracy, **reconciliation-detection accuracy**
    (does it catch the bad total?), validation pass-rate and needs-review rate.

Writes a JSON snapshot under ``runs/eval-<stamp>/`` for the autoreport.
"""

from __future__ import annotations

import json
from typing import Dict, Optional

from ..config import AppConfig, run_dir
from ..logging_utils import get_logger, utc_stamp

logger = get_logger(__name__)

# Gold values for the built-in synthetic invoices (data/samples.py).
_GOLD = {
    "invoice_ok.png": {"subtotal": 1800.0, "tax": 360.0, "total": 2160.0, "reconciles": True},
    "invoice_bad_total.png": {"subtotal": 1800.0, "tax": 360.0, "total": 1860.0, "reconciles": False},
    "receipt_ok.png": {"subtotal": 8.5, "tax": 0.85, "total": 9.35, "reconciles": True},
}


def _agent_eval(cfg: AppConfig) -> Dict:
    from ..agent.invoice_agent import InvoiceAgent
    from ..data.samples import SAMPLE_DOCS

    cfg.serving.log_extractions = False
    agent = InvoiceAgent(cfg, load_model=False)
    field_hits = field_total = reconcile_hits = 0
    needs_review = 0
    per_doc = []
    for name, d in SAMPLE_DOCS.items():
        gold = _GOLD.get(name, {})
        state = agent.process_tokens(d["tokens"], filename=name)
        sd = state.to_dict()
        for k in ("subtotal", "tax", "total"):
            if k in gold:
                field_total += 1
                got = sd["fields"].get(k, {}).get("value")
                if got is not None and abs(float(got) - gold[k]) < 0.01:
                    field_hits += 1
        if "reconciles" in gold and sd["validation"]:
            reconcile_hits += int(sd["validation"]["reconciles"] == gold["reconciles"])
        needs_review += int(sd["status"] == "needs_review")
        per_doc.append({"name": name, "status": sd["status"],
                        "reconciles": sd["validation"]["reconciles"] if sd["validation"] else None})
    n = len(SAMPLE_DOCS)
    return {
        "field_accuracy": round(field_hits / max(1, field_total), 4),
        "reconcile_detection_accuracy": round(reconcile_hits / max(1, n), 4),
        "needs_review_rate": round(needs_review / max(1, n), 4),
        "n_docs": n, "per_doc": per_doc,
    }


def _layout_f1(cfg: AppConfig, which: str, limit: int) -> Optional[Dict]:
    try:
        import numpy as np
        import evaluate as hf_evaluate
        import torch
        from transformers import AutoModelForTokenClassification, AutoProcessor
        from ..data.dataset import load_kie_dataset
        from ..models.model_registry import has_model, resolve_latest
        from ..training.train_layoutlmv3 import _normalize_boxes

        model_dir = resolve_latest(cfg.model.output_dir)
        if not has_model(model_dir):
            return None
        splits, id2label, _, words_col = load_kie_dataset(cfg.data, which=which, limit=limit)
        test = splits["test"] or splits["validation"]
        processor = AutoProcessor.from_pretrained(str(model_dir), apply_ocr=False)
        model = AutoModelForTokenClassification.from_pretrained(str(model_dir)).eval()
        seqeval = hf_evaluate.load("seqeval")
        preds_all, labs_all = [], []
        for ex in test.select(range(min(limit or 50, len(test)))):
            img = ex["image"].convert("RGB")
            boxes = _normalize_boxes(ex["bboxes"], img.size[0], img.size[1])
            enc = processor(img, ex[words_col], boxes=boxes, word_labels=ex["ner_tags"],
                            truncation=True, max_length=cfg.model.max_length, return_tensors="pt")
            labels = enc.pop("labels")
            with torch.no_grad():
                logits = model(**enc).logits[0]
            pred = logits.argmax(-1).tolist()
            lab = labels[0].tolist()
            preds_all.append([id2label[p] for p, l in zip(pred, lab) if l != -100])
            labs_all.append([id2label[l] for p, l in zip(pred, lab) if l != -100])
        r = seqeval.compute(predictions=preds_all, references=labs_all, zero_division=0)
        return {"precision": r["overall_precision"], "recall": r["overall_recall"], "f1": r["overall_f1"]}
    except Exception as exc:
        logger.info("Layout F1 eval skipped (%s)", exc)
        return None


def evaluate(cfg: AppConfig, which: str = "sroie", limit: Optional[int] = 50) -> Dict:
    results = {"agent": _agent_eval(cfg)}
    lf1 = _layout_f1(cfg, which, limit or 50)
    if lf1:
        results["layout_f1"] = lf1
    results["summary"] = {
        "field_accuracy": results["agent"]["field_accuracy"],
        "reconcile_detection_accuracy": results["agent"]["reconcile_detection_accuracy"],
        "layout_f1": lf1["f1"] if lf1 else None,
    }
    out = run_dir() / f"eval-{utc_stamp()}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Eval -> %s | %s", out / "eval.json", results["summary"])
    return results


__all__ = ["evaluate"]
