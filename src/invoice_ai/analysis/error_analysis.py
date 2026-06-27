"""Error analysis: per-document outcome + reconciliation-detection breakdown."""

from __future__ import annotations

import json
from typing import Dict

from ..config import AppConfig, run_dir
from ..logging_utils import get_logger, utc_stamp
from ..training.evaluate import _GOLD

logger = get_logger(__name__)


def error_analysis(cfg: AppConfig) -> Dict:
    from ..agent.invoice_agent import InvoiceAgent
    from ..data.samples import SAMPLE_DOCS

    cfg.serving.log_extractions = False
    agent = InvoiceAgent(cfg, load_model=False)
    categories = {"correct_reconcile": [], "wrong_reconcile": [], "field_error": [], "needs_review": []}
    for name, d in SAMPLE_DOCS.items():
        gold = _GOLD.get(name, {})
        sd = agent.process_tokens(d["tokens"], filename=name).to_dict()
        if sd["status"] == "needs_review":
            categories["needs_review"].append(name)
        if sd["validation"] and "reconciles" in gold:
            if sd["validation"]["reconciles"] == gold["reconciles"]:
                categories["correct_reconcile"].append(name)
            else:
                categories["wrong_reconcile"].append(name)
        for k in ("subtotal", "tax", "total"):
            if k in gold:
                got = sd["fields"].get(k, {}).get("value")
                if got is None or abs(float(got) - gold[k]) >= 0.01:
                    categories["field_error"].append(f"{name}:{k}")
    result = {"n_docs": len(SAMPLE_DOCS),
              "counts": {k: len(v) for k, v in categories.items()}, "details": categories}
    out = run_dir() / "error_analysis"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"error-analysis-{utc_stamp()}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("Error analysis: %s", result["counts"])
    return result


__all__ = ["error_analysis"]
