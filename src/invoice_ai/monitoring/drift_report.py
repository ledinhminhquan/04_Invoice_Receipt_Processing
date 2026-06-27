"""Monitoring: aggregate extraction logs + simple drift (PSI) on outcomes."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Optional

from ..config import AppConfig, run_dir
from ..logging_utils import get_logger, utc_now_iso, utc_stamp

logger = get_logger(__name__)


def _read(path: Path):
    if not path or not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def monitoring_report(cfg: AppConfig, log_path: Optional[str] = None) -> Dict:
    path = Path(log_path) if log_path else cfg.serving.extraction_log_path
    recs = _read(path)
    if not recs:
        logger.info("no extraction logs at %s", path)
        return {"note": "no logs", "generated_at": utc_now_iso(), "total": 0}
    status = Counter(r.get("status") for r in recs)
    dtype = Counter(r.get("doc_type") for r in recs)
    confs = [r.get("overall_confidence", 0.0) for r in recs if isinstance(r.get("overall_confidence"), (int, float))]
    review = sum(1 for r in recs if r.get("needs_review"))
    result = {
        "generated_at": utc_now_iso(), "total": len(recs),
        "status_distribution": dict(status), "doc_type_distribution": dict(dtype),
        "needs_review_rate": round(review / len(recs), 4),
        "mean_confidence": round(sum(confs) / len(confs), 4) if confs else None,
    }
    out = run_dir() / "monitoring"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"monitoring-{utc_stamp()}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def drift_report(cfg: AppConfig, reference_path: str, current_path: str) -> Dict:
    import numpy as np

    def dist(recs):
        c = Counter(r.get("status") for r in recs)
        keys = ["auto_approved", "needs_review", "failed"]
        tot = sum(c.values()) or 1
        return np.array([c.get(k, 0) / tot for k in keys]) + 1e-6

    ref = dist(_read(Path(reference_path)))
    cur = dist(_read(Path(current_path)))
    psi = float(np.sum((cur - ref) * np.log(cur / ref)))
    return {"psi": round(psi, 4), "drift": psi > 0.2}


__all__ = ["monitoring_report", "drift_report"]
