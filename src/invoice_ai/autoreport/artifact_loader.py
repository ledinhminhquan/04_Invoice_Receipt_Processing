"""Gather the latest run artifacts for the autoreport."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from ..config import model_dir, run_dir
from ..logging_utils import get_logger
from ..models.model_registry import load_model_metadata, resolve_latest

logger = get_logger(__name__)


def load_latest(kind: str) -> Optional[dict]:
    """kind in {eval, benchmarks, error_analysis}."""
    base = run_dir()
    dirs = sorted(base.glob(f"{kind}-*"), key=lambda p: p.name, reverse=True)
    for d in dirs:
        if d.is_dir():
            for f in d.glob("*.json"):
                try:
                    return json.loads(f.read_text(encoding="utf-8"))
                except Exception:
                    continue
        elif d.suffix == ".json":
            try:
                return json.loads(d.read_text(encoding="utf-8"))
            except Exception:
                continue
    # flat files (benchmark-*.json, error-analysis-*.json under a subdir)
    sub = base / kind
    if sub.exists():
        files = sorted(sub.glob("*.json"), reverse=True)
        if files:
            try:
                return json.loads(files[0].read_text(encoding="utf-8"))
            except Exception:
                return None
    return None


def load_all_artifacts() -> Dict[str, Optional[dict]]:
    arts = {
        "eval": load_latest("eval"),
        "benchmark": load_latest("benchmarks"),
        "error_analysis": load_latest("error_analysis"),
        "model_meta": None,
    }
    try:
        arts["model_meta"] = load_model_metadata(resolve_latest(model_dir() / "layout_extractor")) or None
    except Exception:
        pass
    present = [k for k, v in arts.items() if v]
    logger.info("Collected %d/%d artifacts (present: %s)", len(present), len(arts), present or "none")
    return arts


__all__ = ["load_latest", "load_all_artifacts"]
