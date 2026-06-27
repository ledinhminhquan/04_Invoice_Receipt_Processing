"""Matplotlib chart builders for the autoreport (Agg, lazy import, never raise)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from ..config import artifacts_dir
from ..logging_utils import get_logger

logger = get_logger(__name__)


def _fig_dir(out_dir: Optional[str]) -> Path:
    d = Path(out_dir) if out_dir else artifacts_dir() / "submission" / "_figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def metrics_chart(artifacts: Dict, out_dir: Optional[str] = None) -> Optional[Path]:
    ev = (artifacts or {}).get("eval") or {}
    summary = ev.get("summary") or {}
    items = [(k, v) for k, v in summary.items() if isinstance(v, (int, float))]
    if not items:
        logger.info("metrics_chart: no numeric eval summary; skipping.")
        return None
    try:
        plt = _plt()
        labels = [k for k, _ in items]
        vals = [v for _, v in items]
        fig, ax = plt.subplots(figsize=(7, 3.6))
        ax.bar(labels, vals, color="#2e6ff2")
        ax.set_ylim(0, 1.0)
        ax.set_title("Extraction quality metrics")
        ax.set_ylabel("score")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        p = _fig_dir(out_dir) / "metrics.png"
        fig.savefig(p, dpi=130); plt.close(fig)
        return p
    except Exception as exc:
        logger.warning("metrics_chart failed: %s", exc)
        return None


def latency_chart(artifacts: Dict, out_dir: Optional[str] = None) -> Optional[Path]:
    b = (artifacts or {}).get("benchmark") or {}
    keys = [k for k in ("p50_ms", "p95_ms", "p99_ms") if k in b]
    if not keys:
        return None
    try:
        plt = _plt()
        fig, ax = plt.subplots(figsize=(5, 3.4))
        ax.bar(keys, [b[k] for k in keys], color="#1f2a44")
        ax.set_title("Agent latency"); ax.set_ylabel("ms")
        plt.tight_layout()
        p = _fig_dir(out_dir) / "latency.png"
        fig.savefig(p, dpi=130); plt.close(fig)
        return p
    except Exception as exc:
        logger.warning("latency_chart failed: %s", exc)
        return None


def build_all_charts(artifacts: Dict, out_dir: Optional[str] = None) -> Dict[str, Path]:
    out = {}
    for name, fn in (("metrics", metrics_chart), ("latency", latency_chart)):
        p = fn(artifacts, out_dir)
        if p:
            out[name] = p
    logger.info("build_all_charts produced %d charts", len(out))
    return out


__all__ = ["metrics_chart", "latency_chart", "build_all_charts"]
