"""Latency benchmark for the extraction agent (p50/p95/p99)."""

from __future__ import annotations

import json
import time
from typing import Dict

from ..config import AppConfig, run_dir
from ..logging_utils import get_logger, utc_stamp

logger = get_logger(__name__)


def benchmark(cfg: AppConfig, n: int = 50, warmup: int = 5) -> Dict:
    import numpy as np
    from ..agent.invoice_agent import InvoiceAgent
    from ..data.samples import SAMPLE_DOCS

    cfg.serving.log_extractions = False
    agent = InvoiceAgent(cfg, load_model=False)
    docs = list(SAMPLE_DOCS.values())
    for i in range(warmup):
        agent.process_tokens(docs[i % len(docs)]["tokens"])
    lat = []
    for i in range(n):
        t0 = time.perf_counter()
        agent.process_tokens(docs[i % len(docs)]["tokens"])
        lat.append((time.perf_counter() - t0) * 1000)
    arr = np.array(lat)
    result = {"kind": "extract_agent", "n": n, "warmup": warmup,
              "p50_ms": round(float(np.percentile(arr, 50)), 2),
              "p95_ms": round(float(np.percentile(arr, 95)), 2),
              "p99_ms": round(float(np.percentile(arr, 99)), 2),
              "mean_ms": round(float(arr.mean()), 2),
              "throughput_per_s": round(1000.0 / max(0.01, float(arr.mean())), 2),
              "model_versions": {"extractor": "regex-1.0.0"}}
    out = run_dir() / "benchmarks"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"benchmark-{utc_stamp()}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("Benchmark: p50=%.1fms p95=%.1fms", result["p50_ms"], result["p95_ms"])
    return result


__all__ = ["benchmark"]
