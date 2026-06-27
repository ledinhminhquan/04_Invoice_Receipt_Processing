"""Download datasets (no large data committed; everything goes to the HF cache)."""

from __future__ import annotations

from typing import Optional

from ..config import AppConfig, ensure_dirs
from ..logging_utils import get_logger

logger = get_logger(__name__)


def download_task(task: str, cfg: Optional[AppConfig] = None) -> dict:
    from datasets import load_dataset

    ensure_dirs()
    cfg = cfg or AppConfig()
    ids = {"sroie": cfg.data.sroie_dataset, "cord": cfg.data.cord_dataset,
           "funsd": cfg.data.funsd_dataset, "invoices": cfg.data.invoices_dataset}
    if task not in ids:
        raise ValueError(f"Unknown task: {task} (choose from {list(ids)})")
    ds = load_dataset(ids[task])
    return {task: {s: len(ds[s]) for s in ds}, "dataset": ids[task]}


def download_all(cfg: Optional[AppConfig] = None) -> dict:
    cfg = cfg or AppConfig()
    out = {}
    for task in ("sroie", "cord", "funsd"):
        try:
            out.update(download_task(task, cfg))
        except Exception as exc:
            logger.warning("Download failed for %s: %s", task, exc)
            out[task] = {"error": str(exc)}
    return out


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["all", "sroie", "cord", "funsd", "invoices"], default="all")
    a = ap.parse_args()
    print(json.dumps(download_all() if a.task == "all" else download_task(a.task), indent=2))
