"""Model versioning + metadata utilities."""

from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from ..logging_utils import get_logger, utc_now_iso

logger = get_logger(__name__)

METADATA_FILE = "model_metadata.json"
LABELS_FILE = "labels.json"


def resolve_latest(base_dir: str | Path) -> Path:
    base = Path(base_dir)
    latest = base / "latest"
    if latest.exists():
        return latest
    if base.exists():
        subs = sorted([p for p in base.iterdir() if p.is_dir()], key=lambda p: p.name)
        if subs:
            return subs[-1]
    return base


def has_model(model_dir: str | Path) -> bool:
    d = Path(model_dir)
    return any((d / f).exists() for f in ("config.json", "model.safetensors", "pytorch_model.bin"))


def _ver(name: str) -> Optional[str]:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


def git_sha() -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def environment_snapshot() -> Dict[str, Any]:
    return {"python": platform.python_version(), "platform": platform.platform(),
            "torch": _ver("torch"), "transformers": _ver("transformers"),
            "datasets": _ver("datasets"), "git_sha": git_sha()}


def make_version_tag(family: str) -> str:
    from datetime import datetime, timezone
    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    sha = git_sha() or "nogit"
    return f"{family}-{date}-{sha}"


def save_model_metadata(model_dir, *, base_model, task, config_subset, dataset_info,
                        metrics=None, id2label=None, version=None) -> Path:
    d = Path(model_dir)
    d.mkdir(parents=True, exist_ok=True)
    meta = {"created_at": utc_now_iso(), "task": task, "base_model": base_model,
            "version": version or make_version_tag("layoutlmv3-kie"),
            "config": config_subset, "dataset": dataset_info, "metrics": metrics or {},
            "environment": environment_snapshot()}
    (d / METADATA_FILE).write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    if id2label is not None:
        (d / LABELS_FILE).write_text(json.dumps({"id2label": {str(k): v for k, v in id2label.items()}}, indent=2),
                                     encoding="utf-8")
    logger.info("Wrote model metadata -> %s", d / METADATA_FILE)
    return d / METADATA_FILE


def load_model_metadata(model_dir) -> Dict[str, Any]:
    p = Path(model_dir) / METADATA_FILE
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def read_labels(model_dir):
    p = Path(model_dir) / LABELS_FILE
    if not p.exists():
        return None
    payload = json.loads(p.read_text(encoding="utf-8"))
    return {int(k): v for k, v in payload["id2label"].items()}


__all__ = ["resolve_latest", "has_model", "save_model_metadata", "load_model_metadata",
           "read_labels", "environment_snapshot", "git_sha", "make_version_tag",
           "METADATA_FILE", "LABELS_FILE"]
