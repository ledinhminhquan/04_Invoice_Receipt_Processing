"""Typed configuration + YAML loader for the Invoice AI system.

Single source of truth for datasets, models, OCR, agent thresholds and serving.
Paths come from environment variables so nothing is hard-coded.

Environment overrides
---------------------
* ``INVOICE_AI_ARTIFACTS_DIR`` – base for data/models/runs (Drive on Colab)
* ``INVOICE_AI_MODEL_DIR``     – trained models
* ``INVOICE_AI_RUN_DIR``       – eval/benchmark/analysis JSON
* ``HF_HOME``                  – HuggingFace cache
* ``INVOICE_AI_LLM_API_KEY``   – optional key for the LLM-vision fallback brain
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(key)
    return v if v not in (None, "") else default


def artifacts_dir() -> Path:
    return Path(_env("INVOICE_AI_ARTIFACTS_DIR", "artifacts")).expanduser()


def data_dir() -> Path:
    return Path(_env("INVOICE_AI_DATA_DIR", str(artifacts_dir() / "data"))).expanduser()


def model_dir() -> Path:
    return Path(_env("INVOICE_AI_MODEL_DIR", str(artifacts_dir() / "models"))).expanduser()


def run_dir() -> Path:
    return Path(_env("INVOICE_AI_RUN_DIR", str(artifacts_dir() / "runs"))).expanduser()


# ─────────────────────────────────────────────────────────────────────────────
# Sub-configs
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DataConfig:
    """Verified public datasets (see docs/data_description.md)."""
    sroie_dataset: str = "mp-02/sroie"                 # LayoutLMv3-ready (image+words+bbox+ner)
    cord_dataset: str = "naver-clova-ix/cord-v2"       # Donut (image + gt_parse JSON), cc-by-4.0
    funsd_dataset: str = "nielsr/funsd-layoutlmv3"     # form KIE BIO 7-class
    invoices_dataset: str = "katanaml-org/invoices-donut-data-v1"  # Donut invoices, MIT
    val_size: float = 0.1
    seed: int = 42


@dataclass
class ModelConfig:
    """Layout extractor + Donut + baseline."""
    # Primary KIE extractor. layoutlmv3-base = best accuracy but CC-BY-NC-SA (non-commercial);
    # switch to the MIT LiLT model for commercial deployments (same token-cls pipeline).
    layout_model: str = "microsoft/layoutlmv3-base"
    layout_model_commercial: str = "SCUT-DLVCLab/lilt-roberta-en-base"  # MIT
    donut_model: str = "naver-clova-ix/donut-base"
    donut_cord: str = "naver-clova-ix/donut-base-finetuned-cord-v2"     # ready-to-run receipts
    baseline_encoder: str = "google-bert/bert-base-uncased"
    max_length: int = 512
    doc_stride: int = 128
    apply_ocr: bool = False           # we supply our own OCR words+boxes (recommended)
    # training (LayoutLMv3 token classification)
    num_train_epochs: int = 8
    learning_rate: float = 5.0e-5
    per_device_train_batch_size: int = 8
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    label_smoothing_factor: float = 0.1
    early_stopping_patience: int = 4
    bf16: bool = True
    fp16: bool = False
    seed: int = 42
    output_subdir: str = "layout_extractor"

    @property
    def output_dir(self) -> Path:
        return model_dir() / self.output_subdir


@dataclass
class OcrConfig:
    """OCR engine selection. native_pdf (born-digital) > paddle/tesseract (scanned)."""
    engine: str = "auto"              # "auto" | "tesseract" | "paddle" | "native"
    fallback_engine: str = "tesseract"
    lang: str = "eng"
    dpi: int = 250                    # rasterise scanned PDFs at 200-300 DPI
    min_token_conf: float = 0.0       # keep all tokens; conf used for gating


@dataclass
class AgentConfig:
    """Agent thresholds + optional LLM-vision fallback."""
    quality_min: float = 0.45         # Q_MIN — scan-quality gate
    ocr_min: float = 0.70             # OCR_MIN — mean OCR confidence gate
    field_conf_min: float = 0.80      # FIELD_CONF_MIN — per-field acceptance
    auto_approve_min: float = 0.85    # AUTO_MIN — auto-approve confidence
    max_ocr_attempts: int = 2
    reconcile_eps_abs: float = 0.01   # absolute epsilon floor
    reconcile_eps_rel: float = 0.005  # relative epsilon (of total)
    # optional LLM-vision brain (the only cloud tool; off by default)
    llm_fallback_enabled: bool = False
    llm_model: str = "claude-haiku-4-5-20251001"
    llm_api_key_env: str = "INVOICE_AI_LLM_API_KEY"


@dataclass
class ServingConfig:
    model_version: str = "v1"
    api_title: str = "Invoice & Receipt Processing API"
    api_version: str = "1.0.0"
    log_extractions: bool = True
    extraction_log_subdir: str = "extraction_logs"
    review_queue_subdir: str = "review_queue"
    max_file_mb: int = 20
    max_pages: int = 15

    @property
    def extraction_log_path(self) -> Path:
        return run_dir() / self.extraction_log_subdir / "extractions.jsonl"

    @property
    def review_queue_path(self) -> Path:
        return run_dir() / self.review_queue_subdir / "queue.jsonl"


@dataclass
class AppConfig:
    project_title: str = "Invoice & Receipt Processing System"
    author: str = "Le Dinh Minh Quan"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ocr: OcrConfig = field(default_factory=OcrConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    serving: ServingConfig = field(default_factory=ServingConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_SECTIONS = {"data": DataConfig, "model": ModelConfig, "ocr": OcrConfig,
             "agent": AgentConfig, "serving": ServingConfig}


def _build(cls, raw: Optional[Dict[str, Any]]):
    raw = raw or {}
    known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    return cls(**{k: v for k, v in raw.items() if k in known})


def load_config(path: Optional[str | os.PathLike] = None) -> AppConfig:
    raw: Dict[str, Any] = {}
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config not found: {p}")
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    top = {k: raw[k] for k in ("project_title", "author") if k in raw}
    sections = {name: _build(cls, raw.get(name)) for name, cls in _SECTIONS.items()}
    return AppConfig(**top, **sections)


def save_config(cfg: AppConfig, path: str | os.PathLike) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False, allow_unicode=True), encoding="utf-8")


def ensure_dirs() -> Dict[str, Path]:
    dirs = {"artifacts": artifacts_dir(), "data": data_dir(), "models": model_dir(), "runs": run_dir()}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


__all__ = ["DataConfig", "ModelConfig", "OcrConfig", "AgentConfig", "ServingConfig", "AppConfig",
           "load_config", "save_config", "ensure_dirs", "artifacts_dir", "data_dir", "model_dir", "run_dir"]
