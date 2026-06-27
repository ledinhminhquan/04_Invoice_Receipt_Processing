"""Lightweight hyperparameter tuning for the LayoutLMv3 KIE model.

A small Optuna sweep over learning-rate / weight-decay scored by validation
entity-F1 (skipped gracefully if Optuna is absent). Writes the best params to a
JSON the trainer can read.
"""

from __future__ import annotations

import copy
import json
from typing import Dict, Optional

from ..config import AppConfig, run_dir
from ..logging_utils import get_logger, utc_stamp

logger = get_logger(__name__)


def tune_layoutlmv3(cfg: AppConfig, n_trials: int = 5, which: str = "sroie", limit: Optional[int] = 300) -> Dict:
    try:
        import optuna
    except Exception:
        logger.warning("Optuna not installed; skipping tuning.")
        return {"skipped": True}

    from .train_layoutlmv3 import train_layoutlmv3

    def objective(trial):
        tcfg = copy.deepcopy(cfg)
        tcfg.model.learning_rate = trial.suggest_float("learning_rate", 1e-5, 8e-5, log=True)
        tcfg.model.weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)
        tcfg.model.num_train_epochs = 3
        tcfg.model.output_subdir = f"layout_tune/trial_{trial.number}"
        res = train_layoutlmv3(tcfg, which=which, limit=limit)
        return res.get("val_f1") or 0.0

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    result = {"best_value_f1": study.best_value, "best_params": study.best_params}
    out = run_dir() / f"tune-{utc_stamp()}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("Best LayoutLMv3 params: %s (f1=%.4f)", study.best_params, study.best_value)
    return result


__all__ = ["tune_layoutlmv3"]
