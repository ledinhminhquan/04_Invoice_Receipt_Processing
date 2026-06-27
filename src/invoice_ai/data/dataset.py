"""Dataset loading for LayoutLMv3 token-classification training.

Loads the verified KIE datasets (SROIE / FUNSD) which ship image + words +
bounding boxes + per-token NER labels — drop-in for ``LayoutLMv3Processor``.
No large data is committed; everything streams from the HF cache.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ..config import DataConfig
from ..logging_utils import get_logger

logger = get_logger(__name__)


def _label_names(ds) -> List[str]:
    feat = ds.features.get("ner_tags")
    try:
        return feat.feature.names  # ClassLabel sequence
    except Exception:
        # fall back to scanning string labels
        uniq = sorted({t for row in ds["ner_tags"] for t in row})
        return [str(u) for u in uniq]


def _words_col(ds) -> str:
    for c in ("tokens", "words"):
        if c in ds.column_names:
            return c
    raise KeyError(f"No words/tokens column in {ds.column_names}")


def load_kie_dataset(cfg: DataConfig, which: str = "sroie", limit: Optional[int] = None):
    """Return (DatasetDict-like dict, id2label, label2id, words_col) for a KIE set."""
    from datasets import load_dataset

    name = {"sroie": cfg.sroie_dataset, "funsd": cfg.funsd_dataset}[which]
    logger.info("Loading KIE dataset: %s", name)
    raw = load_dataset(name)

    train = raw["train"]
    if limit:
        train = train.select(range(min(limit, len(train))))
    test = raw.get("test") or raw.get("validation")

    # carve a validation split from train (SROIE has none)
    if "validation" in raw:
        val = raw["validation"]
    else:
        split = train.train_test_split(test_size=cfg.val_size, seed=cfg.seed)
        train, val = split["train"], split["test"]

    labels = _label_names(raw["train"])
    id2label = {i: l for i, l in enumerate(labels)}
    label2id = {l: i for i, l in id2label.items()}
    words_col = _words_col(raw["train"])
    logger.info("%s: train=%d val=%d test=%d | %d labels", which, len(train), len(val),
                len(test) if test else 0, len(labels))
    return {"train": train, "validation": val, "test": test}, id2label, label2id, words_col


__all__ = ["load_kie_dataset"]
