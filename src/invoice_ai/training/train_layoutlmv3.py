"""Fine-tune LayoutLMv3 for token-classification KIE (resume-safe).

Recipe (see docs/model_selection.md): ``LayoutLMv3Processor(apply_ocr=False)`` with
words + **0-1000-normalised boxes** + label alignment (continuation subwords →
``-100``), bf16+TF32 on H100, entity-level **seqeval** P/R/F1, early stopping.
Saves to ``models/layout_extractor/latest`` with ``labels.json`` + metadata.

Note: ``microsoft/layoutlmv3-base`` is CC-BY-NC-SA (non-commercial); for
commercial use set ``model.layout_model`` to the MIT ``SCUT-DLVCLab/lilt-...``.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Dict, Optional

from ..config import AppConfig
from ..data.dataset import load_kie_dataset
from ..logging_utils import get_logger
from ..models.model_registry import make_version_tag, save_model_metadata

logger = get_logger(__name__)


def _normalize_boxes(boxes, width, height):
    if not boxes:
        return boxes
    mx = max((max(b) for b in boxes), default=0)
    if mx <= 1000:            # already normalised
        return [[min(1000, max(0, int(v))) for v in b] for b in boxes]
    w, h = max(1, width), max(1, height)
    return [[min(1000, max(0, int(1000 * b[0] / w))), min(1000, max(0, int(1000 * b[1] / h))),
             min(1000, max(0, int(1000 * b[2] / w))), min(1000, max(0, int(1000 * b[3] / h)))] for b in boxes]


def _build_args(mcfg, output_dir, bf16, fp16):
    from transformers import TrainingArguments
    sig = set(inspect.signature(TrainingArguments.__init__).parameters)
    eval_key = "eval_strategy" if "eval_strategy" in sig else "evaluation_strategy"
    kw = dict(output_dir=str(output_dir), learning_rate=mcfg.learning_rate,
              num_train_epochs=mcfg.num_train_epochs, per_device_train_batch_size=mcfg.per_device_train_batch_size,
              per_device_eval_batch_size=mcfg.per_device_train_batch_size, warmup_ratio=mcfg.warmup_ratio,
              weight_decay=mcfg.weight_decay, lr_scheduler_type="cosine", save_strategy="epoch",
              save_total_limit=3, load_best_model_at_end=True, metric_for_best_model="f1", greater_is_better=True,
              logging_steps=20, seed=mcfg.seed, bf16=bf16, fp16=fp16, report_to="none", remove_unused_columns=False)
    kw[eval_key] = "epoch"
    return TrainingArguments(**{k: v for k, v in kw.items() if k in sig})


def train_layoutlmv3(cfg: AppConfig, which: str = "sroie", limit: Optional[int] = None) -> Dict:
    import numpy as np
    import torch
    from transformers import (AutoModelForTokenClassification, AutoProcessor,
                              EarlyStoppingCallback, Trainer, get_last_checkpoint)
    import evaluate as hf_evaluate

    mcfg = cfg.model
    splits, id2label, label2id, words_col = load_kie_dataset(cfg.data, which=which, limit=limit)
    labels = [id2label[i] for i in range(len(id2label))]
    processor = AutoProcessor.from_pretrained(mcfg.layout_model, apply_ocr=False)

    def preprocess(examples):
        images = [img.convert("RGB") for img in examples["image"]]
        boxes = [_normalize_boxes(b, im.size[0], im.size[1]) for b, im in zip(examples["bboxes"], images)]
        enc = processor(images, examples[words_col], boxes=boxes, word_labels=examples["ner_tags"],
                        truncation=True, padding="max_length", max_length=mcfg.max_length, return_tensors="pt")
        return enc

    cols = splits["train"].column_names
    train_ds = splits["train"].map(preprocess, batched=True, remove_columns=cols)
    val_ds = splits["validation"].map(preprocess, batched=True, remove_columns=cols)

    model = AutoModelForTokenClassification.from_pretrained(
        mcfg.layout_model, num_labels=len(labels), id2label=id2label, label2id=label2id)

    seqeval = hf_evaluate.load("seqeval")

    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=2)
        true_pred = [[id2label[a] for a, l in zip(pr, lab) if l != -100] for pr, lab in zip(preds, p.label_ids)]
        true_lab = [[id2label[l] for a, l in zip(pr, lab) if l != -100] for pr, lab in zip(preds, p.label_ids)]
        r = seqeval.compute(predictions=true_pred, references=true_lab, zero_division=0)
        return {"precision": r["overall_precision"], "recall": r["overall_recall"],
                "f1": r["overall_f1"], "accuracy": r["overall_accuracy"]}

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported() and mcfg.bf16
    fp16 = (not bf16) and torch.cuda.is_available() and mcfg.fp16
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    ckpt_dir = Path(mcfg.output_dir) / "_ckpt"
    args = _build_args(mcfg, ckpt_dir, bf16, fp16)
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                      compute_metrics=compute_metrics,
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=mcfg.early_stopping_patience)])
    try:
        trainer.processing_class = processor
    except Exception:
        pass

    last = get_last_checkpoint(str(ckpt_dir)) if any(ckpt_dir.glob("checkpoint-*")) else None
    logger.info("Training LayoutLMv3 (%s) | %d labels | bf16=%s", mcfg.layout_model, len(labels), bf16)
    trainer.train(resume_from_checkpoint=last)

    val_metrics = trainer.evaluate()
    final = Path(mcfg.output_dir) / "latest"
    trainer.save_model(str(final))
    processor.save_pretrained(str(final))
    save_model_metadata(final, base_model=mcfg.layout_model, task="kie-token-classification",
                        config_subset={"epochs": mcfg.num_train_epochs, "lr": mcfg.learning_rate,
                                       "max_length": mcfg.max_length, "dataset": which},
                        dataset_info={"dataset": which, "train": len(train_ds), "labels": labels},
                        metrics={"val": val_metrics}, id2label=id2label, version=make_version_tag("layoutlmv3-kie"))
    logger.info("Saved LayoutLMv3 KIE model -> %s | val f1=%.4f", final, val_metrics.get("eval_f1", 0))
    return {"model_dir": str(final), "val_f1": val_metrics.get("eval_f1"), "labels": labels}


__all__ = ["train_layoutlmv3"]
