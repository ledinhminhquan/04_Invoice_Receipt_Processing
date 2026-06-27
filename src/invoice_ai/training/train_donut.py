"""(Optional) Fine-tune Donut (OCR-free image→JSON) on CORD for line-items.

Donut shines where token-classification struggles: nested line-item tables. This
is the OCR-free alternative path (``VisionEncoderDecoderModel``). bf16 only
(generation), resume-safe. Saves to ``models/donut/latest``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Optional

from ..config import AppConfig
from ..logging_utils import get_logger
from ..models.model_registry import make_version_tag, save_model_metadata

logger = get_logger(__name__)


def _json_to_tokens(gt: str) -> str:
    """Flatten a CORD ground_truth JSON into a Donut target token string."""
    try:
        data = json.loads(gt)
        parsed = data.get("gt_parse", data)
    except Exception:
        parsed = {}

    def render(obj) -> str:
        if isinstance(obj, dict):
            return "".join(f"<s_{k}>{render(v)}</s_{k}>" for k, v in obj.items())
        if isinstance(obj, list):
            return "<sep/>".join(render(x) for x in obj)
        return str(obj)

    return render(parsed)


def train_donut(cfg: AppConfig, limit: Optional[int] = None) -> Dict:
    import torch
    from datasets import load_dataset
    from transformers import (DonutProcessor, VisionEncoderDecoderModel,
                              Seq2SeqTrainer, Seq2SeqTrainingArguments, get_last_checkpoint)

    raw = load_dataset(cfg.data.cord_dataset)
    train = raw["train"]
    if limit:
        train = train.select(range(min(limit, len(train))))

    processor = DonutProcessor.from_pretrained(cfg.model.donut_model)
    model = VisionEncoderDecoderModel.from_pretrained(cfg.model.donut_model)
    # add field keys as special tokens (collected from a sample of targets)
    keys = set()
    for ex in train.select(range(min(200, len(train)))):
        keys.update(re.findall(r"<s_([a-zA-Z0-9_.]+)>", _json_to_tokens(ex["ground_truth"])))
    new_tokens = [f"<s_{k}>" for k in keys] + [f"</s_{k}>" for k in keys] + ["<sep/>", "<s_cord>"]
    processor.tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
    model.decoder.resize_token_embeddings(len(processor.tokenizer))
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids("<s_cord>")
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    processor.image_processor.size = {"height": 1280, "width": 960}

    max_len = 768

    def preprocess(ex):
        pixel = processor(ex["image"].convert("RGB"), return_tensors="pt").pixel_values[0]
        target = "<s_cord>" + _json_to_tokens(ex["ground_truth"]) + processor.tokenizer.eos_token
        labels = processor.tokenizer(target, add_special_tokens=False, max_length=max_len,
                                     padding="max_length", truncation=True).input_ids
        labels = [l if l != processor.tokenizer.pad_token_id else -100 for l in labels]
        return {"pixel_values": pixel, "labels": labels}

    train_ds = train.map(preprocess, remove_columns=train.column_names)
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    ckpt = Path(cfg.model.output_dir).parent / "donut" / "_ckpt"
    args = Seq2SeqTrainingArguments(
        output_dir=str(ckpt), num_train_epochs=20, learning_rate=2e-5,
        per_device_train_batch_size=2, gradient_accumulation_steps=4, weight_decay=0.01,
        bf16=bf16, fp16=False, predict_with_generate=True, generation_max_length=max_len,
        save_strategy="epoch", save_total_limit=2, logging_steps=50, report_to="none")
    trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=train_ds, tokenizer=processor.feature_extractor)
    last = get_last_checkpoint(str(ckpt)) if any(ckpt.glob("checkpoint-*")) else None
    logger.info("Training Donut on CORD (bf16=%s) | %d examples", bf16, len(train_ds))
    trainer.train(resume_from_checkpoint=last)

    final = Path(cfg.model.output_dir).parent / "donut" / "latest"
    trainer.save_model(str(final))
    processor.save_pretrained(str(final))
    save_model_metadata(final, base_model=cfg.model.donut_model, task="donut-kie",
                        config_subset={"epochs": 20, "image": "1280x960"},
                        dataset_info={"dataset": cfg.data.cord_dataset, "train": len(train_ds)},
                        version=make_version_tag("donut-cord"))
    logger.info("Saved Donut model -> %s", final)
    return {"model_dir": str(final), "n_examples": len(train_ds)}


__all__ = ["train_donut"]
