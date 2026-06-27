# Model Selection & Optimization

*Project #4 — Invoice & Receipt Processing System · package `invoice_ai` · author Le Dinh Minh Quan (23127460)*

This document satisfies Assignment §5. It specifies the architecture of every model component, justifies each choice against a concrete business need, fixes the training procedure and hyperparameters, lays out the baseline-comparison plan with a clearly-labelled *projected* results table, and closes with error analysis and engineering trade-offs.

---

## 1. Component architectures

The extraction stack is a **layered ensemble of one fixed floor, one diagnostic baseline, and two complementary deep models**, all fed by an OCR/PDF front-end. Each component is independently swappable behind the agent's `extract_layout` / `extract_line_items` tools.

| Component | HF id | Params | License | Architecture / role |
|---|---|---|---|---|
| Layout extractor (**primary accuracy**) | `microsoft/layoutlmv3-base` | 125.3M | cc-by-nc-sa-4.0 ⚠️ NC | LayoutLMv3 token-classification; 2D-layout + text + image patches; flat header fields |
| Layout extractor (**commercial**) | `SCUT-DLVCLab/lilt-roberta-en-base` | 130.8M | MIT ✅ | LiLT token-classification; language-agnostic layout stream, same OCR-token pipeline |
| Line-item / OCR-free | `naver-clova-ix/donut-base-finetuned-cord-v2` | ~200M | MIT ✅ | Donut vision-encoder-decoder; image → nested JSON, no OCR |
| Baseline encoder | `google-bert/bert-base-uncased` | 110.1M | apache-2.0 ✅ | BERT + bbox token-classification (layout-pretraining ablation) |
| Floor | — (regex/heuristic) | 0 | — | Deterministic rules over OCR text; mandatory interpretable floor |

**OCR / PDF front-end (shared).** PaddleOCR (Apache-2.0, primary; rotation/dense/multilingual), docTR (best fallback), Tesseract (born-digital / LayoutLMv3 internal default). Born-digital PDFs use the native text layer via `pdfplumber`/PyMuPDF (conf≈1.0, exact boxes); scanned PDFs rasterize at 200–300 DPI via `pdf2image`+Poppler then OCR. All boxes normalize to **0–1000**.

**LayoutLMv3 (extractive).** Tri-modal transformer: OCR token embeddings + 2D positional embeddings (`int(1000*x/w)`) + linear-projected image patches. Emits a label per token (`S-`/BIO over `COMPANY, DATE, ADDRESS, TOTAL, …`). Every prediction is **grounded to an OCR token with a bbox** — it cannot invent text.

**LiLT (extractive, MIT).** Decouples a layout stream from a text stream, so the English RoBERTa text encoder can be swapped for a multilingual one (`nielsr/lilt-xlm-roberta-base`) **without retraining the layout half**. Same token-classification head and OCR-token data pipeline as LayoutLMv3.

**Donut (generative).** OCR-free `VisionEncoderDecoder` reading the page image and autoregressively decoding a JSON token sequence (`<s_total>1364.00</s_total>`, lists joined with `<sep/>`). Excels at **nested line-item tables** where token-classification struggles to group rows; the cost is hallucination risk and no per-token boxes.

```
PDF/img ─┬─ born-digital → pdfplumber (text+boxes)         ┐
         └─ scanned ─────→ pdf2image → PaddleOCR (w+boxes) ┘
                                  │ normalize 0–1000
        ┌─────────────────────────┼──────────────────────────┐
   PRIMARY: LayoutLMv3-base   COMMERCIAL: LiLT-roberta-en   PARALLEL: Donut-cord-v2
   (apply_ocr=False)          (MIT, multilingual swap)      (image→JSON line items)
        └────────── BASELINE FLOOR: regex/heuristic + bert+bbox ──────────┘
```

---

## 2. Justification per business need

| Business need | Chosen component | Why it wins |
|---|---|---|
| Auditable header fields + review-UI highlighting | **LayoutLMv3 / LiLT** | Grounded predictions with per-word **bbox provenance**; reviewers click the source token in 5 s. Donut gives no boxes. |
| Line items / nested tables | **Donut** | Nested-JSON decoding groups rows natively; token-cls struggles to cluster columns into rows. |
| Commercial / multilingual deployment | **LiLT (MIT)** | LayoutLMv3 is **CC-BY-NC-SA-4.0 (non-commercial)** — internal benchmark only. LiLT is MIT and swaps the text stream for non-English invoices. |
| OCR-hostile docs (low-res, stylized, dense) | **Donut** | OCR-free end-to-end; bypasses an OCR failure mode entirely. |
| Interpretable floor + cost control | **regex/heuristic** | Zero training, fully transparent; every model must beat it. |
| Isolating the value of layout-pretraining | **bert+bbox** | License-clean control that measures how much v3/LiLT pretraining buys over plain BERT + coordinates. |

The key safeguard: Donut **hallucinates** fields at low confidence and gives no boxes, so it is restricted to line-items / OCR-hostile docs, and its totals are always **cross-checked against the validation reconciler** (`sum(lines)+tax==total ±ε`). LayoutLMv3/LiLT feed the review UI because they cannot invent text.

---

## 3. Training procedure

### 3.1 LayoutLMv3 / LiLT — token classification (Recipe 1)

- **Processor:** `LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)` — pass precomputed `images, words, boxes, word_labels`. (SROIE/CORD/FUNSD already ship words+boxes.)
- **Bbox normalization (the #1 silent bug):** `int(1000*x/w)` for every coordinate. HF CORD/FUNSD-v3 boxes ship pre-normalized; raw SROIE images must be normalized by hand.
- **Label alignment:** with `word_labels` supplied, the processor sets **continuation subwords + special tokens to `-100`** (only the first subword keeps the label; `CrossEntropyLoss` ignores `-100`). Self-tokenization must replicate this via `word_ids`.
- **Metric:** `evaluate.load("seqeval")` — **entity-level** P/R/F1, reported **per entity** (not just `overall`) so rare-class collapse is visible.
- **Trainer (H100):** `bf16=True, tf32=True`, `learning_rate=5e-5`, `num_train_epochs=8` (early-stopped), `per_device_train_batch_size=8`, `warmup_ratio=0.1`, `weight_decay=0.01`, `lr_scheduler_type="cosine"`, `eval_strategy/save_strategy="epoch"`, `load_best_model_at_end=True`, `metric_for_best_model="f1"`, `seed=42`. Use `processing_class=processor` (the `tokenizer=` arg is deprecated in 4.5x+). **Resume-safe:** `trainer.train(resume_from_checkpoint=True)`.
- Pin `transformers==4.51.*`; enable TF32 via `torch.backends.cuda.matmul.allow_tf32 = True`.

### 3.2 Donut — image → JSON (Recipe 2)

- `DonutProcessor` + `VisionEncoderDecoderModel` from `naver-clova-ix/donut-base`; **add all field keys as special tokens** then `resize_token_embeddings`. Set the task-start token (`<s_cord>`/`<s_invoice>`), `decoder_start_token_id`, `pad_token_id`. Receipts are tall → `size={"height":1280,"width":960}`, `max_length=768`.
- **`Seq2SeqTrainer` + `Seq2SeqTrainingArguments`** with `predict_with_generate=True`, `generation_max_length=max_length`, **`bf16=True, tf32=True`**, lower LR `1e-5..3e-5`, more epochs (20–40) with **hard early-stop**, `per_device_train_batch_size=2` + `gradient_accumulation_steps=4` (eff. 8). Teacher forcing; pad → `-100`.
- **Eval:** normalized tree-edit-distance (nTED) + field-level F1 (per-key exact-match over flattened `{key:value}`).

---

## 4. Hyperparameter tuning & anti-overfitting

Datasets are small (SROIE ~600, CORD ~800, FUNSD ~150), so tuning prioritizes regularization over capacity.

| Knob | LayoutLMv3 / LiLT | Donut |
|---|---|---|
| Learning rate | 3e-5 – 5e-5 | 1e-5 – 3e-5 |
| Epochs | 5–10 (early-stop) | 20–40 (hard early-stop) |
| Effective batch | 8 | 8 (2 × grad-accum 4) |
| Schedule | cosine, warmup 0.1 | cosine, warmup 0.1 |
| Regularization | weight-decay 0.01–0.05, label-smoothing 0.1, `max_grad_norm=1.0` | + `random_padding=True`, geometry-safe aug |
| Early stop | `EarlyStoppingCallback(patience=3–5)` on eval F1 | same, hard |

Additional levers: freeze early encoder layers for FUNSD (~150 rows); **start from in-domain checkpoints** (`donut-base-finetuned-cord-v2`, LayoutLMv3-on-FUNSD) for the fastest small-data win; report **multi-seed mean±std** (FUNSD variance is high at n=150); oversample / weighted `CrossEntropyLoss` for extreme rare classes. Monitor the train-vs-eval F1 gap — stop or add augmentation if eval plateaus while train climbs.

---

## 5. Baseline comparison plan

**Floor → ablation → primary.** Every model must beat the regex floor; the bert+bbox control isolates exactly how much layout-pretraining buys.

1. **Regex/heuristic over OCR text** — totals (`(?:total|amount due|balance)\D{0,10}([$€£]?\s?\d[\d.,]*)`), dates, invoice numbers, positional heuristics. Zero training, fully interpretable.
2. **bert+bbox token-classification** (apache-2.0) — coordinates as features; license-clean middle ground.
3. **Primary:** fine-tuned **LayoutLMv3-base** (accuracy benchmark) and **LiLT** (commercial); **Donut** for line-item / OCR-free comparison.

**Axes:** entity F1 (seqeval) and end-to-end field accuracy on each test split — `mp-02/sroie` (347), `cord-v2` (100), `funsd-layoutlmv3` (50) — plus line-item F1, validation pass-rate, needs-review rate, and p50/p95 latency (GPU vs CPU, reported separately).

### Projected results — **ILLUSTRATIVE TARGETS, NOT MEASURED**

> The numbers below are **projected expectations** stated to define the comparison shape. They are **not experimental results** and will be replaced by measured values after training.

| Model | SROIE entity-F1 | CORD field-F1 | Line-item F1 | License |
|---|---|---|---|---|
| Regex / heuristic (floor) | ~0.55 | ~0.50 | ~0.45 | — |
| BERT + bbox | ~0.78 | ~0.74 | ~0.60 | apache-2.0 |
| LiLT-roberta-en | ~0.92 | ~0.88 | ~0.70 | MIT |
| **LayoutLMv3-base** | **~0.93** | ~0.89 | ~0.72 | cc-by-nc-sa-4.0 |
| Donut (cord-v2) | — | ~0.90 | **~0.82** | MIT |

Expected ordering: **regex ≪ bert+bbox < LiLT ≈ LayoutLMv3-base ≤ LayoutLMv3-large** for flat fields; **Donut > token-cls** for nested line items.

---

## 6. Error analysis

- **Rare-class TOTAL collapse.** `O` dominates and `B-TOTAL` is rare, so overall accuracy can hide a collapsed TOTAL class. Mitigation: report **per-entity** seqeval F1, weighted `CrossEntropyLoss(ignore_index=-100)` or focal loss, oversample docs with rare entities.
- **OCR errors.** Noisy/rotated/low-DPI scans corrupt tokens before the model sees them. Mitigation: native-PDF text first → PaddleOCR → docTR fallback; deskew/denoise/upscale on retry; the `OCR_MIN=0.70` gate routes bad scans to review after `MAX_OCR_ATTEMPTS=2`.
- **Ambiguous dates.** dd/mm vs mm/dd is unrecoverable from glyphs alone. Mitigation: `validate` flags ambiguous dates; `normalize` resolves to ISO-8601 using `dateutil` + locale hints from detected currency/country; unresolved cases go to human review.
- **Donut hallucination.** Plausible-but-wrong fields at low confidence with no boxes. Mitigation: restrict Donut to line-items / OCR-hostile docs and cross-check its totals against the reconciler.

---

## 7. Trade-offs

| Trade-off | Tension | Resolution |
|---|---|---|
| Accuracy vs latency | Donut generate 300–800 ms GPU / 4–10 s CPU vs LayoutLMv3 forward 15–40 ms GPU | LayoutLMv3/LiLT as default low-latency path; **never run Donut on CPU**; Donut reserved for line-items. |
| License vs accuracy | LayoutLMv3 best accuracy but **NC** | LayoutLMv3 = internal benchmark; ship **LiLT (MIT)** or **Donut (MIT)** commercially — shared OCR-token pipeline makes the swap cheap. |
| Extractive vs generative | Grounded boxes (auditable, flat) vs nested JSON (line-items, hallucination-prone) | Use both: extractive for review-facing header fields, generative for tables, reconciler arbitrates. |
| Float vs Decimal money | speed vs financial correctness | `Decimal(str(x))` 2dp + ISO-4217 + `ε = max(0.01, 0.005*total)`; never float. |

**Bottom line.** LayoutLMv3-base is the accuracy north star and internal benchmark; **LiLT (MIT)** is the commercial-safe twin on the same pipeline; **Donut (MIT)** owns line-items and OCR-hostile cases; regex + bert+bbox form the interpretable floor and the layout-pretraining ablation — and every deep prediction is gated by arithmetic reconciliation before it can auto-approve.
