# Model Card (MODEL_CARD)

> Formal model card for the Key-Information-Extraction (KIE) models that power **`invoice_ai`** — the offline-first Invoice & Receipt Processing System (Project #4). Structure follows the Hugging Face model-card convention. Author: **Le Dinh Minh Quan (23127460)**. Card date: **2026-06-26**.
>
> This card documents three interchangeable extractor families that share one OCR-token pipeline: the **LayoutLMv3** primary (accuracy/internal), the **LiLT** commercial-safe variant, and the **Donut** OCR-free line-item variant. A regex/heuristic floor and a `bert-base + bbox` baseline are documented for context but are not the deployed extractors.

---

## 1. Model Details

### 1.1 Overview

| Property | Value |
|---|---|
| Developed by | Le Dinh Minh Quan (23127460), solo project (team roles simulated) |
| Package | `invoice_ai` |
| Model type | Document-AI Key-Information-Extraction (token classification + OCR-free seq2seq) |
| Primary task | Extract validated, normalized structured JSON (header fields + line items) from invoice/receipt images and PDFs, with per-field confidence and bounding boxes |
| Inputs | Document image (PNG/JPG) or PDF, plus OCR words + boxes (for the token-classification path) |
| Outputs | Per-token BIO/`S-` field labels (LayoutLMv3/LiLT) or a serialized JSON token sequence (Donut), de-coded to fields/line items with confidence + pixel bboxes + `needs_review` |
| Language | English (LiLT-XLM-RoBERTa variant available for multilingual swap, 90+ langs) |
| Finetuned from | `microsoft/layoutlmv3-base`, `SCUT-DLVCLab/lilt-roberta-en-base`, `naver-clova-ix/donut-base` |

### 1.2 Architecture & base checkpoints (VERIFIED ids)

| Role | Base checkpoint | Params | License | Arch / Task |
|---|---|---|---|---|
| **Primary (accuracy / internal)** | `microsoft/layoutlmv3-base` | 125.3M | **CC-BY-NC-SA-4.0** (NON-COMMERCIAL) | `layoutlmv3` token-cls |
| Primary (large) | `microsoft/layoutlmv3-large` | ~368M | **CC-BY-NC-SA-4.0** (NON-COMMERCIAL) | `layoutlmv3` token-cls |
| **Commercial-safe** | `SCUT-DLVCLab/lilt-roberta-en-base` | 130.8M | **MIT** | `lilt` token-cls |
| Commercial (multilingual) | `nielsr/lilt-xlm-roberta-base` | 284.2M | **MIT** | `lilt`, 90+ langs |
| **OCR-free / line items** | `naver-clova-ix/donut-base` | ~200M | **MIT** | vision-enc-dec image→text |
| OCR-free (CORD, ready) | `naver-clova-ix/donut-base-finetuned-cord-v2` | ~200M | **MIT** | image→JSON |
| Baseline encoder | `google-bert/bert-base-uncased` (+ bbox features) | 110.1M | **Apache-2.0** | bert+bbox baseline |

> Correction recorded: `microsoft/lilt-roberta-en-base` **does not exist** → the deployed id is `SCUT-DLVCLab/lilt-roberta-en-base`. `microsoft/layoutlmv2-base-uncased` is avoided (requires `detectron2`, NC license, superseded by v3).

### 1.3 Version tag

Every trained artifact is tagged **`{family}-{date}-{git_sha}`** and the tag is echoed in every API response (`model_version`) and in `/metrics` (`model_info{version}`).

```
example: layoutlmv3-kie-2026-06-20-a1b2c3d
```

The loader resolves a semantic alias (`@prod`, `@canary`) to a **pinned revision sha** at boot; `label_map` / `id2label` / processor config are stored alongside the weights.

---

## 2. Intended Use

### 2.1 In-scope use

- **Automated extraction** of header fields (`invoice_number`, `invoice_date`, `issuer/vendor`, `recipient`, `subtotal`, `tax_rate`, `tax`, `total`, `currency`) and line items (`description`, `quantity`, `unit_price`, `amount`) from invoices and receipts.
- **Offline / on-prem** document processing: the default path runs fully local with **no paid API**.
- Feeding a downstream **validate → normalize → auto-approve vs. human-review** agent (the deterministic state machine; the LLM-vision tool is an optional, feature-flagged fallback).
- **Benchmarking** layout-pretraining gains over a `bert+bbox` baseline and a regex floor.

### 2.2 Intended users

Back-office finance/AP automation, document-AI engineers, and reviewers operating a human-in-the-loop review queue.

### 2.3 Out-of-scope / disallowed use

- **Commercial deployment of the LayoutLMv3 variant.** `microsoft/layoutlmv3-base/-large` are **CC-BY-NC-SA-4.0 (non-commercial)**. For any commercial use, ship the **LiLT (MIT)** or **Donut (MIT)** variant instead — they share the same OCR-token pipeline, so the swap is cheap. LayoutLMv3 is the internal accuracy benchmark only.
- **Not a substitute for human audit on flagged documents.** Any document the agent routes to `NEEDS_REVIEW` (failed reconciliation, missing required field, or low confidence) must be confirmed by a human before its values enter a ledger. The model does not certify financial correctness.
- Not a generic document classifier (e.g. RVL-CDIP 16-class) — it targets invoice/receipt/other KIE, not arbitrary document taxonomy.
- Not validated outside English business documents (use the LiLT-XLM variant and re-evaluate before non-English deployment).

---

## 3. Training Data

All datasets were live-verified on the HF Hub on 2026-06-26. No large data is committed; download scripts pull on demand into git-ignored `data/`.

| Dataset | Use | Format | Rows (tr/val/te) | Ready for | License |
|---|---|---|---|---|---|
| **`mp-02/sroie`** | Receipt KIE w/ images | image + words + bbox + ner (5-class `S-`) | 626 / – / 347 | LayoutLMv3 / LiLT | unknown (card) |
| **`naver-clova-ix/cord-v2`** | Receipt line-items JSON | image + `gt_parse` JSON | 800 / 100 / 100 | Donut | **cc-by-4.0** |
| **`nielsr/funsd-layoutlmv3`** | Form KIE (generic) | image + tokens + bbox + ner (7-class BIO) | 149 / – / 50 | LayoutLMv3 / LiLT | unknown |
| **`katanaml-org/invoices-donut-data-v1`** | Invoice JSON | image + gt JSON | 425 / 50 / 26 | Donut | **MIT** |

- **SROIE (`mp-02/sroie`)** ships single-token `S-` tags (`S-COMPANY, S-DATE, S-ADDRESS, S-TOTAL, O`), not BIO; embedded real images. A deterministic 10% val split (`seed=42`) is carved from train.
- **CORD (`cord-v2`)** target = `json.loads(ground_truth)["gt_parse"]`, the standard receipt line-item benchmark.
- **FUNSD (`funsd-layoutlmv3`)** demonstrates generic form KIE; it is **not** the invoice/receipt target.
- **Avoided / nonexistent:** `jinhybr/SROIE…`, `jordyvl/funsd`, `rvl_cdip` (classification, wrong task). `darentang/sroie` avoided for direct image use (`image_path` is a string, not an `Image`).

**Licensing caveat (production):** SROIE/FUNSD base data are ICDAR research-challenge data (non-commercial-leaning) and several mirrors declare no license. Only `cord-v2` (cc-by-4.0), `katanaml-org/invoices-donut-data-v1` (MIT), `arvindrajan92/sroie_document_understanding` (MIT), and `jsdnrs/ICDAR2019-SROIE` (cc-by-4.0) carry explicit permissive licenses. **Clear with legal before commercial deployment.**

### 3.1 Preprocessing (the #1 silent-bug zone)

- **Bbox normalization to 0–1000** (LayoutLMv3 requirement): `x_norm = int(1000 * x_pixel / image_width)`. Getting this wrong is the leading cause of silent accuracy loss. CORD/FUNSD-v3 boxes ship already normalized; SROIE/raw images must be normalized by the loader.
- `LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)` with **precomputed words + boxes**.
- **Label alignment:** the processor sets continuation subwords + special tokens to `-100` (ignored by CrossEntropyLoss); only the first subword keeps the label. Self-tokenization replicates this via `word_ids`.

---

## 4. Training Procedure

| Hyperparameter | LayoutLMv3 / LiLT (token-cls) | Donut (seq2seq) |
|---|---|---|
| Learning rate | `5e-5` (3e-5..5e-5) | `1e-5..3e-5` |
| Epochs | ~8, early-stopped (5–10 range) | 20–40, hard early-stop |
| Precision | **bf16 + tf32** (H100; not fp16) | **bf16 + tf32** |
| Batch | 8 train / 8 eval | 2 + grad-accum 4 (eff. 8) |
| Schedule | `warmup_ratio=0.1`, cosine, `weight_decay=0.01` | same family + label smoothing 0.1 |
| Seed | 42 | 42 |
| Trainer | `Trainer` (`processing_class=processor`, not `tokenizer=`) | `Seq2SeqTrainer`, `predict_with_generate=True` |
| Best-model | `load_best_model_at_end`, `metric_for_best_model="f1"` | nTED / field-F1 |

- Library pinned to **`transformers==4.51.*`** (latest is 5.12.1; 4.4x–4.5x is the stable line for LayoutLMv3/Donut). Training is resume-safe (`resume_from_checkpoint`).
- **Anti-overfitting** (SROIE~600, CORD~800, FUNSD~150): early stopping on eval F1, low LR + warmup + cosine, geometry-safe augmentation, weight decay + label smoothing, freeze early layers for FUNSD, start from in-domain checkpoints, multi-seed mean±std, `max_grad_norm=1.0`.

---

## 5. Evaluation

### 5.1 Metrics

| Metric | Definition | Tool |
|---|---|---|
| Entity P/R/F1 per field | entity-level for COMPANY/DATE/ADDRESS/TOTAL/etc. | **seqeval** (`overall_*` + per-entity) |
| End-to-end field accuracy | post-normalize exact-match per field | custom flatten + exact-match |
| Line-item F1 | per-row P/R/F1 (desc + amount) + per-cell qty/unit_price/amount | custom |
| Validation pass-rate | docs where `reconciles AND required_present` | from `ValidationReport` |
| Needs-review rate | fraction routed to human review, broken down by reason | from agent status |
| Donut nTED / field-F1 | normalized tree-edit-distance + per-key field-F1 | nltk edit_distance + custom |
| Latency | p50/p95 end-to-end + per-stage, GPU vs CPU | Prometheus histograms |

**Reporting rule:** always report **per-entity** F1 (not just `overall`) so rare-class collapse (e.g. `TOTAL`) is visible; report needs-review rate by reason; report latency p95 separately for GPU and CPU.

### 5.2 Test splits

`mp-02/sroie` test **347**, `cord-v2` test **100**, `funsd-layoutlmv3` test **50**.

### 5.3 Results

> **PROJECTED — not yet measured.** The numbers below are the *expected ordering and target ranges* from the design plan, not reported metrics. Replace with measured values from the H100 training notebook before publishing the report. They are marked clearly so no reader mistakes them for verified results.

| Model (projected) | Field F1 (flat) | Line-item F1 | Notes |
|---|---|---|---|
| Regex / heuristic floor | lowest | n/a | zero-training, fully interpretable floor |
| `bert-base + bbox` | low–mid | low | isolates value of layout-pretraining |
| **LiLT-roberta-en** | high | mid | MIT, commercial; ≈ LayoutLMv3 on flat fields |
| **LayoutLMv3-base** | high | mid | accuracy benchmark (internal) |
| LayoutLMv3-large | highest (flat) | mid | upper bound on flat fields |
| **Donut** | mid | **highest** | best on nested line items; no token boxes |

**Expected ordering:** `regex ≪ bert+bbox < LiLT ≈ LayoutLMv3-base ≤ LayoutLMv3-large` for flat fields; **`Donut > token-cls`** for nested line items.

### 5.4 Decision thresholds (agent gates)

`Q_MIN=0.45`, `OCR_MIN=0.70`, `FIELD_CONF_MIN=0.80`, `AUTO_MIN=0.85`, `MAX_OCR_ATTEMPTS=2`, `ε = max(0.01, 0.005 × total)`. `overall_confidence = min(doc_type_conf, ocr_mean_conf, min(required-field confidences))` — a conservative bottleneck so one shaky required field blocks auto-approval.

---

## 6. Ethical Considerations

- **Financial-decision risk.** Extracted values feed accounting aggregates. A silently wrong `total` can corrupt revenue/tax summaries. The system mitigates this with **arithmetic reconciliation** (`sum(lines)+tax == total ± ε`, `qty×unit_price == amount ± ε`) and routes any failure to a human; the LayoutLMv3/LiLT path is **grounded** — every prediction maps to an OCR token with a bbox, so reviewers can verify provenance.
- **Human-in-the-loop is mandatory for flagged docs.** Auto-approval requires `reconciles AND no missing_required AND overall_confidence ≥ AUTO_MIN`; everything else is reviewed. The model is not authorized to post unreviewed flagged values.
- **License compliance.** Using LayoutLMv3 commercially would violate CC-BY-NC-SA-4.0; the card and code restrict it to internal/benchmark use and route production to MIT-licensed LiLT/Donut.
- **Privacy.** Documents contain PII/financial data. Offline-first operation keeps data on-prem; the optional cloud LLM fallback is feature-flagged and disabled by default — no document leaves the host unless explicitly enabled.
- **Bias / coverage.** Training data skews to English receipts (SROIE/CORD) and a small invoice set; performance on other locales, currencies, and layouts is unverified.

---

## 7. Caveats & Limitations

- **OCR dependency (token-cls path).** LayoutLMv3/LiLT cannot extract what OCR misses. Noisy/rotated/low-DPI scans degrade results. Mitigation: native-PDF text first → PaddleOCR (primary) → docTR (fallback), deskew/denoise/upscale on retry, OCR confidence gate (`OCR_MIN=0.70`) routes bad scans to review after `MAX_OCR_ATTEMPTS=2`.
- **Small training data.** SROIE~600, CORD~800, FUNSD~150 → overfitting and high-variance metrics (especially FUNSD); report mean±std across seeds.
- **Donut hallucination.** Donut can **invent** field values at low confidence and provides **no token boxes** (hard to build a highlight-on-source review UI). Use it only for line-items / OCR-hostile docs; cross-check its totals against the validation reconciler; prefer grounded LayoutLMv3/LiLT for review-facing fields.
- **CPU latency.** LayoutLMv3 is acceptable on CPU at low volume (~0.8–1.5 s p95). **Do not run Donut on CPU** (4–10 s/page).
- **Bbox normalization (0–1000).** A silent killer if mis-applied; verify HF-shipped boxes are already normalized before re-normalizing.
- **transformers version drift.** 5.x dropped deprecated args (`tokenizer=`→`processing_class=`); pin `4.51.*` and record the version in `/health`.

---

## 8. How to Load / Serve

### 8.1 Load (token-classification path)

```python
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification

processor = LayoutLMv3Processor.from_pretrained(
    "microsoft/layoutlmv3-base", apply_ocr=False)        # pass your own words+boxes (0–1000)
model = LayoutLMv3ForTokenClassification.from_pretrained(
    "<hub-repo>", revision="<pinned-sha>")               # resolves @prod alias → sha at boot

# Commercial-safe swap (MIT) — same pipeline:
# from transformers import AutoModelForTokenClassification
# model = AutoModelForTokenClassification.from_pretrained("SCUT-DLVCLab/lilt-roberta-en-base", ...)
```

### 8.2 Load (Donut path)

```python
from transformers import DonutProcessor, VisionEncoderDecoderModel
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
# Donut runs NO OCR — feed the page image directly.
```

### 8.3 Serve

- **FastAPI** endpoints: `/health`, `/extract`, `/classify`, `/batch`, `/batch/{job_id}`, `/review-queue`, `/metrics`. `/extract` returns normalized JSON with per-field confidence + pixel bboxes (top-left origin) + `needs_review`; `model_version` is echoed on every response.
- **Gradio** highlight demo: draws bboxes (green if conf ≥ 0.8 else orange), auto-launches on **port 7860** for HF Spaces.
- **Docker / HF Space:** `python:3.11-slim` + `tesseract-ocr` + `poppler-utils`, `EXPOSE 7860`, `uvicorn app:app --host 0.0.0.0 --port 7860`. `/health` returns `503 MODEL_LOADING` until weights are resident.
- **Latency targets:** end-to-end `/extract` on a 1-page image **~250–500 ms p95 (GPU)** / **~0.8–1.5 s p95 (CPU)**.

---

## 9. Positioning

`invoice_ai` strictly dominates the reference `ruizguille/invoice-processing` (GPT-4o-vision only; no OCR, no arithmetic reconciliation, no confidence, no human-in-the-loop; online-mandatory; first-page-only; float money). Here the entire GPT-4o-vision approach is demoted to **one optional, feature-flagged fallback tool**, while the local **LayoutLMv3/LiLT + arithmetic reconciliation + Decimal money + multi-page + offline** path is the default. A hallucinated total that the reference would silently add to revenue is caught by the reconciler and routed to a human.

---

*Card maintained alongside the model weights; update §5.3 with measured metrics from the H100 training notebook before report publication.*
