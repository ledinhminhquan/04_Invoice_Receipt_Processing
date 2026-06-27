# Data Description Document

**Project #4 — Invoice & Receipt Processing System** · Package `invoice_ai`
**Author:** Le Dinh Minh Quan (23127460) · **Assignment §4 — Data Description**

This document inventories every dataset that feeds the offline-first Key-Information-Extraction (KIE)
pipeline, with verified Hugging Face ids, licenses, sizes, languages, the preprocessing contract, split
justification, and known limitations/biases. All ids below were live-verified on the HF Hub on
**2026-06-26**. No large data is committed to the repository; download scripts pull on demand.

---

## 1. Dataset Sources & Licenses

The system trains and benchmarks two model families — **LayoutLMv3 / LiLT** (token classification over
OCR words+boxes) and **Donut** (OCR-free image→JSON) — so datasets are grouped by which family they
feed. All ids were confirmed live via the HF dataset-viewer (configs, splits, row counts, schema).

| Dataset (HF id) | Source / Task | Format | Rows (tr/val/te) | Ready for | License |
|---|---|---|---|---|---|
| **`mp-02/sroie`** | SROIE / ICDAR-2019 receipt KIE | image + words + bbox + ner (`S-`, 5-class) | 626 / – / 347 | **LayoutLMv3** | unknown (card) ⚠️ ICDAR research data |
| **`naver-clova-ix/cord-v2`** | CORD receipt line-items | image + `ground_truth` JSON | 800 / 100 / 100 | **Donut** | **cc-by-4.0** ✅ |
| **`nielsr/funsd-layoutlmv3`** | FUNSD form KIE | image + tokens + bbox + ner (BIO, 7-class) | 149 / – / 50 | **LayoutLMv3** | unknown ⚠️ ICDAR research data |
| **`katanaml-org/invoices-donut-data-v1`** | Invoice JSON (clean) | image + gt JSON | 425 / 50 / 26 | **Donut** | **MIT** ✅ |

**Supporting / scaling sets (optional):**

| Dataset (HF id) | Note | License |
|---|---|---|
| `darentang/sroie` | BIO 9-class SROIE, but `image_path` is a **string path, not `Image`** (images not bundled). Use only if supplying images yourself. | unknown |
| `nielsr/funsd` | FUNSD with `words` column + un-normalized v1/v2 boxes (for code expecting `words`). | unknown |
| `mychen76/invoices-and-receipts_ocr_v1` | Largest mixed set (2000/70/125), image + `parsed_data` JSON; best for scaling Donut on H100. | not declared ⚠️ |

**Permissive-license fallbacks (verified to exist):**

| Dataset (HF id) | Note | License |
|---|---|---|
| `arvindrajan92/sroie_document_understanding` | SROIE enriched with line-item / line-description labels. | **MIT** ✅ |
| `jsdnrs/ICDAR2019-SROIE` | Parquet image+text, newest mirror. | **cc-by-4.0** ✅ |
| `rth/sroie-2019-v2` | De-duplicated from official RRC source. | parquet, image+text |

### License flags (production-critical)

- **LayoutLMv3 model is NON-COMMERCIAL (CC-BY-NC-SA-4.0)** — flagged here because it constrains the
  whole accuracy-primary path. The *datasets* that feed it (`mp-02/sroie`, `nielsr/funsd-layoutlmv3`)
  are **ICDAR research-challenge data**: SROIE/FUNSD base data are non-commercial-leaning and several
  mirrors declare **no license**. Treat them as **internal / benchmark only**.
- **Commercially-clean alternatives exist and are verified:** `naver-clova-ix/cord-v2` (**cc-by-4.0**),
  `katanaml-org/invoices-donut-data-v1` (**MIT**), `arvindrajan92/sroie_document_understanding`
  (**MIT**), `jsdnrs/ICDAR2019-SROIE` (**cc-by-4.0**). For any commercial deployment, ship these plus a
  commercial-safe model (LiLT-MIT or Donut-MIT). **Clear SROIE/FUNSD with legal before commercial use.**

---

## 2. Sizes & Language

- **Volume is small by design** — SROIE ≈ 973 documents (626+347), CORD = 1000 (800/100/100),
  FUNSD = 199 (149/50), katanaml invoices = 501 (425/50/26). This is typical for document-KIE
  benchmarks and drives the anti-overfitting recipe (early stopping, low LR, multi-seed mean±std).
- **Language: predominantly English.** SROIE receipts are English (Southeast-Asian retail receipts);
  FUNSD forms are English scanned documents; CORD receipts are English-keyed; the katanaml invoices are
  English. There is no in-corpus multilingual coverage — multilingual support is a *model* concern
  (swap to `nielsr/lilt-xlm-roberta-base`), not a data one.
- **Domain mix:** SROIE and CORD are **receipts** (dense, tall, POS-style); FUNSD is **generic forms**
  (included to demonstrate form KIE, *not* the invoice/receipt target); katanaml is **invoices**.

---

## 3. Preprocessing (the canonical internal schema)

Every loader maps its raw rows into one canonical internal schema so the model code is dataset-agnostic:

- **LayoutLMv3 / LiLT path:** `image` + `words`/`tokens` + `bboxes` (0–1000 normalized) + `labels`.
- **Donut path:** `image` + `ground_truth` JSON (target = `json.loads(ground_truth)["gt_parse"]`).

### 3.1 Image
- All images converted to **RGB** (`PIL.Image.convert("RGB")`); EXIF auto-orient; optional deskew.

### 3.2 OCR words + boxes
- Token classification consumes **precomputed** OCR words + boxes via
  `LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)`.
- HF SROIE/FUNSD-v3/CORD ship words+boxes; raw images go through PaddleOCR (primary) → Tesseract /
  native-PDF text layer for born-digital documents.

### 3.3 Bbox normalize to 0–1000 (the #1 silent bug)
- LayoutLMv3 **requires** boxes normalized to a 0–1000 integer grid:
  `x_norm = int(1000 * x_pixel / image_width)` (same for y/height). Getting this wrong is the single
  largest cause of silent accuracy loss. **Verify before re-normalizing:** HF CORD and FUNSD-v3 boxes
  ship **already normalized**; `mp-02/sroie` / raw images must be normalized by the loader. Assert
  `len(words) == len(boxes) == len(labels)`.

### 3.4 Label schemes — `S-` (SROIE) vs BIO (FUNSD)
- **`mp-02/sroie`:** single-token **`S-` tags** (NOT BIO): `S-COMPANY, S-DATE, S-ADDRESS, S-TOTAL, O`
  (5 classes). Either train on `S-` directly or map to canonical BIO `O, B/I-{COMPANY,DATE,ADDRESS,TOTAL}`
  — be consistent in `id2label`/`label2id`.
- **`nielsr/funsd-layoutlmv3`:** **BIO**, 7 classes: `O, B/I-{HEADER,QUESTION,ANSWER}`.
- **`cord-v2`:** target is the `gt_parse` JSON tree (`menu` line items `nm/num/cnt/price/itemsubtotal`,
  `sub_total`, `total`), not tags; per-word `quad` boxes also allow deriving a token-cls version.

### 3.5 Label alignment → −100
- When `word_labels` is passed, the processor sets **continuation subwords + special tokens to `-100`**
  automatically (only the FIRST subword keeps the label; `-100` is ignored by `CrossEntropyLoss`).
  If tokenizing manually, replicate via `word_ids` (`None`→−100, first subword→label, continuation→−100).

---

## 4. Split Justification

| Dataset | Train | Val | Test | Justification |
|---|---|---|---|---|
| `mp-02/sroie` | 626 | — (carve) | 347 | Ships only train/test (**no val**). **Carve a 10% deterministic val set from train, `seed=42`** so the test split stays untouched for the final benchmark. |
| `naver-clova-ix/cord-v2` | 800 | 100 | 100 | All three official splits present — used as-is (standard receipt-line-item benchmark). |
| `nielsr/funsd-layoutlmv3` | 149 | — | 50 | Tiny official train/test only. Use as-is for the form-KIE demonstration; high variance at n=149 → report **mean±std over multiple seeds**. |
| `katanaml-org/invoices-donut-data-v1` | 425 | 50 | 26 | Official 3-way split used as-is for the commercially-clean invoice Donut comparison. |

The `seed=42` carve is reproducible and identical across runs, keeping the held-out test counts (347 /
100 / 50) stable for the baseline comparison axis (regex ≪ bert+bbox < LiLT ≈ LayoutLMv3 ≤ v3-large;
Donut > token-cls for nested line items).

---

## 5. Limitations & Biases

| # | Limitation / Bias | Impact & Mitigation |
|---|---|---|
| 1 | **Small datasets** (SROIE≈600, CORD≈800, FUNSD≈150 train) | High overfitting risk → early stopping on eval F1, low LR + warmup + cosine, weight decay + label smoothing, freeze early layers (FUNSD), start from in-domain checkpoints, multi-seed mean±std. |
| 2 | **Receipt-domain skew** | SROIE/CORD are POS receipts; FUNSD is generic forms; only katanaml is true invoices. Model may underperform on unseen invoice layouts → add `katanaml-org/invoices-donut-data-v1` and (optionally) `mychen76/...` for invoice coverage. |
| 3 | **Currency / locale bias** | Corpora are English with limited currency variety; ambiguous dd/mm vs mm/dd and non-USD/GBP currencies are under-represented → validation flags ambiguous dates; ISO-4217 detection + `Decimal` money normalization at inference. |
| 4 | **OCR noise** | Scanned receipts carry blur/skew/low-DPI artifacts; boxes inherit OCR error → multi-engine retry (PaddleOCR→Tesseract/native-PDF), OCR confidence gate `OCR_MIN=0.70`, route bad scans to human review. |
| 5 | **Label-scheme heterogeneity** | `S-` (SROIE) vs BIO (FUNSD) vs JSON (CORD) → unified `id2label` per dataset; consistency asserted in loaders. |
| 6 | **License non-commercial leaning** | SROIE/FUNSD ICDAR research data + LayoutLMv3 (CC-BY-NC-SA-4.0) → internal/benchmark only; commercial path uses cc-by-4.0/MIT datasets + LiLT/Donut. |

---

## 6. Per-Dataset Schema Tables

**`mp-02/sroie`** (4 columns)

| Column | Type | Notes |
|---|---|---|
| `image` | Image | Real embedded image bytes (RGB). |
| `words` | List[str] | OCR token strings. |
| `bboxes` | List[List[int64]] | Per-token boxes (normalize to 0–1000). |
| `ner_tags` | ClassLabel seq | 5-class `S-COMPANY, S-DATE, S-ADDRESS, S-TOTAL, O`. |

**`naver-clova-ix/cord-v2`** (2 columns)

| Column | Type | Notes |
|---|---|---|
| `image` | Image | Embedded receipt image. |
| `ground_truth` | JSON string | `gt_parse` → `menu` (`nm,num,cnt,price,itemsubtotal`), `sub_total` (`subtotal_price,discount_price,tax_price`), `total` (`total_price,creditcardprice,menuqty_cnt`); plus `meta`, `valid_line` (per-word `quad` + `category`), `roi`. |

**`nielsr/funsd-layoutlmv3`** (5 columns)

| Column | Type | Notes |
|---|---|---|
| `id` | str | Document id. |
| `tokens` | List[str] | v3 column convention. |
| `bboxes` | List[List[int]] | Pre-normalized for v3. |
| `ner_tags` | ClassLabel seq | BIO 7-class `O, B/I-{HEADER,QUESTION,ANSWER}`. |
| `image` | Image | Embedded. |

**`katanaml-org/invoices-donut-data-v1`** (2 columns)

| Column | Type | Notes |
|---|---|---|
| `image` | Image | Invoice image. |
| `ground_truth` | JSON string | Same Donut image→JSON format as CORD. |

---

## 7. Download-Script Note (no large data committed)

```
scripts/  ──►  data/   (git-ignored)
   download_sroie.py   → mp-02/sroie
   download_cord.py    → naver-clova-ix/cord-v2
   download_funsd.py   → nielsr/funsd-layoutlmv3
   download_invoices.py→ katanaml-org/invoices-donut-data-v1
```

- Scripts live in `scripts/` and write **only** into `data/` (git-ignored). `.gitignore` excludes
  `data/`, `artifacts/`, `models/*.bin|*.safetensors`, and downloaded parquet. **No large data committed.**
- Pin dataset **revision SHAs**; scripts **fail loudly** if the dataset-viewer 404s.
- **Nonexistent ids deliberately avoided** (verified): `jinhybr/SROIE...` (does not exist),
  `jordyvl/funsd` (404 from the Hub), `rvl_cdip` / `aharley/rvl_cdip` / `chainyo/rvl-cdip`
  (16-class document *classification* only — **no KIE field labels**, wrong task). **Do not use these.**
