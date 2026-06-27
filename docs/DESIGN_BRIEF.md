# Invoice & Receipt Processing System — Authoritative Design Brief

> **Project #4** — A production, offline-first Document-AI Key-Information-Extraction (KIE) system.
> Deliverables: a runnable Python repo (package **`invoice_ai`**), an **H100 Colab training notebook**, and full docs.
> Synthesized from four verified research briefs (Datasets, Models, Reference/Agentic, Deploy/Training). All Hugging Face ids below were live-verified on **2026-06-26**; every unverified item is flagged with a verified fallback.

---

## 0. System Overview

### 0.1 What this is

A **Document-AI Key-Information-Extraction (KIE)** system that turns invoice/receipt **images and PDFs** into validated, normalized **structured JSON** with per-field **confidence** and a **`needs_review`** flag. It is designed to run **fully offline** (no paid API), with an **optional** cloud LLM-vision "brain" as a fallback for hard cases.

The agent is a **deterministic state machine** (the default, fully-local path) augmented by an **optional LLM-vision tool** that is feature-flagged and isolated. With the network unplugged the system still runs end-to-end — uncertain documents go to human review instead of cloud escalation. This is what lets the system **strictly dominate** the reference implementation (`ruizguille/invoice-processing`): the reference's entire GPT-4o-vision approach is available here as one optional tool, while our baseline runs without it.

### 0.2 Pipeline (canonical flow)

```
ingest (image / PDF)
   → classify doc type            (invoice / receipt / other + scan-quality gate)
   → OCR (words + boxes)          (native PDF text layer | Tesseract | PaddleOCR; per-token conf)
   → layout extractor             (LayoutLMv3 token classification → header fields)
   → line-item extractor          (table-structure → rows: desc / qty / unit_price / amount)
   → validate                     (totals reconcile, per-line math, date valid, currency, required-present)
   → normalize                    (ISO-8601 dates, Decimal money, ISO-4217 currency)
   → structured JSON              (+ per-field confidence + bboxes + needs_review + review_reasons)
```

The **agent = deterministic state machine + optional LLM-vision brain**. Three explicit decision points govern routing (doc-type/quality → OCR retry; validation → human review; final confidence → auto-approve vs review). See Section 3.

### 0.3 Why it exists (positioning vs. the reference)

The reference (`ruizguille/invoice-processing`, MIT) is a batch **PDF → Excel** pipeline driven **entirely by GPT-4o vision** with JSON mode + Pydantic type coercion. It is **not** an agent: no OCR, no arithmetic reconciliation, no confidence, no human-in-the-loop, **first page only**, rasterizes even digital PDFs, and **mandatorily online** (no `OPENAI_API_KEY` ⇒ nothing runs). A hallucinated total flows silently into its financial aggregates.

| Dimension | Reference (ruizguille) | **`invoice_ai` (this system)** |
|---|---|---|
| Primary extractor | GPT-4o vision (cloud) | **LayoutLMv3 / Donut, fine-tuned, local** |
| Works offline / on-prem | No | **Yes (default path)** |
| OCR | None (rasterize + LLM) | **Explicit OCR tool** w/ confidence + boxes |
| Cloud LLM | Mandatory | **Optional fallback brain** (feature-flagged) |
| Validation | Type coercion only | **Arithmetic reconciliation + date/number/currency/format rules** |
| Confidence + routing | None | **Per-field confidence, doc-type routing, quality gate** |
| Human-in-the-loop | None | **Auto-approve vs. flag-for-review** |
| Doc types | Invoice only | **Invoice / receipt / other classifier** |
| Money representation | float (lossy) | **Decimal + detected ISO-4217 currency** |
| Multi-page | First page only | **All pages, merged** |
| Control flow | Linear script | **Stateful agent with 3 decision points** |

---

## 1. DATA Stack

All dataset ids below were confirmed live via the HF Hub dataset-viewer (configs, splits, row counts, schema). **No large data is committed** — download scripts pull on demand into `data/` (git-ignored).

> **Field-type legend:** `Image` = embedded image bytes; `words`/`tokens` = OCR token strings; `bboxes` = per-token boxes; `ner_tags` = `ClassLabel` sequence; `ground_truth`/`parsed_data` = JSON string.

### 1.1 Selection matrix (VERIFIED ids)

| Use case | Dataset id | Format | Rows (tr/val/te) | Ready for | License |
|---|---|---|---|---|---|
| Receipt KIE, has images | **`mp-02/sroie`** | image+words+bbox+ner (5-class `S-`) | 626 / – / 347 | **LayoutLMv3** | unknown (card) |
| Receipt KIE, BIO 9-class | `darentang/sroie` | words+bbox+ner, **`image_path` only** | 626 / – / 347 | LayoutLM* (bring images) | unknown |
| Receipt line-items JSON | **`naver-clova-ix/cord-v2`** | image + `gt_parse` JSON | 800 / 100 / 100 | **Donut** | **cc-by-4.0** |
| Form KIE | **`nielsr/funsd-layoutlmv3`** | image+tokens+bbox+ner (7-class BIO) | 149 / – / 50 | **LayoutLMv3** | unknown |
| Form KIE (v1/v2 boxes) | `nielsr/funsd` | image+words+bbox+ner (7-class BIO) | 149 / – / 50 | LayoutLMv2 | unknown |
| Invoice JSON, MIT | **`katanaml-org/invoices-donut-data-v1`** | image + gt JSON | 425 / 50 / 26 | **Donut** | **MIT** |
| Largest invoice+receipt | `mychen76/invoices-and-receipts_ocr_v1` | image + `parsed_data` JSON | 2000 / 70 / 125 | Donut | unknown |

### 1.2 Primary datasets (decisive picks)

**SROIE → `mp-02/sroie` (LayoutLMv3-ready, embedded images).**
- Config `default`: train **626**, test **347** (no val split — carve a val set deterministically, e.g. 10% of train, `seed=42`).
- Schema (4 cols): `image` (Image) · `words` (List[str]) · `bboxes` (List[List[int64]]) · `ner_tags` (ClassLabel seq).
- **Tag scheme:** single-token `S-` tags (NOT BIO): `S-COMPANY, S-DATE, S-ADDRESS, S-TOTAL, O` (5 classes). Has real embedded images → directly usable by `LayoutLMv3` `AutoProcessor`.
- **Alternative for canonical BIO:** `darentang/sroie` (9-class BIO `O, B/I-{COMPANY,DATE,ADDRESS,TOTAL}`) — the HF LayoutLMv2/v3 tutorial set, BUT its `image_path` is a **string path, not an `Image`** (images not bundled → broken for direct `load_dataset` image use on Colab). Use only if you supply images yourself.

**CORD → `naver-clova-ix/cord-v2` (Donut-ready, line items as JSON).**
- Config `default`: train **800**, val **100**, test **100** (all 3 splits). License **cc-by-4.0** (explicit, commercial-friendly). The standard receipt-line-item benchmark.
- Schema (2 cols): `image` (Image) · `ground_truth` (JSON string).
- `ground_truth` structure (verified from a real row): `gt_parse` → clean target JSON with `menu` (line items: `nm, num, cnt, price, itemsubtotal`), `sub_total` (`subtotal_price, discount_price, tax_price`), `total` (`total_price, creditcardprice, menuqty_cnt`); plus `meta`, `valid_line` (per-word `quad` boxes + `category` like `menu.nm`, `total.total_price`), `roi`, `repeating_symbol`, `dontcare`.
- **For Donut:** target = `json.loads(ground_truth)["gt_parse"]`. The per-word `quad` boxes also let you derive a token-classification version if needed.

**FUNSD → `nielsr/funsd-layoutlmv3` (LayoutLMv3-ready, embedded images).**
- Config `funsd`: train **149**, test **50**. Column named **`tokens`** (v3 convention); boxes pre-normalized for v3.
- Schema (5 cols): `id` · `tokens` · `bboxes` · `ner_tags` (BIO, 7 classes: `O, B/I-{HEADER,QUESTION,ANSWER}`) · `image` (Image, embedded).
- **Use `nielsr/funsd`** (column `words`, un-normalized v1/v2-style boxes) only if your code expects the `words` column. FUNSD is included to demonstrate generic form KIE; it is not the invoice/receipt target.

### 1.3 Optional / scaling datasets

- **`katanaml-org/invoices-donut-data-v1`** — Donut-ready **invoices** (not receipts), train 425 / val 50 / test 26, schema `image` + `ground_truth` JSON (same Donut format as CORD). **License MIT** → best small clean *invoice* set for commercially-friendly Donut fine-tuning.
- **`mychen76/invoices-and-receipts_ocr_v1`** — largest mixed set: train 2000 / valid 70 / test 125; schema `image` + `id` + `parsed_data` (JSON) + `raw_data` (raw OCR). **License not declared** → verify before commercial use. Best for scaling Donut/Pix2Struct on H100; not pre-tokenized for LayoutLMv3.

### 1.4 Permissive-license fallbacks (verified to exist)

- `arvindrajan92/sroie_document_understanding` — **MIT**, enriched with line-item / line-description labels (good if you want SROIE line items commercially).
- `jsdnrs/ICDAR2019-SROIE` — **cc-by-4.0**, parquet image+text (newest, Feb 2026).
- `rth/sroie-2019-v2` — parquet, image+text, de-duplicated from official RRC source.

### 1.5 DO NOT USE (verified nonexistent / wrong-task)

- **`jinhybr/SROIE...`** — does not exist (search returned nothing). Drop it.
- **`jordyvl/funsd`** — does not exist (404 from the Hub). Drop it.
- **`rvl_cdip` / `aharley/rvl_cdip` / `chainyo/rvl-cdip`** — 16-class document *classification* only, **no KIE field labels**. Not for KIE.

### 1.6 Schema, licensing, and the no-commit rule

- **Canonical internal schema** (what every loader maps into): `image` + `words`/`tokens` + `bboxes` (0–1000 normalized for LayoutLMv3) + `labels`. For Donut sets, `image` + `ground_truth` JSON.
- **LayoutLMv3-ready vs Donut-ready:** LayoutLMv3 needs token-classification data (`mp-02/sroie`, `nielsr/funsd-layoutlmv3`); Donut needs image→JSON data (`cord-v2`, `katanaml-org/invoices-donut-data-v1`, `mychen76/...`).
- **Licensing caveat (production):** only `cord-v2` (cc-by-4.0), `katanaml-org/invoices-donut-data-v1` (MIT), `arvindrajan92/sroie_document_understanding` (MIT), and `jsdnrs/ICDAR2019-SROIE` (cc-by-4.0) carry explicit permissive licenses. SROIE/FUNSD base data are ICDAR research-challenge data (non-commercial-leaning) and several mirrors declare no license — **clear with legal before commercial deployment**.
- **Download scripts** live in `scripts/` and write only into `data/` (git-ignored). `.gitignore` must exclude `data/`, `artifacts/`, `models/*.bin|*.safetensors`, and any downloaded parquet. **No large data committed.**

---

## 2. MODEL Stack

> **All ids verified on the Hub (params / license / task as of 2026-06-26).** One correction: **`microsoft/lilt-roberta-en-base` does not exist** — the real id is **`SCUT-DLVCLab/lilt-roberta-en-base`**.

### 2.1 ID verification table

| Component | Verified HF id | Params | License | Task / Arch |
|---|---|---|---|---|
| Layout extractor (**primary, accuracy**) | `microsoft/layoutlmv3-base` | 125.3M | **cc-by-nc-sa-4.0** ⚠️ NC | `layoutlmv3` token-cls |
| Layout extractor (large) | `microsoft/layoutlmv3-large` | ~368M | **cc-by-nc-sa-4.0** ⚠️ NC | `layoutlmv3` |
| Layout (v2 — avoid) | `microsoft/layoutlmv2-base-uncased` | ~200M | **cc-by-nc-sa-4.0** ⚠️ NC | `layoutlmv2` (needs `detectron2`) |
| Layout (**commercial-safe**) | `SCUT-DLVCLab/lilt-roberta-en-base` | 130.8M | **MIT** ✅ | `lilt` |
| Layout (multilingual) | `nielsr/lilt-xlm-roberta-base` | 284.2M | **MIT** ✅ | `lilt`, 90+ langs |
| Donut (base, fine-tune target) | `naver-clova-ix/donut-base` | ~200M | **MIT** ✅ | vision-enc-dec, image→text |
| Donut (CORD receipts, ready) | `naver-clova-ix/donut-base-finetuned-cord-v2` | ~200M | **MIT** ✅ | image→JSON |
| Baseline encoder | `google-bert/bert-base-uncased` | 110.1M | **apache-2.0** ✅ | bert + bbox baseline |

> ⚠️ **License is the single most important production driver.** Every LayoutLM (v2/v3) checkpoint is **CC-BY-NC-SA-4.0 (non-commercial)**. For commercial deployment, ship **LiLT (MIT)** or **Donut (MIT)**, not LayoutLMv3. LayoutLMv3 remains the **accuracy benchmark / internal** model.

### 2.2 Decisive recommendations

- **Primary (accuracy / research / internal):** fine-tune **`microsoft/layoutlmv3-base`** with your own OCR and `apply_ocr=False`.
- **Primary (commercial / multilingual):** fine-tune **`SCUT-DLVCLab/lilt-roberta-en-base`** (or `nielsr/lilt-xlm-roberta-base` for non-English invoices — swap the text stream, no layout retraining).
- **OCR-free / heavy line-items:** **`naver-clova-ix/donut-base-finetuned-cord-v2`** (run today, zero training), fine-tune from `naver-clova-ix/donut-base`.
- **Baseline floor:** regex/heuristics + `google-bert/bert-base-uncased` + bbox token-classification.
- **Do NOT use LayoutLMv2** — requires `detectron2` (notorious Colab install pain), NC license, strictly superseded by v3.

**Train both LayoutLMv3-base and LiLT** — they share the same OCR-token data pipeline (token-classification heads), so pick on your eval set.

### 2.3 LayoutLMv3 vs Donut — when to prefer each

| Prefer **Donut** when… | Prefer **LayoutLMv3 / LiLT** when… |
|---|---|
| No OCR dependency (end-to-end image→JSON) | You already trust an OCR pipeline |
| Output is **nested JSON** (line-items, tables) | Output is **flat field tagging** |
| OCR fails (low-res, stylized fonts, dense receipts) | You need **per-word boxes / provenance** for review |
| You can afford autoregressive decode latency | You need lowest latency per page |

**Key practical contrast:** LayoutLMv3/LiLT **cannot invent text** — every prediction is grounded to an OCR token with a bbox (auditors and human-in-the-loop reviewers love this). Donut **hallucinates** fields at low confidence and gives **no token boxes** (hard to build a highlight-on-source review UI), but **shines on line-item extraction** where token-classification struggles to group rows.

### 2.4 OCR engines (words + boxes)

**Recommended primary: PaddleOCR** (Apache-2.0) — best accuracy/robustness on invoices/receipts (rotation, dense layouts, multilingual incl. Vietnamese), strong box geometry, GPU-accelerated on H100. Trade-off: heavier install, occasional Colab version churn.

| Engine | License | Doc accuracy | Colab ease | GPU | Role |
|---|---|---|---|---|---|
| **PaddleOCR** | Apache-2.0 | Best (rotation, dense, multilingual) | Medium | Yes | **primary** |
| **docTR** | Apache-2.0 | Very good, clean API | Easy | Yes | best **fallback** |
| EasyOCR | Apache-2.0 | Good, 80+ langs | Easy | Yes | simple alt |
| Tesseract (pytesseract) | Apache-2.0 | Baseline (weak on noisy scans) | Easiest | No (CPU) | LayoutLMv3 internal default; OK for born-digital |

**`apply_ocr` note (confirmed in Transformers docs):** `LayoutLMv3Processor`/`LayoutLMv2Processor` default to `apply_ocr=True` and run **Tesseract** internally. **For production, set `apply_ocr=False`**, install PaddleOCR, and pass `words=[...]` + `boxes=[...]` (normalized to 0–1000) into the processor. This decouples OCR quality from the model and is required to beat the Tesseract default. (The DocumentQA pipeline also auto-runs Tesseract for LayoutLM-like models when `word_boxes` aren't supplied; **Donut runs no OCR**.)

### 2.5 PDF handling (router)

- **Born-digital PDFs → `pdfplumber`** (or `PyMuPDF`/fitz): extracts text **with word-level bounding boxes directly** — no OCR, near-perfect, fast. Feed boxes straight into LayoutLMv3/LiLT (normalize to 0–1000).
- **Scanned PDFs → `pdf2image`** (`convert_from_path`, Poppler) to rasterize at **200–300 DPI**, then run OCR (PaddleOCR) → words+boxes, or feed the image to Donut.
- **Router logic:** try `pdfplumber` text-extraction first; if a page yields ~no extractable text → treat as scanned → `pdf2image` + OCR. This born-digital-vs-scanned branch is a meaningful accuracy + latency win and fixes the reference's "rasterize everything" waste.

### 2.6 Preprocessing notes (the #1 silent-bug zone)

- **Bbox normalization to 0–1000** (LayoutLMv3 requirement): `x_norm = int(1000 * x_pixel / image_width)` (same for y/height). Getting this wrong is the **#1 cause of silent accuracy loss** with v3. HF CORD/FUNSD-v3 boxes ship already normalized; SROIE/raw images must be normalized by you.
- **Processor:** `LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)` → pass `images, words, boxes, word_labels`.
- **Label alignment:** when `word_labels` is supplied, the processor sets **continuation subwords + special tokens to `-100`** automatically (only the FIRST subword keeps the label; `-100` is ignored by CrossEntropyLoss). If you tokenize yourself, replicate this with `word_ids`-based alignment.

### 2.7 Recommended production stack (diagram)

```
PDF in ─┬─ born-digital ─→ pdfplumber (text + boxes)            ┐
        └─ scanned ──────→ pdf2image → PaddleOCR (words+boxes)  ┘
                                   │ normalize boxes 0–1000
                                   ▼
   PRIMARY:    LayoutLMv3-base (apply_ocr=False)   [internal/accuracy]
   COMMERCIAL: LiLT-roberta-en (or lilt-xlm-roberta for multilingual)
                                   │
   PARALLEL (line-items / OCR-hostile docs):
        Donut-base-finetuned-cord-v2  →  image→JSON  (no OCR)
                                   │
   BASELINE FLOOR: regex/heuristics + bert+bbox
```

---

## 3. PIPELINE + AGENT Architecture

A stateful agent that classifies, OCRs, extracts, **validates**, normalizes, and **decides** between auto-approve and human review. Runs fully on **LayoutLMv3 + rules** with **no paid API**; the LLM brain is an optional, swappable escalation tool.

### 3.1 Architecture overview

```
                          ┌──────────────────────────────────────────────┐
                          │                AGENT (planner)                │
                          │   policy: rules engine  |  optional LLM brain │
                          └──────────────────────────────────────────────┘
   input doc                         │ reads/writes
 (PDF/PNG/JPG) ─────►  ┌─────────────▼─────────────┐
                       │        AgentState         │  (single shared blackboard)
                       └─────────────┬─────────────┘
        ┌───────────────┬────────────┼─────────────┬───────────────┬───────────────┐
        ▼               ▼            ▼              ▼               ▼               ▼
  classify_document   run_ocr   extract_layout  extract_line_   validate      normalize
  (invoice/receipt/  (text +    (LayoutLMv3 →   items          (reconcile,   (ISO dates,
   other + quality)  conf/box)   header fields) (table → rows)  date/num/cur) Decimal, cur)
                                                          │
                                                          ▼
                                            [optional] llm_vision_fallback
                                                          │
                              ┌───────────────────────────┴──────────────────────┐
                              ▼                                                    ▼
                       AUTO-APPROVE  (high conf + reconciled)            HUMAN REVIEW (flagged)
```

### 3.2 `AgentState` (the blackboard)

Every tool reads/writes one shared object; the trace is reproducible.

```python
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any

class DocType(str, Enum):
    INVOICE="invoice"; RECEIPT="receipt"; OTHER="other"; UNKNOWN="unknown"

class Status(str, Enum):
    NEW="new"; ROUTED="routed"; OCR_DONE="ocr_done"; EXTRACTED="extracted"
    VALIDATED="validated"; NORMALIZED="normalized"
    AUTO_APPROVED="auto_approved"; NEEDS_REVIEW="needs_review"; FAILED="failed"

@dataclass
class FieldValue:
    value: Any
    confidence: float           # 0..1 from model softmax / OCR / rule
    source: str                 # "layoutlmv3"|"donut"|"ocr"|"llm"|"rule"
    bbox: tuple | None = None   # provenance for the review UI

@dataclass
class LineItem:
    description: FieldValue
    quantity:   FieldValue | None
    unit_price: FieldValue | None
    amount:     FieldValue                 # line total

@dataclass
class ValidationReport:
    reconciles: bool                       # sum(lines)+tax ≈ total
    reconcile_delta: Decimal               # total - (subtotal + tax)
    checks: dict[str, bool]                # date_valid, invoice_no_plausible, currency_detected, ...
    missing_required: list[str]
    low_confidence_fields: list[str]
    errors: list[str]

@dataclass
class AgentState:
    doc_path: str
    page_images: list = field(default_factory=list)
    doc_type: DocType = DocType.UNKNOWN
    doc_type_conf: float = 0.0
    scan_quality: float = 0.0              # 0..1 blur/skew/resolution heuristic
    ocr_text: str = ""
    ocr_tokens: list = field(default_factory=list)   # [{text,bbox,conf}]
    ocr_mean_conf: float = 0.0
    ocr_engine: str = "tesseract"
    fields: dict[str, FieldValue] = field(default_factory=dict)
    line_items: list[LineItem] = field(default_factory=list)
    validation: ValidationReport | None = None
    normalized: dict[str, Any] = field(default_factory=dict)
    currency: str | None = None
    status: Status = Status.NEW
    overall_confidence: float = 0.0
    review_reasons: list[str] = field(default_factory=list)
    attempts: dict[str, int] = field(default_factory=dict)
    used_llm_fallback: bool = False
    trace: list[dict] = field(default_factory=list)   # step log for audit
```

### 3.3 Tools (all `state -> state`, pure-ish, mutate + return)

No tool requires a paid API **except** the isolated, optional `llm_vision_fallback`.

1. **`classify_document`** — routing + quality. In: `doc_path`/`page_images`. Out: `doc_type`, `doc_type_conf`, `scan_quality`. How: lightweight image/text classifier (LayoutLMv3-seq or logistic over OCR keywords: "invoice"/"tax invoice" vs "receipt"/"change due"/"POS"); `scan_quality` from Laplacian-variance blur + skew + DPI.
2. **`run_ocr`** — text + per-token confidence. Out: `ocr_text`, `ocr_tokens[{text,bbox,conf}]`, `ocr_mean_conf`, `ocr_engine`. If digital PDF → prefer native text layer (conf≈1.0, exact bboxes). Re-runnable with a different engine on retry.
3. **`extract_layout`** — header fields via fine-tuned LayoutLMv3 (BIO/`S-` tags) or Donut. Out: `fields = {invoice_number, invoice_date, issuer, recipient, subtotal, tax_rate, tax, total, ...}` as `FieldValue(conf, bbox)`. Confidence = mean softmax over the field's tokens.
4. **`extract_line_items`** — table rows via row/col clustering on bboxes + per-cell typing. Out: `line_items=[LineItem(description, quantity, unit_price, amount)]` (qty & unit_price make per-line math checkable — the reference lacked this).
5. **`validate`** — the heart of the agent (pure rules, **no API**). Checks below.
6. **`normalize`** — canonicalize (pure, no API): dates → ISO-8601 `YYYY-MM-DD` (`dateutil` + locale hints from currency/country), amounts → `Decimal(str(x))` 2dp (avoid float money errors), currency → ISO-4217.
7. **`llm_vision_fallback`** — OPTIONAL escalation (the only cloud tool). This is exactly the reference's GPT-4o-vision call, demoted to a fallback. **Guard:** if no API key configured → tool unavailable → agent proceeds to HUMAN REVIEW. System remains fully functional offline.

**`validate` checks:**
- **reconcile:** `sum(item.amount) == subtotal (±ε)`; `subtotal + tax == total (±ε)`; also `tax ≈ subtotal * tax_rate` if `tax_rate` present.
- **per_line:** `quantity * unit_price == amount (±ε)` when both present.
- **date_valid:** parseable & plausible (not future > today, not absurdly old); ambiguous dd/mm vs mm/dd flagged.
- **invoice_no:** plausible pattern (alnum, optional separators, len 3..20).
- **currency:** a symbol/ISO code detected (£/€/$/USD/EUR/VND ...).
- **required_present:** `invoice_number, invoice_date, total, issuer` all non-null.
- **confidence:** collect any field with `conf < FIELD_CONF_MIN`.
- **ε (epsilon):** `max(0.01, 0.005 * total)` — tolerate rounding/penny diffs.

### 3.4 Decision points (3 required)

| # | Where | Condition | Action |
|---|-------|-----------|--------|
| **D1** | after `classify_document` | `doc_type == OTHER` | stop → route out (not an invoice/receipt) |
| | | `scan_quality < Q_MIN` or `ocr_mean_conf < OCR_MIN` | retry: rescan / switch OCR engine (paddle↔tesseract) / deskew; after `MAX_OCR_ATTEMPTS` → **HUMAN REVIEW** |
| **D2** | after `validate` | `not reconciles` **OR** `missing_required` **OR** any required field `conf < FIELD_CONF_MIN` | if LLM fallback available & not tried → escalate, re-validate; else → **HUMAN REVIEW** with reasons |
| **D3** | final gate | `reconciles AND no missing_required AND overall_confidence ≥ AUTO_MIN` | **AUTO-APPROVE** |
| | | otherwise | **HUMAN REVIEW** (attach reasons + bboxes for the UI) |

**Thresholds (config, tunable):** `Q_MIN=0.45`, `OCR_MIN=0.70`, `FIELD_CONF_MIN=0.80`, `AUTO_MIN=0.85`, `MAX_OCR_ATTEMPTS=2`, `ε` as above.
`overall_confidence = min(doc_type_conf, ocr_mean_conf, min(required-field confidences))` — a conservative bottleneck so one shaky required field blocks auto-approval.

### 3.5 Control-flow pseudocode

```python
def run_agent(doc_path: str, cfg: Config) -> AgentState:
    s = AgentState(doc_path=doc_path)
    s.page_images = load_pages(doc_path)            # ALL pages, not just page 1

    # D1: route by doc type + quality
    s = classify_document(s)
    if s.doc_type == DocType.OTHER:
        return finish(s, Status.FAILED, reason="not an invoice/receipt")

    # D1: OCR with quality gate + retry
    engine = "native_pdf" if is_digital_pdf(doc_path) else cfg.default_ocr
    while True:
        s = run_ocr(s, engine=engine)
        if s.ocr_mean_conf >= cfg.OCR_MIN or s.scan_quality >= cfg.Q_MIN:
            break
        s.attempts["ocr"] = s.attempts.get("ocr", 0) + 1
        if s.attempts["ocr"] > cfg.MAX_OCR_ATTEMPTS:
            s.review_reasons.append(f"low OCR confidence ({s.ocr_mean_conf:.2f}); rescan recommended")
            return finish(s, Status.NEEDS_REVIEW)
        engine = next_engine(engine)                # tesseract -> paddleocr -> deskew+retry
        s = preprocess_image(s)                     # deskew/denoise/upscale

    # extraction (route schema by doc type)
    s = extract_layout(s)
    if s.doc_type == DocType.INVOICE:
        s = extract_line_items(s)                   # receipts: line items optional

    # D2: normalize-then-validate
    s = normalize(s)
    s = validate(s)
    needs_help = (not s.validation.reconciles
                  or s.validation.missing_required
                  or s.validation.low_confidence_fields)
    if needs_help and cfg.llm_fallback_enabled and llm_available() and not s.used_llm_fallback:
        s = llm_vision_fallback(s)                  # OPTIONAL cloud escalation
        s = normalize(s); s = validate(s)
        needs_help = (not s.validation.reconciles
                      or s.validation.missing_required
                      or s.validation.low_confidence_fields)

    # D3: final confidence gate
    s.overall_confidence = compute_overall_conf(s)
    if (s.validation.reconciles and not s.validation.missing_required
            and s.overall_confidence >= cfg.AUTO_MIN):
        return finish(s, Status.AUTO_APPROVED)

    if not s.validation.reconciles:
        s.review_reasons.append(
            f"totals don't reconcile: total={s.fields['total'].value} "
            f"vs subtotal+tax off by {s.validation.reconcile_delta}")
    s.review_reasons += [f"missing: {m}" for m in s.validation.missing_required]
    s.review_reasons += [f"low confidence: {f}" for f in s.validation.low_confidence_fields]
    return finish(s, Status.NEEDS_REVIEW)
```

`finish()` stamps `status`, appends to `trace`, returns state to the caller (review-queue writer or auto-post-to-ledger writer). Batch mode wraps `run_agent` in an async-gather loop.

### 3.6 Worked example — totals don't reconcile → flagged for review

**Input:** `INV-2024-077.pdf`, a clean digital-PDF invoice.

| Line | Description | Qty | Unit price | Amount |
|---|---|---|---|---|
| 1 | Consulting services | 10 | 150.00 | 1,500.00 |
| 2 | Onboarding setup | 1 | 300.00 | 300.00 |

Header: `subtotal=1,800.00`, `tax_rate=20%`, `tax=360.00`, **`total=1,860.00`** (the printed total is itself wrong on the document), `invoice_number="INV-2024-077"`, `invoice_date="2024-11-03"`, `currency=GBP`. All field confidences ≥ 0.90.

**Trace:**
1. `classify_document` → INVOICE, conf 0.97, scan_quality 0.92. (D1: pass.)
2. `run_ocr` → engine `native_pdf`, `ocr_mean_conf=0.99`. (D1 gate: pass, no retry.)
3. `extract_layout` + `extract_line_items` → fields + 2 line items.
4. `normalize` → date already ISO; amounts → Decimal; currency → GBP.
5. `validate`:
   - per-line: 10×150.00=1,500.00 ✓; 1×300.00=300.00 ✓
   - `sum(lines)=1,800.00 == subtotal 1,800.00` ✓
   - `tax: 1,800.00×20% = 360.00 == tax 360.00` ✓
   - **reconcile total:** expected `subtotal+tax = 2,160.00`; document `total = 1,860.00`; `reconcile_delta = −300.00`; `ε = max(0.01, 0.005×1,860) = 9.30`; `|−300.00| > 9.30` → **`reconciles=False`**.
   - date_valid ✓, invoice_no ✓, currency ✓, required_present ✓.
6. **D2:** `needs_help=True`. If LLM fallback enabled, it re-reads the same printed `1,860.00` (document is genuinely inconsistent) → re-validation **still fails**. If no key, step skipped — same outcome.
7. **D3:** `reconciles==False` → **not auto-approved.**
8. `finish(NEEDS_REVIEW)` with `review_reasons = ["totals don't reconcile: total=1860.00 vs subtotal+tax=2160.00 (off by -300.00)"]`. The review payload carries every field's `bbox` so the UI highlights the printed total and the two line items for a 5-second human confirmation.

**Contrast with the reference:** ruizguille's pipeline would `Invoice(**data)` successfully (1,860.00 is a valid float), write `Total=1860.00` into Excel, and **silently** add £1,860 to revenue and £360 to the tax summary — the £300 discrepancy vanishes into the aggregates. Our agent **catches it and routes to a human**. That is the core value-add.

---

## 4. DEPLOYMENT (FastAPI)

### 4.1 Endpoints

| Method | Path | Purpose | Auth |
|---|---|---|---|
| `GET` | `/health` | Liveness/readiness, model + version | none |
| `POST` | `/extract` | image/PDF → structured JSON + line items + confidence + bboxes | API key |
| `POST` | `/classify` | doc-type (invoice / receipt / PO / statement / other) | API key |
| `POST` | `/batch` | async multi-file job → `job_id` | API key |
| `GET` | `/batch/{job_id}` | poll batch status/results | API key |
| `GET` | `/review-queue` | list items where `needs_review=true` | API key |
| `POST` | `/review-queue/{id}` | submit human-corrected fields (retraining feedback) | API key |
| `GET` | `/metrics` | Prometheus exposition | internal |

### 4.2 `POST /extract`

**Request** (`multipart/form-data`): `file` (png/jpg/pdf, required) · `doc_type` (optional, skips classify) · `return_crops` (optional bool) · `min_confidence` (optional float, default `0.80`).

**Response** (normalized JSON, per-page merged for multi-page PDF):
```json
{
  "request_id": "req_7f3a9c",
  "model_version": "layoutlmv3-kie-2026-06-20-a1b2c3d",
  "doc_type": "invoice",
  "page_count": 2,
  "processing_ms": 312,
  "fields": {
    "vendor":     {"value":"Acme Corp Ltd","confidence":0.97,"bbox":[62,40,310,72],"page":1},
    "date":       {"value":"2026-05-14","confidence":0.93,"bbox":[420,40,560,70],"page":1,"raw":"14/05/2026"},
    "invoice_no": {"value":"INV-2026-00871","confidence":0.95,"bbox":[420,80,560,108],"page":1},
    "currency":   {"value":"USD","confidence":0.88,"bbox":[500,640,540,668],"page":1},
    "subtotal":   {"value":1240.00,"confidence":0.91,"bbox":[470,600,560,628],"page":1},
    "tax":        {"value":124.00,"confidence":0.90,"bbox":[470,630,560,658],"page":1},
    "total":      {"value":1364.00,"confidence":0.96,"bbox":[470,664,560,694],"page":1}
  },
  "line_items": [
    {"desc":{"value":"Widget A","confidence":0.92,"bbox":[60,300,250,322]},
     "qty":{"value":10,"confidence":0.94,"bbox":[300,300,340,322]},
     "unit_price":{"value":24.00,"confidence":0.89,"bbox":[380,300,440,322]},
     "amount":{"value":240.00,"confidence":0.90,"bbox":[480,300,560,322]},
     "page":1,"row":0}
  ],
  "confidence": 0.92,
  "needs_review": false,
  "review_reasons": [],
  "warnings": ["currency inferred from symbol"],
  "coordinate_space": {"system":"pixel_topleft","page_dims":[[612,792],[612,792]]}
}
```

**`needs_review` triggers** (set `true` if any): a required field (`total, date, vendor, invoice_no`) conf `< min_confidence`; **arithmetic fails** (`abs(subtotal+tax-total) > 0.02*total` or `sum(line_items.amount) ≠ subtotal`); doc-type classifier conf `< 0.70`; no `total` detected / OCR yielded `< N` tokens. `review_reasons` e.g. `["total_below_threshold","arithmetic_mismatch:subtotal+tax!=total"]`.

**Bounding boxes:** returned in **pixel coordinates, top-left origin**, per source page (model's 0–1000 boxes de-normalized to pixels for the UI). `coordinate_space.page_dims` lets the client rescale.

**Error envelope:** `{"request_id":"...","error":{"code":"UNSUPPORTED_MEDIA_TYPE","message":"...","detail":"..."}}`. Codes: `400 BAD_REQUEST`, `413 FILE_TOO_LARGE` (cap ~20 MB / 15 pages), `415 UNSUPPORTED_MEDIA_TYPE`, `422 NO_TEXT_DETECTED`, `429 RATE_LIMITED`, `500 INFERENCE_ERROR`, `503 MODEL_LOADING`.

### 4.3 Other endpoints (shapes)

- **`POST /classify`** → `{"doc_type":"receipt","scores":{"invoice":0.05,"receipt":0.91,...},"confidence":0.91,"needs_review":false}`.
- **`POST /batch`** → `202 {"job_id":"job_4b2","status":"queued","total":48}`; **`GET /batch/{id}`** streams results as **JSONL** (one `/extract` object per line — flat memory); optional `webhook_url` POST on completion. Statuses `queued|running|done|failed|partial`.
- **`GET /review-queue`** → paginated items with `reasons`, `min_field_confidence`, `thumbnail_url`, `extract_url`, `next_cursor`. **`POST /review-queue/{id}`** stores `corrected_fields` + `verdict` for retraining → `{"status":"resolved","stored_for_retraining":true}`.
- **`GET /metrics`** (Prometheus): `extract_requests_total`, `extract_latency_seconds_bucket`, `extract_needs_review_total`, `field_confidence{field,quantile}`, `batch_jobs_inflight`, `model_info{version}`.

### 4.4 Input handling

- **png/jpg** → `PIL.Image.convert("RGB")`, auto-orient via EXIF, optional deskew.
- **PDF** → render each page at **200–300 DPI** (`pdf2image`/`pymupdf`); if embedded text layer exists, prefer its words+boxes (`apply_ocr=False`), else OCR each page image. Cap pages; process pages independently, **merge** (page index retained in every bbox; header fields from page 1 / highest-confidence; line items concatenated across pages; totals reconciled across pages).

### 4.5 Latency targets & batching

| Path | GPU (H100/A10) | CPU (8 vCPU) |
|---|---|---|
| OCR (Tesseract, 1 page) | 150–400 ms | 150–400 ms |
| LayoutLMv3 KIE forward (1 page, ≤512 tok) | 15–40 ms | 200–600 ms |
| Donut forward (generate, 1 page) | 300–800 ms | 4–10 s (not recommended) |
| **End-to-end `/extract`, 1-page image** | **~250–500 ms p95** | **~0.8–1.5 s p95** |
| Throughput (batched, LayoutLMv3) | 40–120 pages/s | 2–6 pages/s |

**Dynamic batching:** queue inference, flush on `max_batch=16` or `max_wait=20ms`. OCR is the bottleneck → OCR in a process pool, GPU forward in a batched micro-service. Keep model warm; `/health` returns `503 MODEL_LOADING` until weights resident.

### 4.6 Gradio highlight demo

Upload → call `/extract` → draw bboxes on the page (green if conf ≥ 0.8 else orange) with `name:conf` labels; outputs: highlighted image, flat fields JSON, line-items JSON, `needs_review` checkbox. Auto-launches on port 7860 for HF Spaces.

### 4.7 Docker / HF Space

```dockerfile
FROM python:3.11-slim
RUN apt-get update && apt-get install -y tesseract-ocr poppler-utils && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt   # pin transformers, torch, gradio, pdf2image, prometheus-client
COPY . .
EXPOSE 7860
CMD ["uvicorn","app:app","--host","0.0.0.0","--port","7860"]
```
- **HF Space:** GPU Space for the Gradio demo; weights in a Hub model repo, loaded by **pinned revision sha**.
- **Production:** FastAPI behind gunicorn/uvicorn workers; OCR in a separate CPU pool; GPU inference container with dynamic batching; Prometheus scrapes `/metrics`.

### 4.8 Model versioning

- Tag every artifact `{family}-{date}-{git_sha}` (e.g. `layoutlmv3-kie-2026-06-20-a1b2c3d`); echo in every response (`model_version`) and `/metrics`.
- Registry: MLflow or HF Hub repo with semantic alias (`@prod`, `@canary`); loader resolves alias → pinned sha at boot.
- Store `label_map` / `id2label` / processor config alongside weights. **Canary** a % of `/extract` traffic to a new version; compare review-rate & field-F1 before promotion.
- Pin `transformers`, `tokenizers`, `Pillow`, OCR-engine versions in the image; record in `/health`.

---

## 5. TRAINING Recipe (H100)

> **API versions confirmed (web search, June 2026):** `transformers` latest is **5.12.1** (2026-06-15); the stable, battle-tested line for LayoutLMv3/Donut is **4.4x–4.5x**. **Pin a version** (e.g. `transformers==4.51.*`) for reproducibility; if on 5.x, re-verify processor/VisionEncoderDecoder signatures (5.0 dropped some deprecated args). In 4.5x+, `Trainer` uses `processing_class=` (the `tokenizer=` arg is deprecated). Entity-level metrics via `evaluate.load("seqeval")`.

### 5.0 Environment + H100 flags

```bash
pip install "transformers==4.51.*" datasets evaluate seqeval accelerate \
            Pillow pytesseract sentencepiece nltk "torch>=2.3"   # CUDA build matching the H100 node
```
```python
import torch
torch.backends.cuda.matmul.allow_tf32 = True   # TF32 on H100
torch.backends.cudnn.allow_tf32 = True
```

### 5.1 RECIPE 1 — LayoutLMv3 token classification (KIE / NER)

**Labels (BIO entity-level).** SROIE keys `{company, date, address, total}` → `O, B/I-COMPANY, B/I-DATE, B/I-ADDRESS, B/I-TOTAL`. (Note: `mp-02/sroie` ships single-token `S-` tags; map to BIO or train on `S-` directly — be consistent in `id2label`.) Build `id2label`/`label2id` from the dataset.

**Processor — two modes.**
```python
from transformers import LayoutLMv3Processor
# (a) Precomputed OCR boxes (RECOMMENDED — SROIE/CORD/FUNSD ship words+boxes):
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
# (b) Raw images only: apply_ocr=True (Tesseract internally)
```

**Preprocessing — image + words + NORMALIZED bboxes (0–1000) + label alignment.**
```python
def normalize_box(box, w, h):     # pixel -> 0..1000 (v3 requirement)
    return [int(1000*box[0]/w), int(1000*box[1]/h),
            int(1000*box[2]/w), int(1000*box[3]/h)]

def preprocess(examples):
    images = [img.convert("RGB") for img in examples["image"]]
    enc = processor(
        images, examples["tokens"], boxes=examples["bboxes"],
        word_labels=examples["ner_tags"],
        truncation=True, padding="max_length", max_length=512,
        stride=128, return_overflowing_tokens=True, return_tensors="pt")
    # When word_labels is passed, the processor sets continuation subwords +
    # special tokens to -100 automatically (CrossEntropyLoss ignores them).
    enc.pop("overflow_to_sample_mapping", None)   # map labels/images per overflow window first if used
    return enc
```
If you tokenize yourself, replicate alignment via `word_ids` (`None`→-100, first subword→label, continuation→-100).

**Model.**
```python
from transformers import LayoutLMv3ForTokenClassification
model = LayoutLMv3ForTokenClassification.from_pretrained(
    "microsoft/layoutlmv3-base", num_labels=len(label_list),
    id2label=id2label, label2id=label2id)
```

**Metrics — seqeval, ENTITY-LEVEL P/R/F1.**
```python
import evaluate, numpy as np
seqeval = evaluate.load("seqeval")
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=2)
    true_pred = [[id2label[a] for a,l in zip(pr,lab) if l!=-100] for pr,lab in zip(preds,p.label_ids)]
    true_lab  = [[id2label[l] for a,l in zip(pr,lab) if l!=-100] for pr,lab in zip(preds,p.label_ids)]
    r = seqeval.compute(predictions=true_pred, references=true_lab)
    return {"precision":r["overall_precision"],"recall":r["overall_recall"],
            "f1":r["overall_f1"],"accuracy":r["overall_accuracy"]}
```

**Trainer (H100: bf16 + tf32).**
```python
from transformers import TrainingArguments, Trainer
args = TrainingArguments(
    output_dir="out/layoutlmv3-kie", learning_rate=5e-5,   # 3e-5..5e-5
    num_train_epochs=8,                                    # SROIE/CORD few; FUNSD small → watch overfit
    per_device_train_batch_size=8, per_device_eval_batch_size=8,
    warmup_ratio=0.1, weight_decay=0.01, lr_scheduler_type="cosine",
    bf16=True, tf32=True,                                  # bf16 on H100 (not fp16) — stable, no loss scaler
    eval_strategy="epoch", save_strategy="epoch",
    load_best_model_at_end=True, metric_for_best_model="f1", greater_is_better=True,
    save_total_limit=3, logging_steps=20, seed=42, report_to="none")
trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
                  processing_class=processor, compute_metrics=compute_metrics)
trainer.train()                                            # resume-safe:
# trainer.train(resume_from_checkpoint=True)  # or "out/layoutlmv3-kie/checkpoint-500"
```

**Class imbalance** (`O` dominates; `B-TOTAL` rare): seqeval entity-F1 already discounts `O`; if a rare class lags, use weighted `CrossEntropyLoss(weight=..., ignore_index=-100)` or focal loss; **report per-entity F1** (seqeval returns it) — don't hide rare-class collapse.

### 5.2 RECIPE 2 — Donut (optional, OCR-free image→JSON)

- `DonutProcessor` + `VisionEncoderDecoderModel` from `naver-clova-ix/donut-base`.
- **GT = JSON serialized to a token sequence:** `{"vendor":"Acme","total":"1364.00"}` → `<s_vendor>Acme</s_vendor><s_total>1364.00</s_total>`; lists joined with `<sep/>`.
- **Add all field keys as special tokens** then `model.decoder.resize_token_embeddings(len(tokenizer))`.
- Set task-start token (`<s_cord>`/`<s_invoice>`), `decoder_start_token_id`, `pad_token_id`, `vocab_size`. Receipts are tall → `image_processor.size = {"height":1280,"width":960}`, `max_length=768`.
- **Labels:** teacher forcing; pad → `-100`. Use **`Seq2SeqTrainer` + `Seq2SeqTrainingArguments`** with `predict_with_generate=True`, `generation_max_length=max_length`, **`bf16=True, tf32=True`**, lower LR (1e-5..3e-5), more epochs (20–40) with **hard early-stop**, `per_device_train_batch_size=2` + `gradient_accumulation_steps=4` (eff. 8).
- **Eval:** normalized tree-edit-distance (nTED, Donut-paper metric) + **field-level F1** (per-key exact-match over flattened `{key:value}`).

### 5.3 Anti-overfitting (small datasets: SROIE~600, CORD~800, FUNSD~150)

1. **Augmentation (geometry-safe — keep boxes valid):** Donut `random_padding=True`, small rotation/scale/brightness/blur, JPEG noise; LayoutLMv3 jitter `pixel_values` but **never** shift boxes inconsistently with words.
2. **Regularization:** `weight_decay 0.01–0.05`, default dropout, label smoothing 0.1.
3. **Low LR + `warmup_ratio=0.1` + cosine decay**; freeze early encoder layers for FUNSD (~150).
4. **Early stopping** on eval F1 (`EarlyStoppingCallback(patience=3–5)`).
5. Fewer epochs for LayoutLMv3 (5–10); Donut 20–40 but early-stop **hard**.
6. **Multiple seeds / CV** (FUNSD variance high at n=150) — report mean±std.
7. **Start from in-domain checkpoints** (`donut-base-finetuned-cord-v2`, LayoutLMv3-fine-tuned-on-FUNSD) and continue — fastest small-data win.
8. Gradient clipping `max_grad_norm=1.0`; `save_total_limit` + `load_best_model_at_end`.
9. Monitor train-vs-eval F1 gap; if eval plateaus while train climbs → stop / add aug.
10. Extreme class rarity → oversample docs with rare entities, or weighted loss.

---

## 6. METRICS + Baseline Plan

### 6.1 Metrics

| Metric | Definition | Tool |
|---|---|---|
| **Entity P/R/F1 per field** | entity-level precision/recall/F1 for COMPANY/DATE/ADDRESS/TOTAL/etc | **seqeval** (`overall_*` + per-entity) |
| **End-to-end field accuracy** | post-normalize exact-match per field over the whole pipeline (OCR→extract→normalize) | custom flatten + exact-match |
| **Line-item F1** | per-row P/R/F1 (matched on description + amount), and per-cell F1 for qty/unit_price/amount | custom |
| **Validation pass-rate** | fraction of docs where `reconciles AND required_present` | from `ValidationReport` |
| **Needs-review rate** | fraction routed to human review (and breakdown by `review_reasons`) | from agent status |
| **Donut nTED / field-F1** | normalized tree-edit-distance + field-F1 (for the Donut path) | nltk edit_distance + custom |
| **Latency** | p50/p95 end-to-end and per-stage (OCR / forward / generate), GPU vs CPU | Prometheus histograms |

**Reporting rules:** always report **per-entity** F1 (not just `overall`) so rare-class collapse (e.g. TOTAL) is visible; report needs-review rate broken down by reason; report latency p95 separately for GPU and CPU.

### 6.2 Baseline plan (the floor every model must beat)

1. **Regex/heuristic over OCR text** (zero training, fully interpretable — the floor):
   - regex for totals (`(?:total|amount due|balance)\D{0,10}([$€£]?\s?\d[\d.,]*)`), dates, invoice numbers, tax IDs;
   - positional heuristics (largest currency value near "total"; vendor = top-of-page largest text block).
2. **`google-bert/bert-base-uncased` + bbox token-classification** (apache-2.0) — coordinates appended/embedded as features. A license-clean middle ground that **isolates how much the layout-pretraining (v3/LiLT) actually buys** over plain BERT + coordinates.
3. **Primary models:** fine-tuned **LayoutLMv3-base** (accuracy benchmark) and **LiLT** (commercial); **Donut** for the line-item / OCR-free comparison.

**Comparison axis:** field-level F1 and end-to-end field accuracy on each dataset's test split (`mp-02/sroie` test 347, `cord-v2` test 100, `funsd-layoutlmv3` test 50). Expectation: regex ≪ bert+bbox < LiLT ≈ LayoutLMv3-base ≤ LayoutLMv3-large for flat fields; Donut > token-cls for nested line-items.

---

## 7. Risks, Pitfalls & Fallbacks

| # | Risk / Pitfall | Mitigation / Fallback |
|---|---|---|
| 1 | **Dataset availability** — invalid ids (`jinhybr/SROIE`, `jordyvl/funsd`, `rvl_cdip` for KIE) | Use **verified** ids only (Section 1). Verified fallbacks: `arvindrajan92/sroie_document_understanding` (MIT), `jsdnrs/ICDAR2019-SROIE` (cc-by-4.0), `rth/sroie-2019-v2`. Pin dataset revision SHAs; download scripts fail loudly if a viewer 404s. |
| 2 | **`darentang/sroie` images missing** (`image_path` is a string, not `Image`) | Prefer `mp-02/sroie` (embedded images). If BIO 9-class is required, supply images locally and join by `image_path`. |
| 3 | **OCR quality** on noisy/rotated/low-DPI scans | Multi-engine: native-PDF text first → PaddleOCR (primary) → docTR (fallback); deskew/denoise/upscale on retry; OCR confidence gate (`OCR_MIN=0.70`) routes bad scans to review after `MAX_OCR_ATTEMPTS`. |
| 4 | **Bbox/label-alignment bugs** (the #1 silent accuracy killer) | Enforce 0–1000 normalization (`int(1000*x/w)`); unit-test that processor sets continuation subwords/special tokens to `-100`; assert `len(words)==len(boxes)==len(labels)`; verify HF-shipped boxes are already normalized before re-normalizing. |
| 5 | **Small-data overfitting** (SROIE~600, CORD~800, FUNSD~150) | Section 5.3 playbook: early stopping on eval F1, low LR + warmup + cosine, geometry-safe aug, weight decay + label smoothing, freeze early layers, start from in-domain checkpoints, multi-seed mean±std. |
| 6 | **Multi-page PDFs** (reference loses all but page 1) | Process **all** pages; retain `page` index in every bbox; merge — header fields from page 1 / highest-confidence, line items concatenated, totals reconciled across pages. |
| 7 | **CPU latency** for layout models / **Donut on CPU impractical** | LayoutLMv3 acceptable on CPU for low volume (~0.8–1.5 s p95). **Do not run Donut on CPU** (4–10 s/page). Dynamic batching + warm models on GPU; OCR in a process pool. |
| 8 | **Commercial license blocker** — LayoutLMv3 is CC-BY-NC-SA-4.0 | Ship **LiLT (MIT)** or **Donut (MIT)** for commercial deployments; keep LayoutLMv3 as the internal accuracy benchmark. Same OCR-token pipeline → swap is cheap. |
| 9 | **LLM fallback unavailable / offline** (no API key) | `llm_vision_fallback` is feature-flagged and isolated; if unavailable, uncertain docs degrade gracefully to **human review**. System never hard-fails on missing network — the strict-dominance guarantee. |
| 10 | **Donut hallucination** at low confidence (no token boxes) | Use Donut only for line-items / OCR-hostile docs; prefer LayoutLMv3/LiLT (grounded, bbox provenance) for fields that feed the review UI; cross-check Donut totals against the validation reconciler. |
| 11 | **transformers version drift** (5.x dropped deprecated args; `tokenizer=`→`processing_class=`) | **Pin `transformers==4.51.*`**; record version in `/health`; if on 5.x, re-verify `LayoutLMv3Processor`/`VisionEncoderDecoder` signatures before training. |
| 12 | **Float money / currency errors** (reference used float) | Normalize to `Decimal(str(x))` 2dp; detect ISO-4217 currency; reconcile with `ε = max(0.01, 0.005*total)` to tolerate penny rounding without masking real discrepancies. |

---

## Appendix — Verified IDs Quick Reference

**Datasets (verified 2026-06-26):** `mp-02/sroie`, `darentang/sroie`, `naver-clova-ix/cord-v2` (cc-by-4.0), `nielsr/funsd-layoutlmv3`, `nielsr/funsd`, `katanaml-org/invoices-donut-data-v1` (MIT), `mychen76/invoices-and-receipts_ocr_v1`, `arvindrajan92/sroie_document_understanding` (MIT), `jsdnrs/ICDAR2019-SROIE` (cc-by-4.0), `rth/sroie-2019-v2`.
**Nonexistent — DO NOT USE:** `jinhybr/SROIE...`, `jordyvl/funsd`, `rvl_cdip` (classification, not KIE).

**Models (verified 2026-06-26):** `microsoft/layoutlmv3-base` (NC), `microsoft/layoutlmv3-large` (NC), `SCUT-DLVCLab/lilt-roberta-en-base` (MIT), `nielsr/lilt-xlm-roberta-base` (MIT), `naver-clova-ix/donut-base` (MIT), `naver-clova-ix/donut-base-finetuned-cord-v2` (MIT), `google-bert/bert-base-uncased` (apache-2.0).
**Corrected:** `microsoft/lilt-roberta-en-base` does **not** exist → use `SCUT-DLVCLab/lilt-roberta-en-base`.
**Avoid:** `microsoft/layoutlmv2-base-uncased` (needs `detectron2`, NC, superseded).

**Library versions:** `transformers` 5.12.1 latest; **pin `4.51.*`** for LayoutLMv3/Donut stability. seqeval for entity-F1. PaddleOCR (primary OCR), docTR (fallback), Tesseract (born-digital). `pdfplumber`/PyMuPDF (born-digital PDFs), `pdf2image`+Poppler (scanned).
