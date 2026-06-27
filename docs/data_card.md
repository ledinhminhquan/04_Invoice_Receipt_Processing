# Dataset Card (DATA_CARD)

> **Project #4 — Invoice & Receipt Processing System** (package `invoice_ai`)
> Author: Le Dinh Minh Quan (student 23127460). All Hugging Face ids below were live-verified on **2026-06-26**.
> This card consolidates the four datasets used to train and benchmark the Key-Information-Extraction (KIE) pipeline. **No large data is committed** to the repo — download scripts in `scripts/` pull on demand into `data/` (git-ignored).

---

## 1. Intended Use

These datasets support a production, offline-first **Document-AI KIE** system that converts invoice/receipt **images and PDFs** into validated, normalized **structured JSON** with per-field confidence, bounding-box provenance, and a `needs_review` flag. They serve two complementary modeling tracks:

- **Token-classification (LayoutLMv3 / LiLT):** flat header-field tagging grounded to OCR tokens with boxes — `mp-02/sroie`, `nielsr/funsd-layoutlmv3`.
- **OCR-free image→JSON (Donut):** nested output including line items — `naver-clova-ix/cord-v2`, `katanaml-org/invoices-donut-data-v1`.

**Primary intended use:** fine-tuning and benchmarking the layout extractor, the line-item extractor, and the baseline floor (regex/heuristic vs. `bert-base-uncased`+bbox vs. LayoutLMv3). Evaluation is on each dataset's held-out test split.

## 2. Composition (verified)

| Dataset id | Domain | Format | Rows (tr / val / te) | Schema (cols) | Label scheme | Ready for | License |
|---|---|---|---|---|---|---|---|
| **`mp-02/sroie`** | Receipts | image + words + bbox + ner | **626 / – / 347** | `image`, `words`, `bboxes`, `ner_tags` (4 cols) | Single-token `S-` tags (5-class): `S-COMPANY, S-DATE, S-ADDRESS, S-TOTAL, O` | **LayoutLMv3** | unknown (per card) |
| **`naver-clova-ix/cord-v2`** | Receipts (line items) | image + `ground_truth` JSON | **800 / 100 / 100** | `image`, `ground_truth` (2 cols) | `gt_parse` JSON: `menu` (`nm,num,cnt,price,itemsubtotal`), `sub_total`, `total`; per-word `quad` boxes + `category` | **Donut** | **cc-by-4.0** |
| **`nielsr/funsd-layoutlmv3`** | Forms | image + tokens + bbox + ner | **149 / – / 50** | `id`, `tokens`, `bboxes`, `ner_tags`, `image` (5 cols) | BIO 7-class: `O, B/I-{HEADER,QUESTION,ANSWER}` | **LayoutLMv3** | unknown |
| **`katanaml-org/invoices-donut-data-v1`** | Invoices | image + gt JSON | **425 / 50 / 26** | `image`, `ground_truth` (same Donut format as CORD) | image→JSON | **Donut** | **MIT** |

**Notes on composition:**
- `mp-02/sroie` has **no validation split** — carve one deterministically (e.g. 10% of train, `seed=42`).
- `mp-02/sroie` ships **single-token `S-` tags, not BIO**. Either map to BIO (`O, B/I-{COMPANY,DATE,ADDRESS,TOTAL}`) or train on `S-` directly — be consistent in `id2label`.
- `cord-v2` `ground_truth` is a JSON **string**; the Donut target is `json.loads(ground_truth)["gt_parse"]`. The per-word `quad` boxes also allow deriving a token-classification version if needed.
- `nielsr/funsd-layoutlmv3` uses the v3 column name **`tokens`** with boxes **pre-normalized to 0–1000**. FUNSD is a generic **form** KIE demonstrator — it is *not* the invoice/receipt target.
- `katanaml-org/invoices-donut-data-v1` is the best small, clean, **commercially-friendly invoice** set (MIT) and shares CORD's Donut format.

**Canonical internal schema** every loader maps into: token-classification sets → `image` + `words`/`tokens` + `bboxes` (normalized 0–1000 for LayoutLMv3) + `labels`; Donut sets → `image` + `ground_truth` JSON.

## 3. Collection & Provenance

- **SROIE** (`mp-02/sroie`): derived from the **ICDAR 2019 Robust Reading Challenge on Scanned Receipts OCR and Information Extraction (SROIE)** — scanned retail receipts annotated for company, date, address, and total.
- **CORD** (`naver-clova-ix/cord-v2`): the **Consolidated Receipt Dataset** released by NAVER CLOVA — Indonesian receipt photos with structured line-item ground truth; the standard receipt-line-item benchmark.
- **FUNSD** (`nielsr/funsd-layoutlmv3`): the **Form Understanding in Noisy Scanned Documents** dataset (from the ICDAR community) — noisy scanned forms with header/question/answer entity links, re-packaged for LayoutLMv3 with pre-normalized boxes.
- **Invoices-Donut** (`katanaml-org/invoices-donut-data-v1`): a curated invoice image→JSON set published under MIT by katanaml-org for Donut fine-tuning.

## 4. Licensing (with the non-commercial caveat)

| Dataset | License | Commercial-safe? |
|---|---|---|
| `naver-clova-ix/cord-v2` | **cc-by-4.0** | Yes (attribution) |
| `katanaml-org/invoices-donut-data-v1` | **MIT** | Yes |
| `mp-02/sroie` | unknown (card declares none) | **Clear with legal first** |
| `nielsr/funsd-layoutlmv3` | unknown | **Clear with legal first** |

> **Non-commercial caveat.** Only `cord-v2` (cc-by-4.0) and `katanaml-org/invoices-donut-data-v1` (MIT) carry **explicit permissive licenses**. **SROIE and FUNSD base data are ICDAR research-challenge data (non-commercial-leaning)**, and several mirrors declare no license. **Clear with legal before commercial deployment.**

**Model-side license interaction (important):** the primary accuracy model `microsoft/layoutlmv3-base` is itself **CC-BY-NC-SA-4.0 (non-commercial)** — it is the **internal/benchmark** model only. For commercial deployment, ship **LiLT** (`SCUT-DLVCLab/lilt-roberta-en-base`, MIT) or **Donut** (MIT), and train on the permissively-licensed datasets above.

**Verified permissive fallbacks** (if SROIE licensing blocks a commercial use): `arvindrajan92/sroie_document_understanding` (**MIT**, enriched line-item labels), `jsdnrs/ICDAR2019-SROIE` (**cc-by-4.0**), `rth/sroie-2019-v2` (de-duplicated from the official RRC source).

## 5. PII & Sensitive-Data Considerations

Invoices and receipts are inherently **PII-bearing financial documents**. The fields these datasets expose include:

- **Vendor / company names and addresses** (`S-COMPANY`, `S-ADDRESS`; FUNSD answers; CORD/Donut header fields).
- **Customer / recipient names and addresses** on invoices.
- **Monetary amounts, totals, tax, line-item prices** — commercially sensitive.
- **Dates, invoice/receipt numbers, tax IDs** — potentially identifying.

**Handling guidance:**
- Treat all rows as containing real-world PII even though they are public research datasets; do **not** re-publish, scrape, or redistribute beyond each license.
- The system stores money as **`Decimal`** and dates as **ISO-8601** — never log raw documents to shared sinks; keep `data/`, `artifacts/`, and model binaries git-ignored (enforced).
- When the optional `llm_vision_fallback` (the only cloud tool) is enabled, document images leave the machine — **disabled by default**; offline runs route uncertain docs to human review instead.
- Human-review feedback (`/review-queue` corrections) may contain PII — store it under the same access controls as production documents.

## 6. Recommended Uses

- Fine-tune **LayoutLMv3-base / LiLT** token classification on `mp-02/sroie` (receipts) and `nielsr/funsd-layoutlmv3` (forms), reporting **entity-level seqeval P/R/F1** including **per-entity** F1 so rare-class collapse (e.g. `TOTAL`) is visible.
- Fine-tune **Donut** for nested line-item extraction on `naver-clova-ix/cord-v2` and `katanaml-org/invoices-donut-data-v1`, reporting **nTED + field-level F1**.
- Use the test splits as the comparison axis: `mp-02/sroie` test **347**, `cord-v2` test **100**, `funsd-layoutlmv3` test **50**.
- Always normalize **bboxes to 0–1000** (`int(1000 * x / w)`) for LayoutLMv3 — the **#1 silent accuracy bug**. CORD/FUNSD-v3 boxes ship pre-normalized; SROIE/raw images must be normalized by you.

## 7. Discouraged / Out-of-Scope Uses

- **Do NOT use these ids** (verified nonexistent or wrong-task): `jinhybr/SROIE…` (does not exist), `jordyvl/funsd` (404), `rvl_cdip` / `aharley/rvl_cdip` / `chainyo/rvl-cdip` (16-class document *classification*, **no KIE field labels**).
- **Avoid `darentang/sroie` for direct image loading** — its `image_path` is a **string path, not an `Image`** (images not bundled). Use only if you supply images yourself; otherwise prefer `mp-02/sroie` (embedded images).
- Do not deploy commercially on SROIE/FUNSD-trained weights without legal sign-off (Section 4).
- Do not treat FUNSD as an invoice/receipt target — it is a generic form-KIE demonstrator.
- Do not use Donut output as ground truth for financial postings without the validation reconciler (Donut can hallucinate fields and provides no token boxes).

## 8. Citations

- **SROIE:** Huang, Z. et al. *ICDAR 2019 Competition on Scanned Receipt OCR and Information Extraction (SROIE).* ICDAR 2019.
- **CORD:** Park, S. et al. *CORD: A Consolidated Receipt Dataset for Post-OCR Parsing.* Document Intelligence Workshop @ NeurIPS 2019. (`naver-clova-ix/cord-v2`)
- **FUNSD:** Jaume, G., Ekenel, H. K., Thiran, J.-P. *FUNSD: A Dataset for Form Understanding in Noisy Scanned Documents.* ICDAR-OST 2019.
- **Donut (model that consumes CORD/Invoices-Donut):** Kim, G. et al. *OCR-free Document Understanding Transformer.* ECCV 2022. (`naver-clova-ix/donut-base`)
- **LayoutLMv3 (token-classification model):** Huang, Y. et al. *LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking.* ACM MM 2022. (`microsoft/layoutlmv3-base`)

---

*Verified Hugging Face dataset ids (2026-06-26):* `mp-02/sroie`, `naver-clova-ix/cord-v2` (cc-by-4.0), `nielsr/funsd-layoutlmv3`, `katanaml-org/invoices-donut-data-v1` (MIT). *Permissive fallbacks:* `arvindrajan92/sroie_document_understanding` (MIT), `jsdnrs/ICDAR2019-SROIE` (cc-by-4.0), `rth/sroie-2019-v2`. *Do not use:* `jinhybr/SROIE…`, `jordyvl/funsd`, `rvl_cdip`.
