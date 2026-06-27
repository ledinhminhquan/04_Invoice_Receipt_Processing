# Deployment Document

**Project #4 — Invoice & Receipt Processing System** · Package `invoice_ai`
Author: Le Dinh Minh Quan (23127460) · Date: 2026-06-26

This document covers Assignment §6 (deployment). It specifies the serving formats, the
`/extract` inference pipeline, the full endpoint contract with I/O JSON, image/PDF handling,
latency targets, scalability, model versioning, the Docker / Hugging Face Space recipe, and
the deployment challenges with mitigations. Every number, threshold, and id below matches the
authoritative `DESIGN_BRIEF.md`.

---

## 1. Deployment formats

The system ships in four interchangeable surfaces over one shared agent core
(`run_agent` → `AgentState` → normalized JSON). The model is loaded once and reused.

| Format | Surface | Primary use | Entry point |
|---|---|---|---|
| **REST API** | FastAPI app (`app:app`) | Programmatic integration, batch jobs, metrics | `uvicorn app:app --port 7860` |
| **Gradio highlight demo** | Web UI on port 7860 | Visual demo: upload → bbox overlay + JSON | Auto-launch for HF Space |
| **CLI** | `python -m invoice_ai extract <file>` | Local single-file / scripting, offline | stdout JSON |
| **Batch** | `/batch` async job + CLI glob | Folder-scale processing, JSONL output | `job_id` → JSONL |

All four run **fully offline**. The optional `llm_vision_fallback` (the reference's GPT-4o
call, demoted to one tool) is feature-flagged; with no API key the agent degrades gracefully
to human review rather than hard-failing.

---

## 2. Inference pipeline for `/extract`

```
multipart upload (png/jpg/pdf)
   │
   ▼
[1] ingest + page render ──── born-digital PDF → pdfplumber (text+boxes, apply_ocr=False)
   │                          scanned PDF / image → pdf2image @200–300 DPI → OCR
   ▼
[2] classify_document  → doc_type (invoice/receipt/other) + scan_quality        (D1)
   │   OTHER → stop (FAILED);  quality/OCR below gate → retry/switch engine
   ▼
[3] run_ocr            → words + per-token boxes + per-token confidence
   ▼
[4] extract_layout     → header fields via fine-tuned LayoutLMv3 (token-cls)
   │
   ▼
[5] extract_line_items → rows: desc / qty / unit_price / amount
   ▼
[6] normalize          → ISO-8601 dates · Decimal money · ISO-4217 currency
   ▼
[7] validate           → totals reconcile, per-line math, required-present       (D2)
   │   fail → LLM fallback if available, re-validate; else HUMAN REVIEW
   ▼
[8] final gate         → reconciles & complete & conf ≥ AUTO_MIN → AUTO-APPROVE   (D3)
   │                      else needs_review = true (+ reasons + bboxes)
   ▼
normalized JSON response (fields • line_items • confidence • needs_review • coordinate_space)
```

Thresholds (config, tunable): `Q_MIN=0.45`, `OCR_MIN=0.70`, `FIELD_CONF_MIN=0.80`,
`AUTO_MIN=0.85`, `MAX_OCR_ATTEMPTS=2`, `ε = max(0.01, 0.005 * total)`.
`overall_confidence = min(doc_type_conf, ocr_mean_conf, min(required-field confidences))` — a
conservative bottleneck so one shaky required field blocks auto-approval.

---

## 3. Endpoint table

| Method | Path | Purpose | Auth |
|---|---|---|---|
| `GET` | `/health` | Liveness/readiness, model + version | none |
| `POST` | `/extract` | image/PDF → structured JSON + line items + confidence + bboxes | API key |
| `POST` | `/classify` | doc-type (invoice / receipt / other) + quality | API key |
| `POST` | `/batch` | async multi-file job → `job_id` | API key |
| `GET` | `/batch/{job_id}` | poll batch status / stream JSONL results | API key |
| `GET` | `/review-queue` | list items where `needs_review = true` | API key |
| `POST` | `/review-queue/{id}` | submit human-corrected fields (retraining feedback) | API key |
| `GET` | `/metrics` | Prometheus exposition | internal |

### 3.1 `GET /health`

Returns `503 MODEL_LOADING` until weights are resident, then `200`. Echoes the running
`model_version` and pinned library versions for reproducibility.

```json
{
  "status": "ok",
  "model_version": "layoutlmv3-kie-2026-06-20-a1b2c3d",
  "model_loaded": true,
  "ocr_engines": ["tesseract", "paddleocr"],
  "llm_fallback_enabled": false,
  "versions": {"transformers": "4.51.3", "torch": "2.3.1", "pillow": "10.x"}
}
```

### 3.2 `POST /extract`

**Request** (`multipart/form-data`): `file` (png/jpg/pdf, required) · `doc_type` (optional,
skips classify) · `return_crops` (optional bool) · `min_confidence` (optional float, default
`0.80`).

**Response** — normalized JSON, per-page merged for multi-page PDFs:

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

**Normalized response shape.** Every header field is a `FieldValue`:
`{value, confidence, bbox, page}` (plus optional `raw` pre-normalization string). `line_items`
is a list of `{desc, qty, unit_price, amount}` cells (each itself a `FieldValue`) carrying
`page` + `row`. Top-level keys: `confidence` (overall bottleneck), `needs_review`,
`review_reasons`, `warnings`, and `coordinate_space`.

**`needs_review` triggers** (set `true` if any): a required field
(`total, date, vendor, invoice_no`) below `min_confidence`; **arithmetic fails**
(`abs(subtotal + tax − total) > 0.02 * total` or `sum(line_items.amount) ≠ subtotal`);
doc-type classifier conf `< 0.70`; no `total` detected / OCR yielded `< N` tokens.
Example `review_reasons`: `["total_below_threshold","arithmetic_mismatch:subtotal+tax!=total"]`.

**Coordinate space.** Boxes are returned in **pixel coordinates, top-left origin**, per source
page (the model's internal 0–1000 boxes de-normalized to pixels for the UI).
`coordinate_space.page_dims` lets the client rescale to its own render resolution.

### 3.3 Other endpoint shapes

- **`POST /classify`** →
  `{"doc_type":"receipt","scores":{"invoice":0.05,"receipt":0.91,"other":0.04},"confidence":0.91,"scan_quality":0.88,"needs_review":false}`.
- **`POST /batch`** → `202 {"job_id":"job_4b2","status":"queued","total":48}`. Optional
  `webhook_url` is POSTed on completion. Statuses: `queued | running | done | failed | partial`.
- **`GET /batch/{job_id}`** → status JSON, and on completion streams results as **JSONL** —
  one `/extract` object per line, keeping memory flat regardless of job size.
- **`GET /review-queue`** → paginated items with `reasons`, `min_field_confidence`,
  `thumbnail_url`, `extract_url`, `next_cursor`.
- **`POST /review-queue/{id}`** → stores `corrected_fields` + `verdict` for retraining →
  `{"status":"resolved","stored_for_retraining":true}`.
- **`GET /metrics`** (Prometheus): `extract_requests_total`, `extract_latency_seconds_bucket`,
  `extract_needs_review_total`, `field_confidence{field,quantile}`, `batch_jobs_inflight`,
  `model_info{version}`.

### 3.4 Error codes

Error envelope:
`{"request_id":"...","error":{"code":"UNSUPPORTED_MEDIA_TYPE","message":"...","detail":"..."}}`.

| HTTP | Code | Meaning |
|---|---|---|
| 400 | `BAD_REQUEST` | malformed request / missing `file` |
| 413 | `FILE_TOO_LARGE` | exceeds cap (~20 MB / 15 pages) |
| 415 | `UNSUPPORTED_MEDIA_TYPE` | not png/jpg/pdf |
| 422 | `NO_TEXT_DETECTED` | OCR yielded too few tokens |
| 429 | `RATE_LIMITED` | throttle exceeded |
| 500 | `INFERENCE_ERROR` | model forward / generate failed |
| 503 | `MODEL_LOADING` | weights not yet resident |

---

## 4. Image + PDF handling

- **png / jpg** → `PIL.Image.convert("RGB")`, auto-orient via EXIF, optional deskew.
- **Born-digital PDF → `pdfplumber`** (or PyMuPDF/fitz): extracts text **with word-level
  bounding boxes directly** — no OCR, near-perfect, fast. Boxes feed straight into LayoutLMv3
  (`apply_ocr=False`) after 0–1000 normalization.
- **Scanned PDF → `pdf2image`** (`convert_from_path`, Poppler) rasterizes each page at
  **200–300 DPI**, then OCR (PaddleOCR primary, Tesseract for born-digital default) produces
  words + boxes.
- **Router logic:** try `pdfplumber` text-extraction first; if a page yields ~no extractable
  text → treat it as scanned → `pdf2image` + OCR. This born-digital-vs-scanned branch is a
  meaningful accuracy + latency win and fixes the reference's "rasterize everything" waste.

**Multi-page merge.** Pages are capped, processed **independently**, then merged: every bbox
retains its `page` index; header fields are taken from page 1 / highest-confidence; line items
are concatenated across pages preserving order; totals are reconciled across the full document.
This is the direct fix for the reference's first-page-only limitation.

---

## 5. Latency targets

| Path | GPU (H100/A10) | CPU (8 vCPU) |
|---|---|---|
| OCR (Tesseract, 1 page) | 150–400 ms | 150–400 ms |
| LayoutLMv3 KIE forward (1 page, ≤512 tok) | 15–40 ms | 200–600 ms |
| Donut forward (generate, 1 page) | 300–800 ms | 4–10 s (not recommended) |
| **End-to-end `/extract`, 1-page image** | **~250–500 ms p95** | **~0.8–1.5 s p95** |
| Throughput (batched, LayoutLMv3) | 40–120 pages/s | 2–6 pages/s |

OCR — not the layout forward — is the dominant cost, especially on CPU, which drives the
scalability design below.

---

## 6. Scalability

- **Dynamic batching.** Queue inference requests; flush on `max_batch=16` **or**
  `max_wait=20 ms`, whichever fires first. Amortizes the GPU forward across concurrent
  requests without hurting p95.
- **OCR process pool.** OCR is the bottleneck and is CPU-bound, so it runs in a separate
  process pool, decoupled from the batched GPU micro-service. This keeps the GPU saturated
  while OCR scales horizontally on CPU cores.
- **Warm models.** Weights are loaded once at boot and kept resident; `/health` returns
  `503 MODEL_LOADING` until they are. No per-request load cost.
- **Production topology.** FastAPI behind gunicorn/uvicorn workers; OCR in a separate CPU
  pool; GPU inference container with dynamic batching; Prometheus scrapes `/metrics`.

---

## 7. Model versioning

- **Tag scheme** `{family}-{date}-{git_sha}`, e.g. `layoutlmv3-kie-2026-06-20-a1b2c3d`.
  Echoed in every response (`model_version`), in `/health`, and in `/metrics`
  (`model_info{version}`).
- **Registry.** MLflow or an HF Hub model repo with a semantic alias (`@prod`, `@canary`); the
  loader resolves the alias → a **pinned revision sha** at boot (never a floating `main`).
- **Co-located artifacts.** `label_map` / `id2label` / processor config are stored alongside
  the weights so the serving stack and the training stack agree exactly.
- **Canary.** Route a small % of `/extract` traffic to a new version; compare **needs-review
  rate** and **field-F1** against the incumbent before promoting `@canary → @prod`.
- **Pinned environment.** `transformers` (pin `4.51.*`), `tokenizers`, `Pillow`, and OCR-engine
  versions are pinned in the image and reported in `/health`.

---

## 8. Docker + Hugging Face Space

The image must carry the **system** OCR/PDF dependencies (`tesseract-ocr` + `poppler-utils`)
in addition to the Python packages.

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

**HF Space steps:**
1. Create a **GPU Space** for the Gradio highlight demo (CPU Space works for LayoutLMv3 at low
   volume; Donut needs GPU).
2. Store fine-tuned weights in a **Hub model repo**; load them by **pinned revision sha** (not
   `main`) so the Space is reproducible.
3. Provide the `Dockerfile` above (Space SDK = Docker) — the `apt-get` line for `tesseract-ocr`
   + `poppler-utils` is mandatory, otherwise PDF rasterization and Tesseract OCR fail at
   runtime.
4. Space auto-launches on **port 7860**; the Gradio app calls `/extract`, draws bboxes (green
   if conf ≥ 0.8 else orange) with `name:conf` labels, and shows highlighted image + flat
   fields JSON + line-items JSON + a `needs_review` checkbox.

---

## 9. Deployment challenges

| Challenge | Impact | Mitigation |
|---|---|---|
| **OCR bottleneck** | OCR (150–400 ms/page) dwarfs the ~15–40 ms GPU forward; serial OCR caps throughput | Run OCR in a dedicated process pool, decoupled from batched GPU inference; prefer native-PDF text (conf ≈ 1.0, zero OCR) for born-digital docs via the router |
| **Donut impractical on CPU** | Autoregressive decode is 4–10 s/page on CPU | Do **not** run Donut on CPU; reserve it for GPU and for line-item / OCR-hostile docs only; LayoutLMv3 is the CPU-acceptable path (~0.8–1.5 s p95) |
| **Cold start** | First request after boot waits on weight load → high tail latency | Keep models warm (load once at boot, resident); `/health` returns `503 MODEL_LOADING` until ready so load balancers don't route prematurely |
| **Missing system deps** | `pdf2image` / Tesseract fail without `poppler-utils` / `tesseract-ocr` | Bake both into the Docker image (§8); fail loudly at startup if absent |
| **Offline / no LLM key** | `llm_vision_fallback` unavailable | Feature-flagged and isolated; uncertain docs degrade gracefully to **human review**, never a hard failure — the strict-dominance guarantee over the online-mandatory reference |
| **License (production)** | `microsoft/layoutlmv3-base` is CC-BY-NC-SA-4.0 (non-commercial) | Ship **LiLT (MIT)** or **Donut (MIT)** for commercial deployment; keep LayoutLMv3 as the internal accuracy benchmark — identical OCR-token pipeline makes the swap cheap |

---

## 10. Positioning recap

This deployment strictly dominates the reference `ruizguille/invoice-processing` (GPT-4o-vision
only, no OCR / validation / confidence / HITL, online-mandatory, first-page-only, float money).
The `invoice_ai` serving stack adds: a local LayoutLMv3 extractor, arithmetic reconciliation,
Decimal money + ISO-4217 currency, multi-page merge, per-field confidence + bbox provenance,
an offline default path, and a human-review queue — with the reference's GPT-4o call retained
only as one optional, feature-flagged fallback tool.
