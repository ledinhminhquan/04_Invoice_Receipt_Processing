# Presentation Slide Deck Outline

> **Project #4 — Invoice & Receipt Processing System** (offline-first Document-AI KIE).
> Author: **Le Dinh Minh Quan (23127460)** · Package `invoice_ai`.
> A 12-slide deck mapping 1:1 onto the 10–15 page PDF report. Each slide lists a title, 3–5 bullets, and a visual/diagram note. Presenter timing target: ~12–15 min + Q&A.

---

## Slide 1 — Title & Project Information

- **Title:** *Invoice & Receipt Processing System — an offline-first Document-AI Key-Information-Extraction (KIE) agent.*
- Author: Le Dinh Minh Quan, student **23127460**; package `invoice_ai`; solo project (team roles simulated).
- One-line thesis: *image/PDF → validated, normalized structured JSON with per-field confidence, bounding-box provenance, and a `needs_review` flag.*
- Deliverables: runnable Python repo, H100 Colab training notebook, FastAPI + Gradio demo, full docs.
- **Visual:** Cover image — a sample invoice on the left with colored bounding boxes; the extracted JSON on the right. Footer: date 2026-06-26, `model_version` tag.

## Slide 2 — Business Problem & Motivation (the silent-total error)

- Manual invoice/receipt entry is slow, costly, and error-prone; accounts-payable teams need structured, auditable data.
- **Core failure mode:** a wrong or hallucinated `total` flows *silently* into financial aggregates (revenue, tax) — no alarm is raised.
- LLM-only extractors coerce any plausible float (e.g. `1860.00`) and write it straight to the ledger; the discrepancy vanishes.
- Requirements that follow: arithmetic **reconciliation**, **confidence**, **human-in-the-loop**, **offline** operation, **Decimal** money.
- **Visual:** "Silent error" before/after panel — a £300 total mismatch slipping into an Excel revenue sum (left) vs. being caught and routed to a human (right).

## Slide 3 — Proposed Solution (vs. the GPT-4o-only reference)

- Local **LayoutLMv3 / Donut** primary extractor + a deterministic **rules engine**; the cloud LLM is one *optional, feature-flagged* fallback tool.
- **Strictly dominates** the reference `ruizguille/invoice-processing` (GPT-4o-vision only): we add OCR, validation, confidence, HITL, multi-page, Decimal money, and offline operation.
- With the network unplugged the system still runs end-to-end — uncertain docs go to human review instead of cloud escalation.
- The reference's *entire* approach (GPT-4o vision) is available here demoted to a single fallback tool.
- **Visual:** Side-by-side comparison table (subset): Primary extractor · Offline · OCR · Validation · Confidence · HITL · Multi-page · Money type — reference "No/None" column vs. `invoice_ai` "Yes" column.

## Slide 4 — System Architecture Diagram

- Canonical pipeline: **ingest → classify → OCR → layout extract → line-item extract → validate → normalize → structured JSON**.
- The agent = **deterministic state machine + optional LLM-vision brain**, all tools reading/writing one shared `AgentState` blackboard.
- OCR router: native-PDF text layer (`pdfplumber`) for born-digital; Tesseract / PaddleOCR for scanned; boxes normalized to **0–1000**.
- Output carries per-field confidence, bboxes (provenance), `needs_review`, and `review_reasons`.
- **Visual (mermaid):**

```mermaid
flowchart LR
    A[ingest image/PDF] --> B{classify<br/>type + quality}
    B -->|OTHER| X[stop / route out]
    B --> C[OCR words+boxes<br/>native | Tesseract | Paddle]
    C --> D[LayoutLMv3<br/>header fields]
    D --> E[line-item extractor]
    E --> F[VALIDATE<br/>reconcile + rules]
    F -. optional .-> L[LLM-vision fallback]
    F --> G[normalize<br/>ISO-8601 / Decimal / ISO-4217]
    G --> H{final gate}
    H -->|reconciles + conf>=AUTO_MIN| AP[AUTO-APPROVE]
    H -->|else| HR[HUMAN REVIEW]
```

## Slide 5 — Data Overview

- LayoutLMv3-ready (token-cls): **`mp-02/sroie`** (626/347, 5-class `S-` tags) and **`nielsr/funsd-layoutlmv3`** (149/50, 7-class BIO).
- Donut-ready (image→JSON, line items): **`naver-clova-ix/cord-v2`** (800/100/100, cc-by-4.0) and **`katanaml-org/invoices-donut-data-v1`** (MIT invoices).
- Licensing driver: only `cord-v2` (cc-by-4.0) and `katanaml-org/...` (MIT) carry clean permissive licenses; SROIE/FUNSD are research-challenge data — clear with legal before commercial use.
- **Verified ids only; DO NOT USE** `jinhybr/SROIE`, `jordyvl/funsd`, `rvl_cdip` (nonexistent / classification-only, wrong task).
- **Visual:** Dataset selection-matrix table (id · format · rows tr/val/te · ready-for · license), with a "verified 2026-06-26" badge and a red "do-not-use" sidebar.

## Slide 6 — Model & Evaluation Results (per-field F1 + baseline table)

- **Primary (accuracy/internal):** `microsoft/layoutlmv3-base` (125M, CC-BY-NC-SA-4.0 → non-commercial); **commercial-safe:** `SCUT-DLVCLab/lilt-roberta-en-base` (MIT).
- **Line items / OCR-free:** `naver-clova-ix/donut-base-finetuned-cord-v2` (MIT, image→JSON); **baseline floor:** regex/heuristic and `google-bert/bert-base-uncased` + bbox.
- Metrics: **per-entity seqeval P/R/F1** (report per-entity so rare-class collapse, e.g. TOTAL, is visible), end-to-end field accuracy, line-item F1.
- Expected ranking (flat fields): **regex ≪ bert+bbox < LiLT ≈ LayoutLMv3-base ≤ LayoutLMv3-large**; **Donut > token-cls** for nested line items.
- **Visual:** Baseline comparison table (model × field-F1 per entity on `mp-02/sroie` test 347, `cord-v2` test 100, `funsd-layoutlmv3` test 50) + a grouped bar chart of overall-F1 per model. *(Note: `microsoft/lilt-roberta-en-base` does NOT exist — use `SCUT-DLVCLab/...`.)*

## Slide 7 — Agentic AI Component (validation + 3 decision points)

- The agent's heart is `validate` — pure rules, no API: **reconcile** `sum(lines)+tax == total (±ε)`, **per-line** `qty*unit_price == amount`, date/number/currency/required-present checks.
- **Three decision points:** **D1** doc-type/quality (OTHER→stop; low OCR conf→retry/switch engine→review); **D2** validation (not reconcile / missing / low conf → LLM fallback if available, else **HUMAN REVIEW**); **D3** final gate (reconciles + complete + conf≥`AUTO_MIN` → **AUTO-APPROVE**, else review).
- Thresholds: `Q_MIN=0.45`, `OCR_MIN=0.70`, `FIELD_CONF_MIN=0.80`, `AUTO_MIN=0.85`, `ε = max(0.01, 0.005*total)`.
- **Worked example (`INV-2024-077`):** lines reconcile to subtotal `1,800.00`, tax `360.00` → expected total `2,160.00`, but the document prints `1,860.00`; `delta = −300.00`, `ε = 9.30` → `reconciles=False` → **flagged for review** (reference would silently book £1,860).
- **Visual:** D1/D2/D3 decision-tree diagram beside the worked totals-mismatch trace (the −300.00 line highlighted in red).

## Slide 8 — Deployment Overview (highlight demo)

- **FastAPI** endpoints: `/health`, `/extract`, `/classify`, `/batch` (+ `/batch/{job_id}`), `/review-queue`, `/metrics`; every response echoes `model_version`.
- **Gradio highlight demo:** upload → `/extract` → draw bboxes on the page (**green if conf ≥ 0.8 else orange**, `name:conf` labels) + fields JSON + `needs_review` checkbox.
- **Docker / HF Space:** image installs **tesseract-ocr + poppler-utils**; auto-launches on **port 7860**; weights pinned by revision sha.
- **Latency:** `/extract` 1-page image **~250–500 ms p95 GPU**, **~0.8–1.5 s p95 CPU**; dynamic batching (`max_batch=16` / `max_wait=20ms`).
- **Visual:** Screenshot mockup of the Gradio demo — invoice with colored boxes on the left, JSON + needs-review panel on the right.

## Slide 9 — Ethics, Privacy & Risks (offline-first, HITL)

- **Offline-first / on-prem:** sensitive financial documents never need to leave the premises; no mandatory cloud API — the strict-dominance guarantee.
- **Human-in-the-loop:** low-confidence or non-reconciling docs are routed to a human with bbox provenance, not auto-posted — auditability by design.
- **License risk:** LayoutLMv3 is **CC-BY-NC-SA-4.0 (non-commercial)** → ship **LiLT (MIT)** or **Donut (MIT)** for commercial deployment; LayoutLMv3 stays an internal benchmark.
- **Other risks:** OCR quality on noisy scans (multi-engine + retry gate), the **0–1000 bbox normalization** silent bug (#1 accuracy killer), small-data overfitting, Donut hallucination (no token boxes), float-money errors (use `Decimal`).
- **Visual:** Risk → Mitigation table (offline/HITL/license/OCR/bbox-bug/overfit) with a "non-commercial license" warning callout.

## Slide 10 — Continual Learning & Monitoring (review-queue feedback loop)

- **Feedback loop:** `GET /review-queue` lists `needs_review` items; `POST /review-queue/{id}` stores `corrected_fields` + verdict (`stored_for_retraining: true`) for the next training round.
- **Monitoring:** Prometheus `/metrics` — `extract_requests_total`, `extract_latency_seconds_bucket`, `extract_needs_review_total`, `field_confidence{field,quantile}`, `model_info{version}`.
- **Versioning:** tag artifacts `{family}-{date}-{git_sha}`; **canary** a % of `/extract` traffic; compare review-rate & field-F1 before promotion.
- Track **needs-review rate broken down by reason** and per-field confidence quantiles over time to catch drift.
- **Visual:** Closed-loop diagram — `extract → review-queue → human corrections → retrain → canary → promote → extract`.

## Slide 11 — Key Takeaways & Future Work

- **Takeaway:** validation + confidence + HITL turn a brittle LLM script into an auditable agent that *catches* the silent-total error instead of booking it.
- **Takeaway:** offline-first + Decimal money + multi-page merge + bbox provenance = production-grade, on-prem, commercially shippable (via LiLT/Donut MIT).
- **Future work:** multilingual invoices via `nielsr/lilt-xlm-roberta-base` (swap text stream, no layout retraining); scale Donut on `mychen76/invoices-and-receipts_ocr_v1` (verify license).
- **Future work:** active-learning from the review queue; per-vendor template memory; tighten line-item table-structure extraction; richer cross-page total reconciliation.
- **Visual:** Two-column "Shipped vs. Next" summary card; a small roadmap timeline.

## Slide 12 — Q&A

- Thank-you slide; contact + repo path `D:/NLP Industry Projects/04_Invoice_Receipt_Processing` and HF Space URL.
- Anticipated questions: *Why not GPT-4o only?* (offline + reconciliation + provenance); *Why LayoutLMv3 if non-commercial?* (internal accuracy benchmark; LiLT for commercial).
- *How is confidence computed?* (`overall = min(doc_type_conf, ocr_mean_conf, min(required-field conf))` — conservative bottleneck).
- *What happens with no API key / no network?* (LLM fallback disabled → graceful degrade to human review; system never hard-fails).
- **Visual:** Minimal closing slide — thesis restated in one line + the highlighted-invoice cover image reused as a backdrop.

---

### Mapping to the report

| Slide | Report section |
|---|---|
| 1 Title | Cover / front matter |
| 2 Problem | §0.3 Motivation |
| 3 Solution | §0.1–0.3 (positioning table) |
| 4 Architecture | §0.2, §3.1 |
| 5 Data | §1 Data stack |
| 6 Models & Eval | §2 Models, §6 Metrics/baseline |
| 7 Agent | §3.3–3.6 (validate, D1–D3, worked example) |
| 8 Deployment | §4 FastAPI + Gradio |
| 9 Ethics/Risks | §7 Risks + license caveats |
| 10 Continual learning | §4.1/§4.3 review-queue, §4.8 versioning |
| 11 Takeaways/Future | §0, §2.2, §5.3 |
| 12 Q&A | — |
