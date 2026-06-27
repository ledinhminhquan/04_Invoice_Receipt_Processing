# Project Management & Teamwork

> **Project #4 — Invoice & Receipt Processing System** (`invoice_ai`)
> Author: Le Dinh Minh Quan (23127460). Solo project; team roles below are **simulated** to model how the work would be split and scaled in a real organization. Every fact, threshold, dataset id, and metric here is drawn from the authoritative `docs/DESIGN_BRIEF.md`.

This document covers Assignment §10: a phased delivery timeline, a task breakdown mapped to simulated engineering roles and code modules, a risk register, and a reflection on scaling the system inside a real team.

---

## 1. Phased Timeline (7 weeks)

The plan runs **7 weeks** across seven phases. Each phase has an explicit deliverable and an exit gate so progress is measurable rather than vibes-based. The critical path is **Data + OCR → LayoutLMv3 training → validation agent**, because the agent's value proposition (arithmetic reconciliation) depends on trustworthy fields, which depend on correct bbox-normalized training data.

| Wk | Phase | Scope (from brief) | Key deliverable | Exit gate |
|---|---|---|---|---|
| 1 | **Scope & scaffold** | Repo `invoice_ai`, package layout, `.gitignore` (excludes `data/`, `artifacts/`, weights), config (`Q_MIN`, `OCR_MIN`, `FIELD_CONF_MIN`, `AUTO_MIN`, `MAX_OCR_ATTEMPTS`, `ε`), canonical internal schema | Skeleton repo + `DESIGN_BRIEF.md` frozen | Config + schema reviewed |
| 2 | **Data + OCR pipeline** | Download scripts for verified ids (`mp-02/sroie`, `naver-clova-ix/cord-v2`, `nielsr/funsd-layoutlmv3`, `katanaml-org/invoices-donut-data-v1`); PDF router (`pdfplumber` born-digital vs `pdf2image`+Poppler scanned); OCR engines (native PDF / Tesseract / PaddleOCR) emitting words+boxes+conf | Loaders mapping into canonical schema; OCR returns `ocr_tokens[{text,bbox,conf}]` | Boxes normalized 0–1000; `len(words)==len(boxes)==len(labels)` asserted |
| 3 | **LayoutLMv3 training** | `LayoutLMv3Processor(apply_ocr=False)`; label alignment (continuation subwords + special tokens → `-100`); H100 train (bf16+tf32, lr 5e-5, ~8 epochs early-stopped); seqeval entity-level P/R/F1; `processing_class=` | Fine-tuned `microsoft/layoutlmv3-base` + LiLT (MIT) checkpoints; `id2label`/`label_map` saved with weights | Per-entity F1 reported (no rare-class collapse hidden) |
| 4 | **Line-items + validation agent** | Donut path (`donut-base-finetuned-cord-v2`) for nested line items; `AgentState` blackboard; tools (`classify_document`→`normalize`); `validate` reconciler (`sum(lines)+tax==total`, `qty*unit_price==amount`, `ε=max(0.01,0.005*total)`); 3 decision points D1/D2/D3 | Deterministic state machine + optional `llm_vision_fallback` | Worked example (INV-2024-077, off by −300.00) routes to NEEDS_REVIEW |
| 5 | **API + UI** | FastAPI `/health /extract /classify /batch /review-queue /metrics`; Gradio highlight demo (green ≥0.8 else orange bboxes); Docker (`tesseract-ocr` + `poppler-utils`), HF Space port 7860; `model_version` echoed everywhere | Running service + demo; Prometheus exposition | `/extract` 1-page p95 ~250–500 ms GPU / ~0.8–1.5 s CPU |
| 6 | **Evaluation** | seqeval per-entity F1, end-to-end field accuracy, line-item F1, validation pass-rate, needs-review rate (by reason), latency p50/p95 GPU vs CPU; baseline ladder regex ≪ bert+bbox < LiLT ≈ LayoutLMv3 | Results tables on test splits (sroie 347, cord-v2 100, funsd-layoutlmv3 50) | Primary beats baseline floor on every dataset |
| 7 | **Docs / report** | 10–15 page PDF report, README, model card, license posture (LayoutLMv3 NC = internal only; LiLT/Donut MIT = commercial) | Final report + reproducible notebook | Report compiles; ids/metrics match brief |

```
Wk1 ─ Wk2 ───── Wk3 ───── Wk4 ──────── Wk5 ──── Wk6 ──── Wk7
scope  data+OCR  LLMv3     line-items+   API+UI   eval     docs/
scaffold pipeline training  validation                     report
        ▲ critical path ──────────────▶ (fields must be trustworthy
          before the reconciler adds value)
```

---

## 2. Task Breakdown by Simulated Role

Five roles cover the system. In the solo build, one author wears all five hats sequentially; the mapping below shows how the work decomposes for a real team and which modules each role owns.

| Role | Owns (modules) | Core tasks |
|---|---|---|
| **Data Engineer** | `scripts/` download, dataset loaders, PDF router, OCR adapters | Pull verified ids on demand into git-ignored `data/`; map every source into the canonical schema (`image`+`words`/`tokens`+`bboxes`+`labels`); pin dataset revision SHAs; enforce 0–1000 bbox normalization; born-digital vs scanned routing. |
| **ML / Document-AI Engineer** | training notebook, `extract_layout`, `extract_line_items`, metrics | LayoutLMv3 + LiLT token-classification fine-tune (bf16+tf32, seqeval); Donut seq2seq for line items; label-alignment correctness; anti-overfit playbook (early stop, low LR, multi-seed mean±std on FUNSD); baseline ladder. |
| **Backend Engineer** | FastAPI app, Gradio demo, error envelope | `/extract`/`/classify`/`/batch`/`/review-queue`; multipart handling; multi-page merge (page index in every bbox); pixel-coordinate bbox output + `coordinate_space`; error codes (413/415/422/503); highlight UI. |
| **MLOps Engineer** | Dockerfile, HF Space, model versioning, `/metrics` | Image with `tesseract-ocr`+`poppler-utils`; pinned `transformers==4.51.*`; artifact tags `{family}-{date}-{git_sha}`; `@prod`/`@canary` aliases → pinned SHA; Prometheus histograms; dynamic batching (`max_batch=16`/`max_wait=20ms`); warm-model `503 MODEL_LOADING` gate. |
| **Project Manager** | timeline, risk register, scope guard | Drive the 7-week plan + exit gates; keep assertions matched to the brief (no invented ids/metrics); defend the strict-dominance positioning vs `ruizguille/invoice-processing`; manage the license posture decision (NC vs MIT). |

**Module ↔ agent mapping.** The agent tools (`classify_document`, `run_ocr`, `extract_layout`, `extract_line_items`, `validate`, `normalize`, `llm_vision_fallback`) span Data Eng (OCR), ML Eng (extract), and Backend (orchestration surface). The `validate` reconciler — the heart of the agent — is shared design territory owned jointly by ML and Backend because it is the gate that converts model output into an auto-approve/human-review decision (D2, D3).

---

## 3. Risk Register

The 12 risks below are carried verbatim in posture from the brief's risk table. They are the items most likely to silently degrade accuracy or block deployment.

| # | Risk | Mitigation / fallback | Owner |
|---|---|---|---|
| 1 | Invalid dataset ids (`jinhybr/SROIE`, `jordyvl/funsd`, `rvl_cdip` for KIE) | Verified ids only; fallbacks `arvindrajan92/...` (MIT), `jsdnrs/ICDAR2019-SROIE` (cc-by-4.0); pin revision SHAs; loaders fail loudly on viewer 404 | Data Eng |
| 2 | `darentang/sroie` images missing (`image_path` is a string) | Prefer `mp-02/sroie` (embedded images); supply images locally only if BIO 9-class needed | Data Eng |
| 3 | OCR quality on noisy/rotated/low-DPI scans | Native-PDF → PaddleOCR → docTR; deskew/denoise/upscale on retry; `OCR_MIN=0.70` gate → review after `MAX_OCR_ATTEMPTS=2` | Data Eng |
| 4 | **Bbox/label-alignment bugs (#1 silent killer)** | Enforce `int(1000*x/w)`; unit-test continuation subwords/special tokens → `-100`; assert equal word/box/label lengths; don't re-normalize already-normalized HF boxes | ML Eng |
| 5 | Small-data overfitting (SROIE ~600, CORD ~800, FUNSD ~150) | Early stop on eval F1, low LR + warmup + cosine, geometry-safe aug, weight decay + label smoothing, freeze early layers, in-domain checkpoints, multi-seed mean±std | ML Eng |
| 6 | Multi-page PDFs (reference loses all but page 1) | Process all pages; retain `page` in every bbox; merge header from page 1/highest-conf, concat line items, reconcile totals across pages | Backend |
| 7 | CPU latency / Donut on CPU impractical | LayoutLMv3 OK on CPU (~0.8–1.5 s p95); never run Donut on CPU (4–10 s/page); dynamic batching + warm GPU; OCR in process pool | MLOps |
| 8 | Commercial license blocker — LayoutLMv3 is CC-BY-NC-SA-4.0 | Ship LiLT (MIT) or Donut (MIT) for commercial; keep LayoutLMv3 internal benchmark; same token pipeline → cheap swap | PM / MLOps |
| 9 | LLM fallback unavailable / offline (no API key) | `llm_vision_fallback` feature-flagged + isolated; degrade to human review; system never hard-fails offline (strict-dominance guarantee) | Backend |
| 10 | Donut hallucination at low conf (no token boxes) | Donut only for line items / OCR-hostile docs; prefer LayoutLMv3/LiLT (grounded bboxes) for review-UI fields; cross-check Donut totals against reconciler | ML Eng |
| 11 | `transformers` version drift (5.x dropped args; `tokenizer=`→`processing_class=`) | Pin `transformers==4.51.*`; record version in `/health`; re-verify processor/VisionEncoderDecoder signatures if on 5.x | MLOps |
| 12 | Float money / currency errors (reference used float) | `Decimal(str(x))` 2dp; detect ISO-4217; reconcile with `ε=max(0.01,0.005*total)` to tolerate pennies without masking real gaps | ML Eng / Backend |

---

## 4. Reflection: Scaling in a Real Team

A solo build proves the architecture; a real team would invest in the operational layers a single author can only stub out.

**Annotation tooling.** The verified datasets are small (SROIE ~600, CORD ~800, FUNSD ~150), so a production team's first hire-multiplier is a labeling pipeline. Because LayoutLMv3/LiLT predictions are **grounded to OCR tokens with bboxes**, the natural annotation UI is the same highlight-on-source view the Gradio demo already draws — annotators correct field spans directly on the page, and the corrected `words`/`boxes`/`labels` flow straight back into the canonical schema. This keeps the #1 silent-bug zone (bbox normalization, `-100` alignment) inside one reviewed code path.

**Active learning from the review queue.** The system already routes uncertain documents to human review (D2/D3) and stores `corrected_fields` via `POST /review-queue/{id}` with `stored_for_retraining=true`. That makes the review queue a self-replenishing labeled set: prioritize documents that failed reconciliation or had `conf < FIELD_CONF_MIN` for re-labeling, fold them into the next fine-tune, and watch the needs-review rate fall. The conservative `overall_confidence = min(...)` bottleneck guarantees genuinely shaky fields surface for labeling rather than being auto-approved.

**GPU inference ops.** Scaling throughput means the MLOps patterns in the brief become load-bearing: dynamic batching (`max_batch=16`/`max_wait=20ms`), OCR isolated in a CPU process pool while the GPU forward runs as a batched micro-service, warm models gated by `503 MODEL_LOADING`, and canary rollout of a new `{family}-{date}-{git_sha}` version against a slice of `/extract` traffic — comparing review-rate and field-F1 before promotion via the `@prod`/`@canary` aliases.

**On-call.** Production needs alerting on the Prometheus signals already exposed: rising `extract_needs_review_total` (model drift or a new vendor template), `extract_latency_seconds` p95 regressions, and `model_info{version}` mismatches after a deploy. The graceful-degradation design means an LLM-fallback outage is a quality event (more human review), not an availability incident — the offline path keeps serving.

**Data governance.** The license posture is a governance decision, not a footnote: LayoutLMv3 (CC-BY-NC-SA-4.0) is internal/benchmark only, while LiLT and Donut (MIT) and CORD (cc-by-4.0) / `katanaml-org` invoices (MIT) are the commercial-safe stack. A real team adds PII handling (invoices carry vendor, customer, and tax identifiers), retention rules on the review queue, and a legal sign-off gate before any non-commercially-licensed asset reaches a paying deployment. The brief's "clear with legal before commercial deployment" caveat becomes a checklist item, not an afterthought.
