# Continual Learning & Monitoring

> **Project #4 — Invoice & Receipt Processing System** (`invoice_ai`). Author: Le Dinh Minh Quan (23127460).
> Assignment §8. How the system **learns from human corrections**, **retrains safely**, **detects degradation**, and **monitors itself** in production — all consistent with the offline-first agent and the `/metrics` exposition defined in the Deployment section.

The system is designed so that every uncertain document a human touches becomes future training signal. The human-in-the-loop (HITL) review queue is not just a safety net — it is the **data-collection flywheel** that drives continual improvement, while a frozen evaluation harness and the Prometheus monitoring module watch for silent regressions.

---

## 8.1 New-data acquisition (the corrections flywheel)

Two streams feed the retraining corpus, both already emitted by the running pipeline:

| Source | Endpoint / mechanism | What is captured | Quality |
|---|---|---|---|
| **Human corrections** | `POST /review-queue/{id}` | `corrected_fields` + `verdict` (accept / fix / reject), keyed to `request_id` and `model_version` | Gold (human-verified) |
| **Extraction logs** | every `/extract` call | full `AgentState.trace`: OCR tokens+boxes, per-field `FieldValue(value, confidence, source, bbox)`, `ValidationReport`, `review_reasons`, `processing_ms` | Silver (model-labeled, weakly verified) |

Items reach a reviewer through the **D2/D3 decision points** (validation failure or `overall_confidence < AUTO_MIN=0.85`). When the reviewer submits, `/review-queue/{id}` stores `corrected_fields` with `stored_for_retraining: true`. Because every field carries its **bbox provenance** (LayoutLMv3/LiLT cannot invent text — predictions are grounded to an OCR token), a correction can be mapped straight back to `words + boxes + label`, producing a **directly re-trainable token-classification example** in the canonical internal schema (`image + words/tokens + bboxes(0–1000) + labels`). Donut corrections are stored as `image + gt_parse` JSON.

This is a hard-negative goldmine: every flagged document is, by construction, one the current model found difficult (low confidence, failed reconciliation, or a new template). Curating these closes the loop the reference (`ruizguille/invoice-processing`) cannot — it has no confidence, no queue, and no corrections to learn from.

---

## 8.2 Retraining loop (champion / challenger + canary)

```
accumulated corrections ──► curate & dedup ──► fine-tune CHALLENGER
   (review-queue store)        (vs frozen           (resume from current
                                eval set)            champion checkpoint)
                                                          │
   frozen eval set ◄───── offline gate ◄──────────────────┘
   (per-entity F1)        (F1 ≥ champion − δ?)
                                │ pass
                                ▼
            CANARY: route a % of /extract traffic to challenger
                    compare review-rate & field-F1 (model_version echoed)
                                │ challenger wins (lower review-rate, ≥ F1)
                                ▼
                    PROMOTE  @canary ──► @prod  (registry alias → pinned sha)
```

- **Periodic fine-tune.** On a fixed cadence (e.g. weekly, or when ≥ N new corrections accumulate), continue-train from the current champion checkpoint on the accumulated corrections **mixed with the original splits** (`mp-02/sroie` 626/347, `cord-v2` 800/100/100, `funsd-layoutlmv3` 149/50) to avoid catastrophic forgetting. Reuse the §5 H100 recipe: `transformers==4.51.*`, `lr=5e-5`, ~8 epochs early-stopped on eval F1, `bf16+tf32`, `seqeval` entity-level metrics, `processing_class=processor`, resume-safe.
- **Champion / challenger.** The live model is the **champion**; the freshly fine-tuned model is the **challenger**. Both are tagged `{family}-{date}-{git_sha}` (e.g. `layoutlmv3-kie-2026-06-20-a1b2c3d`) and resolved via the registry alias (`@prod`, `@canary`) → pinned revision sha.
- **Offline gate.** Before any traffic, the challenger must match or beat the champion on the **frozen eval set** (per-entity seqeval F1, end-to-end field accuracy, line-item F1, validation pass-rate). A challenger that lifts overall F1 but **collapses a rare entity** (e.g. `TOTAL`) is rejected — per-entity F1 is reported, never just `overall`.
- **Canary.** A small percentage of `/extract` traffic is routed to the challenger (the versioning hook already echoes `model_version` in every response and `/metrics`). Promotion criteria: **needs-review-rate does not rise** and **field-F1 holds or improves** on canary traffic; otherwise auto-rollback by flipping the alias back to the previous pinned sha.

Commercial vs internal stays clean here: LayoutLMv3-base (CC-BY-NC-SA-4.0, **non-commercial / internal benchmark**) and LiLT (`SCUT-DLVCLab/lilt-roberta-en-base`, MIT, commercial) share the same OCR-token pipeline, so the same corrections retrain either head — promote LiLT for commercial deployments.

---

## 8.3 Degradation detection

Detection runs on three independent signals so a regression must defeat all of them to stay hidden:

| Signal | What it watches | Trip condition |
|---|---|---|
| **Frozen-eval F1** | per-field seqeval F1 on a held-out, version-pinned eval set re-scored each release/canary | any entity F1 drops > δ vs champion |
| **Needs-review-rate spike** | rolling fraction routed to HITL (overall + by `review_reasons`) | rate exceeds baseline band (e.g. +X% over 7-day trailing) |
| **OCR-confidence drift** | distribution of `ocr_mean_conf` over live traffic | median/p10 shifts below `OCR_MIN=0.70` more often |
| **New-vendor / template drift** | unseen issuer strings, novel layout clusters (bbox geometry), unfamiliar `field_confidence` profile | surge of low-confidence-but-OCR-clean docs |

The **frozen eval set never changes** between releases — it is the absolute yardstick that separates "the model got worse" from "the incoming documents got harder." When the eval set holds steady but the live needs-review-rate climbs, the cause is **input drift** (new templates/locales), and the fix is curation + retraining, not a model rollback. When eval F1 itself drops on a new checkpoint, the challenger is rejected at the offline gate before it ever sees traffic.

---

## 8.4 Monitoring metrics (the monitoring module)

The monitoring module exposes Prometheus metrics at **`GET /metrics`** (scraped in production; `model_info{version}` ties every series to a model build):

| Metric (exposition) | Meaning | Alert intuition |
|---|---|---|
| `field_confidence{field,quantile}` | per-field confidence **quantiles** (p10/p50/p90) | p10 sinking → extraction quality eroding |
| `extract_needs_review_total` (by reason) | needs-review **rate by `review_reasons`** | which failure mode dominates (`arithmetic_mismatch`, `total_below_threshold`, low OCR) |
| `extract_latency_seconds_bucket` | p50/p95 end-to-end **and per-stage** (OCR / forward / generate), GPU vs CPU | p95 breach of ~250–500 ms GPU / ~0.8–1.5 s CPU targets |
| validation pass-rate | fraction with `reconciles AND required_present` (from `ValidationReport`) | drop → upstream OCR/extraction or new-template problem |
| `extract_requests_total`, `batch_jobs_inflight` | throughput / load | capacity + backpressure |

Reporting rules carried over from §6: always break **needs-review-rate down by reason**, always report **per-entity** F1 (so rare-class collapse like `TOTAL` is visible), and report **latency p95 separately for GPU and CPU**.

---

## 8.5 Drift risks & mitigation

| Drift risk | Symptom in monitoring | Mitigation |
|---|---|---|
| **New invoice templates / vendors** | needs-review spike with *clean* OCR but low `field_confidence`; novel layout clusters | route to HITL; harvest `corrected_fields`; periodic fine-tune; geometry-safe augmentation (§5.3) |
| **New currencies / locales** | `currency` check fails or ambiguous dd/mm vs mm/dd dates rise; ISO-4217 not detected | extend currency/locale rules in `normalize`; for non-English, swap to `nielsr/lilt-xlm-roberta-base` (text stream only, **no layout retraining**) |
| **OCR engine changes** (Tesseract ↔ PaddleOCR ↔ docTR, version churn) | `ocr_mean_conf` distribution shift; box-geometry changes break alignment | pin OCR-engine versions (recorded in `/health`); re-validate the **0–1000 bbox normalization** (`int(1000*x/w)` — the #1 silent bug) after any engine swap; multi-engine retry already gated by `OCR_MIN=0.70` |
| **Library drift** (`transformers` 5.x dropped deprecated args) | processor/Trainer signature breaks at retrain time | pin `transformers==4.51.*`; record version in `/health`; re-verify `LayoutLMv3Processor` / `VisionEncoderDecoder` before promoting |
| **Concept drift in totals/tax rules** | reconciliation `ε = max(0.01, 0.005*total)` mismatches cluster | tune ε and tax-rate rules; keep `Decimal` money to avoid float masking real discrepancies |

A drift that the model cannot yet handle never produces a silent wrong answer: it fails the validator or the confidence gate and lands in the review queue — which is exactly where the next batch of training data comes from. **Degradation is converted into supervision.**

---

### Summary

The continual-learning system is a closed loop: HITL corrections (`/review-queue`) + extraction logs → curated training data → champion/challenger fine-tune gated on a **frozen eval set** → canary by **review-rate and field-F1** → promote via registry alias. Monitoring (`/metrics`) watches field-confidence quantiles, needs-review-rate by reason, latency, and validation pass-rate, while three degradation signals (frozen-eval F1, review-rate spikes, OCR drift) plus template/currency/engine drift tracking guarantee that regressions surface as **flagged documents, not corrupted ledgers**.
