# Problem Definition Document

> **Project #4 — Invoice & Receipt Processing System** · Package `invoice_ai`
> Author: Le Dinh Minh Quan (23127460) · Offline-first Document-AI Key-Information-Extraction (KIE)
> This document covers Assignment §2 (business context, stakeholders, precise problem, why NLP/Document-AI is required, and success metrics). All facts are aligned with the authoritative `docs/DESIGN_BRIEF.md`.

---

## 1. Business Context

Accounts-Payable (AP) and finance teams still process most invoices and receipts by **manual keying**: a clerk opens a PDF or photo, reads vendor, dates, line items, tax and totals, and retypes them into an ERP or ledger. This is **slow** (minutes per document), **error-prone** (transposed digits, mis-keyed dates, wrong currency), and **costly** at scale. Errors propagate silently into financial aggregates — a single mis-keyed total inflates revenue or tax summaries and is expensive to find downstream in audit.

Two classes of automation tools exist, and **both fall short**:

- **OCR-only tools** convert pixels to text but carry **no validation semantics** — they happily output a `total` that does not equal `subtotal + tax`, and offer no confidence or human-routing signal. Garbage flows through unchecked.
- **Pure-LLM tools** (e.g., the reference `ruizguille/invoice-processing`, GPT-4o-vision only) **hallucinate totals silently**: a fabricated or misread `1,860.00` is a perfectly valid float, so it passes type-coercion, lands in Excel, and corrupts the aggregate with no trace. The reference is also **online-mandatory** (no API key ⇒ nothing runs), **first-page-only**, rasterizes even digital PDFs, and uses **float money**.

This system targets the gap between them: a **local, offline-first** pipeline that combines layout-aware extraction with **arithmetic reconciliation**, **per-field confidence**, **Decimal money**, **multi-page** handling, and **human-in-the-loop** routing — demoting that GPT-4o-vision call to a single *optional* fallback tool.

## 2. Stakeholders

| Stakeholder | Need | What the system gives them |
|---|---|---|
| **AP clerks** | Stop retyping; only touch hard cases | Auto-extracted structured JSON; only `needs_review` items reach a queue with bbox highlights for ~5-second confirmation |
| **Finance / audit** | Trustworthy, traceable numbers | Arithmetic reconciliation (`sum(lines)+tax==total`), per-field provenance bboxes, no silent discrepancies |
| **Procurement** | Match invoices to POs, vendor/line-item visibility | Normalized vendor, line items (desc/qty/unit_price/amount), ISO-8601 dates, ISO-4217 currency |
| **Developers / MLOps** | Integrate, deploy, monitor | FastAPI (`/extract`, `/classify`, `/batch`, `/review-queue`, `/metrics`), versioned models, Prometheus metrics, offline Docker/HF Space |

## 3. Precise Problem Statement

> **Given** an invoice or receipt as an **image (PNG/JPG) or PDF** (born-digital or scanned, possibly multi-page), **produce a validated, normalized structured JSON record** containing header fields and line items, each with a **per-field confidence and bounding box**, plus a document-level **`needs_review`** flag with reasons — routing high-confidence, arithmetically-reconciled documents to **auto-approve** and everything else to **human review**.

The deterministic agent (a state machine, with an optional LLM-vision brain) walks the canonical pipeline:

```
ingest (image/PDF)
  → classify (invoice / receipt / other + scan-quality gate)   [D1]
  → OCR (words + boxes; native-PDF text | Tesseract | PaddleOCR)
  → layout extractor (LayoutLMv3 token classification → header fields)
  → line-item extractor (table → desc / qty / unit_price / amount)
  → validate (totals reconcile, per-line math, date/number/currency)  [D2]
  → normalize (ISO-8601 dates, Decimal money, ISO-4217 currency)
  → structured JSON (+ confidence + bboxes + needs_review)     [D3]
```

**Three decision points** govern routing — **D1** doc-type/quality (`OTHER` stops; low OCR/quality → retry, switch engine, or human review), **D2** validation (not reconciling / missing required / low confidence → LLM fallback if available, else human review), **D3** final gate (reconciles + complete + `overall_confidence ≥ AUTO_MIN` → auto-approve, else review). Thresholds: `Q_MIN=0.45`, `OCR_MIN=0.70`, `FIELD_CONF_MIN=0.80`, `AUTO_MIN=0.85`, `ε = max(0.01, 0.005·total)`. The system runs **fully offline** — the LLM fallback is feature-flagged; with no API key, uncertain documents degrade gracefully to human review rather than hard-failing.

## 4. Why NLP / Document-AI Is Required

This is **not** plain OCR and **not** a fixed-template parser. Invoices and receipts vary wildly across vendors — no two layouts agree on where the total, tax, or invoice number sit. Correct extraction needs the **joint signal of text + 2-D layout + semantics**:

- **Layout matters:** a number is a `total`, a `subtotal`, or a line `amount` depending on its position relative to labels and the table grid — pure text loses this. Models like **LayoutLMv3** (primary accuracy benchmark, internal/non-commercial) and **LiLT** (MIT, commercial-safe) fuse token text with normalized 0–1000 bounding boxes; **Donut** reads image→JSON OCR-free for nested line items.
- **No fixed templates:** rule/regex parsing breaks on every new vendor format; learned token-classification generalizes across layouts.
- **Semantics + grounding:** field meaning (vendor vs recipient, date vs due-date) is semantic, and token-classification models **ground every prediction to an OCR token with a bbox** — essential for the audit trail and the highlight-on-source review UI.
- **A regex/heuristic floor** is retained as a mandatory, interpretable baseline — but it is the floor every learned model must beat, not the solution.

## 5. Success Metrics

Metrics are split into **business outcomes** (what AP/finance care about) and **technical model/pipeline quality**.

### 5.1 Business metrics

| Metric | Definition | Why it matters |
|---|---|---|
| **Manual-entry time saved** | Clerk minutes/document before vs after (auto-approved docs need no keying) | Direct labor-cost reduction |
| **Straight-through-processing (auto-approve) rate** | Fraction of docs passing D3 → `AUTO_APPROVED` without human touch | Core automation ROI |
| **Error / discrepancy catch rate** | Fraction of arithmetic/total discrepancies caught and routed to review (vs silently passed) | The key advantage over OCR-only and pure-LLM tools |
| **Cost per document** | Compute + residual human-review cost per processed document | Operating-cost benchmark vs manual baseline |

### 5.2 Technical metrics

| Metric | Definition | Tool |
|---|---|---|
| **Per-entity F1** | Entity-level precision/recall/F1 per field (COMPANY/DATE/ADDRESS/TOTAL/…); report per-entity to expose rare-class collapse | seqeval |
| **End-to-end field accuracy** | Post-normalize exact-match per field across the full OCR→extract→normalize pipeline | custom flatten + exact-match |
| **Line-item F1** | Per-row P/R/F1 (matched on description + amount) and per-cell F1 for qty/unit_price/amount | custom |
| **Validation pass-rate** | Fraction of docs where `reconciles AND required_present` | from `ValidationReport` |
| **Needs-review rate** | Fraction routed to human review, broken down by `review_reasons` | from agent status |
| **Latency** | p50/p95 end-to-end and per-stage (OCR / forward / generate), GPU vs CPU | Prometheus histograms |

**Latency target:** `/extract` on a 1-page image ≈ **250–500 ms p95 on GPU** and **0.8–1.5 s p95 on CPU**. **Baseline ladder:** regex/heuristic ≪ bert+bbox < LiLT ≈ LayoutLMv3-base on flat fields; Donut > token-classification on nested line items. Reporting rule: always show **per-entity** F1, needs-review broken down by reason, and latency p95 split GPU vs CPU.

---

*This problem definition feeds the 10–15 page PDF report and is consistent with the authoritative `docs/DESIGN_BRIEF.md`.*
