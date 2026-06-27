# Ethics & Responsible AI

> Project #4 — Invoice & Receipt Processing System (`invoice_ai`). Author: Le Dinh Minh Quan (23127460).
> This document addresses Assignment §11. It is grounded in the system's verified design: an offline-first, validation-driven Key-Information-Extraction (KIE) pipeline with per-field confidence, arithmetic reconciliation, bbox provenance, and a human-in-the-loop review queue.

A document-AI system that reads invoices and receipts and writes numbers into a financial ledger touches money, jobs, and trust. The same automation that saves a finance team hours can, if it fails silently, mis-pay a vendor or hide a tax discrepancy. This statement names who benefits, who could be harmed, the concrete risks, and the design choices in `invoice_ai` that mitigate them.

## Who benefits

- **Finance and accounts-payable teams.** Manual key-entry of invoice fields and line items is slow and error-prone. The system extracts `vendor, invoice_no, date, currency, subtotal, tax, total` plus line items into normalized JSON, cutting per-document handling from minutes to roughly **250–500 ms (GPU) / 0.8–1.5 s (CPU)** of machine time plus a short human confirmation only when flagged.
- **Small businesses and on-prem deployments.** Because the default path runs **fully offline** (no paid API), organizations that cannot send financial documents to a cloud LLM can still automate extraction. This strictly dominates the GPT-4o-vision-only reference, which is mandatorily online.
- **Auditors and reviewers.** Every extracted field carries a **bounding box (provenance)** and a **confidence score**, so a reviewer can see exactly where on the page a value came from rather than trusting an opaque number.

## Who could be harmed

| Stakeholder | Potential harm | Cause |
|---|---|---|
| Vendors / payees | Mis-payment, under/over-payment | Wrong total, wrong line amount, or wrong payee extracted |
| Employing business | Bad books, tax mis-statement, fraud slipping through | An incorrect or fabricated total silently entering aggregates |
| Data-entry / AP workers | Job displacement, deskilling | Automation removing routine entry roles |
| Document subjects | Privacy exposure | Invoices contain names, addresses, account details |

The single most dangerous failure mode is a **plausible-but-wrong number flowing silently into financial aggregates** — exactly what the reference pipeline does when it coerces a hallucinated float into a valid `Invoice` and adds it to revenue.

## Risks and mitigations

**1. Automation bias (trusting wrong totals).** Users tend to trust machine output, so a wrong total can be approved unread.
*Mitigation:* The agent never auto-approves on model confidence alone. The **`validate`** step performs arithmetic reconciliation — `sum(line amounts) == subtotal`, `subtotal + tax == total`, `quantity * unit_price == amount` per line, each within `ε = max(0.01, 0.005 * total)`. A document whose printed total does not reconcile is routed to human review with an explicit reason, **even when every field confidence is high**. (In the worked example, a document with a wrong printed total and all confidences ≥ 0.90 is still flagged.)

**2. Bias across invoice formats and languages.** Models trained on mostly English receipts (SROIE ~600, CORD ~800, FUNSD ~150) can systematically under-perform on unfamiliar layouts, scripts, or currencies, mis-extracting for some vendors more than others.
*Mitigation:* Multi-engine OCR (native-PDF → PaddleOCR → docTR) supports rotated, dense, and multilingual (incl. Vietnamese) documents; a LiLT-XLM multilingual option exists. We **report per-entity F1, not only overall F1**, so weak performance on a specific field or class is visible rather than hidden in an average. Uncertain documents degrade to human review instead of silently guessing.

**3. Over-reliance / loss of the human check.** If reviewers rubber-stamp the queue, the safety net erodes.
*Mitigation:* The final gate (D3) auto-approves **only** when the document reconciles, all required fields are present, and `overall_confidence ≥ AUTO_MIN (0.85)`, where `overall_confidence` is a conservative bottleneck — `min(doc_type_conf, ocr_mean_conf, min(required-field confidences))` — so one shaky required field blocks auto-approval. Thresholds (`Q_MIN=0.45`, `OCR_MIN=0.70`, `FIELD_CONF_MIN=0.80`) are explicit and tunable, and `review_reasons` are logged for audit.

**4. Worker displacement.** Automating extraction reduces routine entry work.
*Mitigation (design intent):* The system is positioned as a **human-in-the-loop assistant**, not a replacement. Corrected fields from the review queue are stored for retraining, redirecting human effort from typing toward verification and exception handling.

## Explainability for non-technical stakeholders

The system is built so a finance manager with no ML background can understand and challenge any output:

- **Every field shows its source location and confidence.** The `/extract` response and the Gradio demo draw each value's bounding box on the page (green if conf ≥ 0.8, orange otherwise), so a reviewer literally sees where "total = 1,364.00" was read from.
- **Reconciliation explains *why* a document is flagged** in plain arithmetic, e.g. `"totals don't reconcile: total=1860.00 vs subtotal+tax=2160.00 (off by -300.00)"`, rather than an inscrutable confidence number.
- **Grounded, not generative, for fields.** LayoutLMv3/LiLT cannot invent text — each prediction is tied to a real OCR token and box. Donut (which can hallucinate and gives no token boxes) is restricted to line-item extraction and cross-checked against the reconciler.

```
extract → validate (reconcile + per-line math) → confidence gate
                                  │
              reconciles + complete + conf ≥ 0.85 ──► AUTO-APPROVE
                                  │
              else ──► HUMAN REVIEW  (reasons + bboxes attached)
```

## Misuse and safeguards

- **Fabricating or altering invoices.** A KIE system could be paired with generation to forge invoices, or used to launder altered documents into "verified" records.
  *Safeguard:* The validator is an integrity check, not a rubber stamp — inconsistent arithmetic surfaces tampering rather than smoothing it over. The full `trace` (step log) and provenance boxes make every auto-approval auditable after the fact.
- **Surveillance / over-collection.** Invoices carry personal and commercial data that could be repurposed for profiling vendors or individuals.
  *Safeguard:* **Offline-first operation** means documents need never leave the premises; the optional cloud LLM tool is feature-flagged and isolated, and if no API key is configured the system simply routes hard cases to human review. This minimizes data exposure by default. Deployments should apply retention limits and access controls on the review queue and stored corrections.

## Note on data exposure

Running fully offline materially reduces data-exposure harm: financial documents are processed locally, with no mandatory third-party transmission. Where the optional LLM-vision fallback is enabled, operators should treat it as a deliberate, logged exception (the only cloud tool) and disclose it, since it sends document images to an external provider.

## Licensing as an ethical obligation

Responsible deployment includes respecting model and data licenses. LayoutLMv3 is **CC-BY-NC-SA-4.0 (non-commercial)** and is used here only as an internal accuracy benchmark; commercial deployments must ship the **MIT-licensed LiLT or Donut** path instead. Research datasets (SROIE/FUNSD) lean non-commercial and must be cleared with legal before commercial use.
