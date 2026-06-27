# Invoice & Receipt Processing System (Document-AI)

A **production, offline-first, agentic** Document-AI system that turns invoice /
receipt **images and PDFs** into **validated structured JSON** — with per-field
confidence, bounding-box provenance, and automatic **human-review routing** when
the numbers don't add up. Built for the *NLP in Industry* final assignment (Project #4).

> **Pipeline:** `classify → OCR → extract fields → extract line items → validate
> (totals reconcile?) → normalize → JSON + confidence + needs_review`

It runs **fully offline with no paid API** (regex extractor + OCR + a rules-based
validation agent) and upgrades to a fine-tuned **LayoutLMv3 / Donut** extractor and
an **optional** LLM-vision fallback when configured.

> **Why it beats the reference** (`ruizguille/invoice-processing`, GPT-4o-vision only):
> that pipeline has no OCR, no arithmetic check, no confidence, no human-in-the-loop,
> is online-mandatory and first-page-only — a hallucinated total flows silently into
> the ledger. **Our agent reconciles the arithmetic and flags discrepancies for a
> human.** The GPT-4o call is demoted to one optional fallback tool.

---

## ✨ Highlights

| Requirement (assignment) | How this project delivers it |
|---|---|
| Trainable model + baseline | Fine-tuned **LayoutLMv3** KIE vs **regex/heuristic** + **bert+bbox** baselines |
| Hyperparameter tuning | Optuna sweep (LR / weight-decay) by validation entity-F1 |
| Deployment | **FastAPI** (`/extract /classify /batch /review-queue`) + **Gradio** highlight demo + Docker/HF Space |
| Agentic AI component | 7-tool state machine, **3 decision points**, arithmetic reconciliation, full audit trace |
| Continual learning & monitoring | review-queue corrections → retraining; field-F1 + needs-review drift |
| Data privacy & robustness | offline-first, PII redaction, multi-engine OCR, Decimal money |
| Ethics & validation | reconciliation catches silent errors; human-in-the-loop; bbox provenance |
| Reproducible repo | `src/ data/ models/ configs/ tests/ docs/`, Docker, CI |
| Auto report + slides | one-button **autopilot** → `report.pdf` + `slides.pptx` |

---

## 🗂️ Repository structure

```
04_Invoice_Receipt_Processing/
├── src/invoice_ai/
│   ├── config.py · cli.py · logging_utils.py
│   ├── ocr/            # OCR engines (Tesseract/Paddle) + PDF router (born-digital vs scanned)
│   ├── models/         # LayoutLMv3 extractor, Donut parser, regex baseline, registry
│   ├── agent/          # state, tools, validation (reconciliation), orchestrator, agent
│   ├── training/       # train LayoutLMv3 / Donut, tune, evaluate
│   ├── api/            # FastAPI app, schemas, Gradio UI, combined app
│   ├── analysis/ · autoreport/ · monitoring/ · automation/ · grading/
├── configs/ · data/ · models/ · tests/ · docs/ · notebooks/ · app/ · deploy/ · scripts/ · sample_data/
├── Dockerfile · docker-compose.yml · Makefile · pyproject.toml · requirements*.txt · README.md
```

---

## 📚 Data & models (verified, public)

| Stage | Dataset (HF) | License | Model (HF) | License |
|---|---|---|---|---|
| Receipt KIE | [`mp-02/sroie`](https://huggingface.co/datasets/mp-02/sroie) | research | `microsoft/layoutlmv3-base` | CC-BY-NC-SA ⚠️ |
| Commercial KIE | — | — | `SCUT-DLVCLab/lilt-roberta-en-base` | **MIT** ✅ |
| Line-items (JSON) | [`naver-clova-ix/cord-v2`](https://huggingface.co/datasets/naver-clova-ix/cord-v2) | CC-BY-4.0 | `naver-clova-ix/donut-base` | MIT |
| Form KIE | [`nielsr/funsd-layoutlmv3`](https://huggingface.co/datasets/nielsr/funsd-layoutlmv3) | research | — | — |
| Baseline | — | — | **regex/heuristic** + `bert-base-uncased`+bbox | — |

⚠️ `microsoft/layoutlmv3-base` is **non-commercial** (CC-BY-NC-SA); for commercial
use set `model.layout_model: SCUT-DLVCLab/lilt-roberta-en-base` (MIT, same pipeline).
No large data is committed; download with `invoice-ai data --task all`. Details:
[docs/data_description.md](docs/data_description.md).

---

## 🚀 Quickstart (local)

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt && pip install -e .
# system OCR/PDF tools (Linux): sudo apt-get install tesseract-ocr poppler-utils

# Run the agent on built-in synthetic invoices (offline; shows reconciliation)
invoice-ai demo-agent --config configs/infer.yaml

# Extract from your own file
invoice-ai extract --file sample.pdf --config configs/infer.yaml

# Launch the highlight demo UI (http://localhost:7860)
python app/gradio_app.py
```

### Train + evaluate

```bash
invoice-ai data --task sroie                         # download dataset
invoice-ai train --config configs/train.yaml         # fine-tune LayoutLMv3 (seqeval entity-F1)
invoice-ai evaluate --config configs/train.yaml      # entity-F1 + end-to-end agent eval
```

### Serve the REST API

```bash
invoice-ai serve --config configs/infer.yaml --host 0.0.0.0 --port 8000   # http://localhost:8000/docs
curl -X POST http://localhost:8000/extract -F "file=@invoice.pdf"
```

---

## 🖥️ Train on Google Colab (H100 / flexible GPU)

Open [`notebooks/Invoice_AI_Colab_Training_H100_AUTOPILOT.ipynb`](notebooks/Invoice_AI_Colab_Training_H100_AUTOPILOT.ipynb).
It installs Tesseract + Poppler + Colab-safe deps (never reinstalls torch),
fine-tunes LayoutLMv3 (bf16/TF32, seqeval entity-F1, resume-safe), evaluates, and
**auto-generates the report + slides**. Step-by-step Drive layout + testing:
[`notebooks/COLAB_GUIDE.md`](notebooks/COLAB_GUIDE.md).

---

## 🤖 Agentic component

A deterministic state machine over 7 `state→state` tools with **three decision points**:

1. **Doc-type / quality** — route invoice/receipt/other; retry/switch OCR engine on low confidence → else human review.
2. **Validation** — `subtotal + tax == total` (Decimal, ε-tolerant), per-line `qty × price == amount`, date/number/currency rules. If it fails → optional LLM fallback → else **human review**.
3. **Final gate** — auto-approve only when reconciled **and** confident; otherwise flag for review with reasons + bboxes.

Worked example + diagram: [docs/agent_architecture.md](docs/agent_architecture.md).

---

## 🧰 One-button autopilot

```bash
invoice-ai autopilot --config configs/train.yaml \
  --title "Invoice & Receipt Processing System" --author "Le Dinh Minh Quan"
```

Runs train → evaluate → benchmark → error analysis → **`report.pdf` + `slides.pptx`**
+ `grading_checklist.json` + `submission_bundle.zip` under `artifacts/submission/`.

---

## 🧪 Tests

```bash
pytest -q          # CPU-only; exercises validation + regex + the agent (no model/OCR needed)
```

## 📖 Documentation

Problem · Data · Model Selection · Deployment · **Agent Architecture** · Continual
Learning & Monitoring · Privacy & Robustness · Project Plan · Ethics · **Validation
Evaluation** · Architecture · Model/Data Card · Slide outline — all in [`docs/`](docs/).

## ⚖️ Responsible use

Offline-first keeps invoices on-prem (privacy by design). The agent **never silently
accepts a total that doesn't add up** — it flags it for a human, with each field's
source box for a fast audit. See [docs/ethics_statement.md](docs/ethics_statement.md)
and [docs/privacy_robustness.md](docs/privacy_robustness.md).

## License

MIT (project code) — see [LICENSE](LICENSE). Models/datasets keep their own licenses
(`microsoft/layoutlmv3-base` is **non-commercial**; use LiLT/Donut for commercial).
