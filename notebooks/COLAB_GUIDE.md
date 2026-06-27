# 🚀 Colab Training Guide (H100 / flexible GPU)

From zero to a fine-tuned LayoutLMv3 extractor + auto-generated report & slides,
using [`Invoice_AI_Colab_Training_H100_AUTOPILOT.ipynb`](Invoice_AI_Colab_Training_H100_AUTOPILOT.ipynb).
You do two things: **(1) put the code on GitHub** and **(2) press Run all.**

---

## Step 1 — Put the project on GitHub (one time)

```bash
cd "04_Invoice_Receipt_Processing"
git init && git add . && git commit -m "Invoice & Receipt Processing System"
git branch -M main
git remote add origin https://github.com/<your-username>/invoice-ai.git
git push -u origin main
```

> `models/`, `artifacts/` and large data are git-ignored.
> **Alternative (no GitHub):** upload the folder to Drive and set `DRIVE_REPO_DIR`.

---

## Step 2 — Google Drive layout (auto-created)

```
MyDrive/
└── NLP_Project/
    └── invoice_ai/
        ├── hf_cache/                          # HuggingFace cache (survives disconnects)
        └── artifacts/
            ├── data/
            ├── models/
            │   └── layout_extractor/latest/    # fine-tuned LayoutLMv3 + labels.json
            ├── runs/                           # eval / benchmark / error-analysis JSON
            └── submission/
                └── submission-<timestamp>/     # report.pdf, slides.pptx, bundle.zip
```

`DRIVE_PROJECT_DIR` (default `NLP_Project/invoice_ai`) sets this path.

---

## Step 3 — Open in Colab and run

1. Upload the notebook to Colab (or open from Drive/GitHub).
2. **Runtime → Change runtime type → GPU.** Prefer **H100**; if unavailable pick
   **A100 / L4 / T4** — the notebook auto-adapts batch size + precision.
3. In **Controls** (cell 0): set `GIT_REPO_URL`. Keep `LAYOUT_MODEL =
   microsoft/layoutlmv3-base` (or switch to the MIT `SCUT-DLVCLab/lilt-roberta-en-base`
   for a commercial build). Leave `RUN_AUTOPILOT = True`.
4. **Runtime → Run all.**

The autopilot (cell 10): install Tesseract+Poppler → fine-tune LayoutLMv3 (seqeval
entity-F1, bf16/TF32, early-stopped, resume-safe) → evaluate → benchmark → error
analysis → **generate `report.pdf` + `slides.pptx`** + grading checklist.

⏱️ On H100, SROIE (~600 docs, 8 epochs) is a few minutes. Use `DEBUG_LIMIT = 100`
for a fast smoke test.

---

## Step 4 — If Colab disconnects (no work lost)

Reconnect → **Runtime → Run all** again. Training **resumes from the last
checkpoint** on Drive; finished steps are skipped.

---

## Step 5 — Test the trained model

* **Cell 13a** runs the validation agent on the built-in synthetic invoices —
  one reconciles, one is **flagged** (printed total 1,860 ≠ 2,160).
* **Cell 13b** loads the **fine-tuned LayoutLMv3** and extracts fields from a real
  test document, printing each field + confidence.

Test on your own document:

```python
from invoice_ai.config import load_config
from invoice_ai.agent.invoice_agent import InvoiceAgent
agent = InvoiceAgent(load_config("configs/train_colab.yaml"))
print(agent.process(doc_path="/content/my_invoice.pdf", filename="my_invoice.pdf").to_dict())
```

---

## Step 6 — Collect your deliverables

Cell 14 prints the submission folder. Download from Drive:
`report.pdf`, `slides.pptx`, `submission_bundle.zip`, `grading_checklist.json`.
Submit the **GitHub link** + the **report** + the **slides**.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Set GIT_REPO_URL ...` in cell 4 | Fill `GIT_REPO_URL` (or `DRIVE_REPO_DIR`) in Controls. |
| H100 unavailable | Pick A100/L4/T4 — the notebook adapts automatically. |
| OOM with layoutlmv3-large | Use `layoutlmv3-base`, or lower `DEBUG_LIMIT`. |
| Tesseract errors on local images | Installed via apt in cell 2; born-digital PDFs need no OCR. |
| Want commercial license | Set `LAYOUT_MODEL = SCUT-DLVCLab/lilt-roberta-en-base` (MIT). |
| Want line-item JSON extraction | Train Donut: `invoice-ai train-donut --config configs/train_colab.yaml`. |
