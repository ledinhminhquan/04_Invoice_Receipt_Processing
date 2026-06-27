# `sample_data/` — trying the system without real documents

Real invoice images aren't committed (PII / size). Instead the package ships
**synthetic invoices as OCR tokens** (`src/invoice_ai/data/samples.py`) so the full
agent — extraction + arithmetic validation + decision — runs **fully offline**:

```bash
# Run the agent on the built-in synthetic invoices
invoice-ai demo-agent --config configs/infer.yaml
```

You'll see one invoice **reconcile** (subtotal 1800 + tax 360 == total 2160), one
get **flagged for review** (printed total 1860 ≠ 2160, delta −300), and a receipt
that reconciles.

## Test on your own document

```bash
# image or PDF — needs Tesseract installed for scanned files (born-digital PDFs
# work without OCR via pdfplumber)
invoice-ai extract --file /path/to/invoice.pdf --config configs/infer.yaml
```

Or use the Gradio demo (`python app/gradio_app.py`) and upload a file — the UI
highlights each extracted field's bounding box on the page.
