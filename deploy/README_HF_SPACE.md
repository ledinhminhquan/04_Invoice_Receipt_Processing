# Deploying to a Hugging Face Space (Docker SDK)

The system serves a combined **REST API + Gradio demo** in one process on port
**7860**, and the Docker image bundles **Tesseract** + **Poppler** (required for
OCR and PDF rasterisation).

## Steps

1. Create a Space → SDK = **Docker** → CPU (or GPU for the neural extractor).
2. Push this repo to the Space repo. Add this YAML block to the top of the
   Space's `README.md`:

```yaml
---
title: Invoice & Receipt Processing System
emoji: 🧾
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
---
```

3. The Space builds the `Dockerfile` and starts
   `uvicorn invoice_ai.api.app_combined:app --host 0.0.0.0 --port 7860`.

4. Once live:
   - Demo UI: `https://<user>-<space>.hf.space/ui`
   - API docs: `https://<user>-<space>.hf.space/docs`
   - Health:   `https://<user>-<space>.hf.space/health`

## Notes

* Default deployment needs **no secrets** and runs the regex extractor + validation
  agent fully offline. Upload a fine-tuned `models/layout_extractor/latest` (set
  `INVOICE_AI_MODEL_DIR`) to use the neural extractor.
* Enable the optional LLM-vision fallback with a Space secret
  `INVOICE_AI_LLM_API_KEY` and `agent.llm_fallback_enabled: true`.
* **Licensing:** `microsoft/layoutlmv3-base` is CC-BY-NC-SA (non-commercial). For
  commercial use, set `model.layout_model: SCUT-DLVCLab/lilt-roberta-en-base` (MIT).
