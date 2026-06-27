# Invoice & Receipt Processing — container image (REST API + Gradio demo, port 7860).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    INVOICE_AI_ARTIFACTS_DIR=/app/artifacts

# OCR (Tesseract) + PDF rasterisation (Poppler) are required system packages.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl tesseract-ocr poppler-utils libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt pyproject.toml README.md ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY src ./src
COPY configs ./configs
COPY app ./app
COPY docs ./docs
RUN pip install -e .

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -fsS http://localhost:7860/health || exit 1

CMD ["uvicorn", "invoice_ai.api.app_combined:app", "--host", "0.0.0.0", "--port", "7860"]
