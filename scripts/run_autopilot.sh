#!/usr/bin/env bash
# One button: train -> evaluate -> analysis -> auto-generate report + slides.
set -euo pipefail
python -m invoice_ai.cli autopilot \
  --config "${1:-configs/train.yaml}" \
  --title "Invoice & Receipt Processing System" \
  --author "Le Dinh Minh Quan"
