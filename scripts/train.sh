#!/usr/bin/env bash
# Fine-tune LayoutLMv3 on SROIE, then evaluate.
set -euo pipefail
CONFIG="${1:-configs/train.yaml}"
python -m invoice_ai.cli train    --config "$CONFIG" --dataset sroie
python -m invoice_ai.cli evaluate --config "$CONFIG"
