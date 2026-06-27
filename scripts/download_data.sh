#!/usr/bin/env bash
# Download the KIE datasets (SROIE / CORD / FUNSD) into the HF cache.
set -euo pipefail
python -m invoice_ai.cli data --task all
