.DEFAULT_GOAL := help
PY ?= python

.PHONY: help install install-all data train evaluate serve demo extract autopilot report slides grade test lint docker clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
	awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install (core deps) in editable mode
	pip install -r requirements.txt && pip install -e .

install-all: ## Install with all optional extras
	pip install -e ".[all,dev]"

data: ## Download datasets
	$(PY) -m invoice_ai.cli data --task all

train: ## Fine-tune LayoutLMv3 KIE
	$(PY) -m invoice_ai.cli train --config configs/train.yaml

evaluate: ## Entity-F1 + end-to-end agent evaluation
	$(PY) -m invoice_ai.cli evaluate --config configs/train.yaml

serve: ## Start the FastAPI server (port 8000)
	$(PY) -m invoice_ai.cli serve --config configs/infer.yaml --host 0.0.0.0 --port 8000

demo: ## Launch the Gradio demo (port 7860)
	$(PY) app/gradio_app.py

extract: ## Run the agent on built-in samples
	$(PY) -m invoice_ai.cli demo-agent --config configs/infer.yaml

autopilot: ## One button: train -> eval -> analysis -> report + slides
	$(PY) -m invoice_ai.cli autopilot --config configs/train.yaml

report: ## Generate the PDF report
	$(PY) -m invoice_ai.cli generate-report --config configs/train.yaml

slides: ## Generate the PPTX slide deck
	$(PY) -m invoice_ai.cli generate-slides --config configs/train.yaml

grade: ## Rubric completeness self-check
	$(PY) -m invoice_ai.cli grade

test: ## Run the test suite
	pytest -q

lint: ## Lint with ruff
	ruff check src tests

docker: ## Build the Docker image
	docker build -t invoice-ai:1.0.0 .

clean: ## Remove caches + build artifacts
	rm -rf .pytest_cache .ruff_cache **/__pycache__ build dist *.egg-info
