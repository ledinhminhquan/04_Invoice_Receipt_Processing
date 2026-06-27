"""Rubric completeness self-check (PASS/WARN/FAIL per assignment requirement)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from ..logging_utils import get_logger

logger = get_logger(__name__)

_REQUIRED_SRC = [
    "src/invoice_ai/config.py", "src/invoice_ai/cli.py",
    "src/invoice_ai/agent/invoice_agent.py", "src/invoice_ai/agent/validation.py",
    "src/invoice_ai/models/layout_extractor.py", "src/invoice_ai/models/baseline_regex.py",
    "src/invoice_ai/ocr/engine.py", "src/invoice_ai/api/main.py",
    "src/invoice_ai/training/train_layoutlmv3.py",
]
_REQUIRED_DIRS = ["src", "data", "models", "configs", "tests", "docs", "notebooks"]
_REQUIRED_ROOT = ["README.md", "requirements.txt", "pyproject.toml", "Dockerfile"]
_REQUIRED_DOCS = ["problem_definition", "data_description", "model_selection", "deployment",
                  "agent_architecture", "continual_learning_monitoring", "privacy_robustness",
                  "project_plan", "ethics_statement"]


def build_checklist(repo_root) -> Dict:
    root = Path(repo_root)
    items = []

    def check(name, ok, detail, optional=False):
        items.append({"name": name, "status": "PASS" if ok else ("WARN" if optional else "FAIL"), "detail": detail})

    for rel in _REQUIRED_DIRS:
        check(f"Dir: {rel}/", (root / rel).is_dir(), str(root / rel))
    for rel in _REQUIRED_ROOT:
        check(f"File: {rel}", (root / rel).is_file(), str(root / rel))
    for rel in _REQUIRED_SRC:
        check(f"Module: {rel}", (root / rel).is_file(), str(root / rel))
    for d in _REQUIRED_DOCS:
        check(f"Doc: {d}.md", (root / "docs" / f"{d}.md").is_file(), f"docs/{d}.md")
    nb = list((root / "notebooks").glob("*.ipynb")) if (root / "notebooks").is_dir() else []
    check("Notebooks: >=1 .ipynb", len(nb) >= 1, f"{len(nb)} notebook(s)")
    tests = list((root / "tests").glob("test_*.py")) if (root / "tests").is_dir() else []
    check("Tests: >=1 test file", len(tests) >= 1, f"{len(tests)} test file(s)")

    summary = {"PASS": sum(i["status"] == "PASS" for i in items),
               "WARN": sum(i["status"] == "WARN" for i in items),
               "FAIL": sum(i["status"] == "FAIL" for i in items)}
    logger.info("checklist: %s", summary)
    return {"items": items, "summary": summary, "ok": summary["FAIL"] == 0}


def write_checklist(repo_root, out_path) -> Path:
    res = build_checklist(repo_root)
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=2), encoding="utf-8")
    return p


__all__ = ["build_checklist", "write_checklist"]
