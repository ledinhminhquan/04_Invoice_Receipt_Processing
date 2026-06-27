"""Auto-generate the PDF submission report from docs + run artifacts."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from ..config import AppConfig, artifacts_dir
from ..logging_utils import get_logger, utc_stamp
from .artifact_loader import load_all_artifacts
from .charts import build_all_charts

logger = get_logger(__name__)

_DOCS_ORDER = ["problem_definition", "data_description", "model_selection", "agent_architecture",
               "deployment", "validation_evaluation", "continual_learning_monitoring",
               "privacy_robustness", "project_plan", "ethics_statement"]


def _docs_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "docs"


def _strip_md(line: str) -> str:
    line = re.sub(r"`([^`]*)`", r"\1", line)
    line = re.sub(r"\*\*([^*]*)\*\*", r"\1", line)
    line = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", line)
    return line


def generate_report(cfg: AppConfig, title: str = None, author: str = None, out_path=None) -> Path:
    title = title or cfg.project_title
    author = author or cfg.author
    stamp = utc_stamp()
    out = Path(out_path) if out_path else artifacts_dir() / "submission" / f"submission-{stamp}" / "report.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    artifacts = load_all_artifacts()
    charts = build_all_charts(artifacts, out_dir=str(out.parent / "_figures"))

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle)
        from reportlab.lib import colors
    except Exception as exc:
        logger.warning("reportlab unavailable (%s); writing markdown fallback.", exc)
        return _md_fallback(out, title, author, artifacts)

    styles = getSampleStyleSheet()
    story = [Spacer(1, 6 * cm), Paragraph(title, styles["Title"]),
             Paragraph(f"{author} — NLP in Industry, Project #4", styles["Heading3"]),
             Paragraph(f"Generated {stamp}", styles["Normal"]), PageBreak()]

    # metrics table
    ev = (artifacts.get("eval") or {}).get("summary") or {}
    if ev:
        story.append(Paragraph("Evaluation summary", styles["Heading2"]))
        rows = [["Metric", "Value"]] + [[k, f"{v:.3f}" if isinstance(v, (int, float)) else str(v)] for k, v in ev.items()]
        t = Table(rows, hAlign="LEFT")
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2e6ff2")),
                               ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                               ("GRID", (0, 0), (-1, -1), 0.5, colors.grey)]))
        story += [t, Spacer(1, 0.4 * cm)]
    for name, p in charts.items():
        try:
            story.append(Image(str(p), width=14 * cm, height=7 * cm))
        except Exception:
            pass
    story.append(PageBreak())

    docs = _docs_dir()
    for stem in _DOCS_ORDER:
        f = docs / f"{stem}.md"
        if not f.exists():
            continue
        for raw in f.read_text(encoding="utf-8").splitlines():
            s = raw.rstrip()
            if not s:
                continue
            if s.startswith("# "):
                story.append(Paragraph(_strip_md(s[2:]), styles["Heading1"]))
            elif s.startswith("## "):
                story.append(Paragraph(_strip_md(s[3:]), styles["Heading2"]))
            elif s.startswith("### "):
                story.append(Paragraph(_strip_md(s[4:]), styles["Heading3"]))
            elif s.startswith(("- ", "* ", "| ")):
                story.append(Paragraph(_strip_md(s), styles["Normal"]))
            else:
                story.append(Paragraph(_strip_md(s), styles["Normal"]))
        story.append(PageBreak())

    SimpleDocTemplate(str(out), pagesize=A4, title=title, author=author).build(story)
    logger.info("Wrote PDF report -> %s", out)
    return out


def _md_fallback(out: Path, title, author, artifacts) -> Path:
    md = out.with_suffix(".md")
    lines = [f"# {title}", f"_{author} — Project #4_", "",
             "> reportlab not installed — markdown fallback.", ""]
    docs = _docs_dir()
    for stem in _DOCS_ORDER:
        f = docs / f"{stem}.md"
        if f.exists():
            lines.append(f.read_text(encoding="utf-8"))
            lines.append("\n---\n")
    md.write_text("\n".join(lines), encoding="utf-8")
    return md


__all__ = ["generate_report"]
