"""Gradio demo UI: upload an invoice → highlighted fields + validated JSON.

Two tabs:
  * **Upload** — image/PDF → the agent extracts fields, draws their bounding boxes
    on the page (green = confident, orange = needs attention) and shows the
    validated JSON + the auto-approve / needs-review decision.
  * **Built-in samples** — runs the agent on synthetic invoices fully offline
    (no OCR binary / model needed), demonstrating the arithmetic reconciliation
    that catches a wrong total.
"""

from __future__ import annotations

from typing import Optional

from ..agent.invoice_agent import InvoiceAgent, get_agent
from ..config import AppConfig
from ..data.samples import SAMPLE_DOCS
from ..logging_utils import get_logger

logger = get_logger(__name__)

_BADGE = {"auto_approved": "🟢 AUTO-APPROVED", "needs_review": "🟠 NEEDS REVIEW",
          "failed": "🔴 NOT AN INVOICE/RECEIPT"}


def _fields_md(sd: dict) -> str:
    rows = ["| Field | Value | Conf | Source |", "|---|---|---|---|"]
    for k, fv in sd.get("fields", {}).items():
        rows.append(f"| {k} | {fv.get('value')} | {fv.get('confidence',0):.2f} | {fv.get('source','')} |")
    return "\n".join(rows)


def _annotate(image, sd: dict):
    try:
        from PIL import ImageDraw
        img = image.convert("RGB").copy()
        d = ImageDraw.Draw(img)
        for k, fv in sd.get("fields", {}).items():
            bbox = fv.get("bbox")
            if not bbox:
                continue
            color = (34, 160, 34) if fv.get("confidence", 0) >= 0.8 else (230, 140, 0)
            d.rectangle(list(bbox), outline=color, width=2)
            d.text((bbox[0], max(0, bbox[1] - 10)), f"{k}:{fv.get('confidence',0):.2f}", fill=color)
        return img
    except Exception as exc:
        logger.warning("annotate failed: %s", exc)
        return image


def build_demo(agent: Optional[InvoiceAgent] = None):
    import gradio as gr

    agent = agent or get_agent(AppConfig())

    def upload_fn(filepath):
        if not filepath:
            return None, "Upload an invoice/receipt (PNG/JPG/PDF).", "", ""
        from ..ocr.pdf import load_pages
        loaded = load_pages(filepath, dpi=agent.cfg.ocr.dpi)
        state = agent.process(doc_path=filepath, filename=str(filepath).split("/")[-1])
        sd = state.to_dict()
        badge = _BADGE.get(sd["status"], sd["status"])
        header = (f"### {badge}\n\n**Doc type:** {sd['doc_type']} ({sd['doc_type_conf']:.2f}) · "
                  f"**Confidence:** {sd['overall_confidence']:.2f}\n\n"
                  f"**Reconciles:** {sd['validation']['reconciles'] if sd['validation'] else '—'}")
        reasons = "\n".join(f"- {r}" for r in sd.get("review_reasons", [])) or "_none_"
        img = _annotate(loaded.page_images[0], sd) if loaded.page_images else None
        return img, header, _fields_md(sd), reasons

    def sample_fn(name):
        d = SAMPLE_DOCS[name]
        state = agent.process_tokens(d["tokens"], filename=name)
        sd = state.to_dict()
        badge = _BADGE.get(sd["status"], sd["status"])
        header = (f"### {badge}  ({d['note']})\n\n"
                  f"**Reconciles:** {sd['validation']['reconciles'] if sd['validation'] else '—'} "
                  f"(delta {sd['validation']['reconcile_delta'] if sd['validation'] else '—'})")
        reasons = "\n".join(f"- {r}" for r in sd.get("review_reasons", [])) or "_none_"
        return header, _fields_md(sd), reasons

    with gr.Blocks(title="Invoice & Receipt Processing") as demo:
        gr.Markdown("# 🧾 Invoice & Receipt Processing (Document-AI)\n"
                    "Extracts structured fields from invoices/receipts, **validates the arithmetic**, "
                    "and routes uncertain or non-reconciling documents to human review — fully offline.")
        with gr.Tab("Upload a document"):
            with gr.Row():
                with gr.Column():
                    file_in = gr.File(label="Invoice / receipt (PNG / JPG / PDF)", type="filepath")
                    btn = gr.Button("Extract", variant="primary")
                    out_img = gr.Image(label="Highlighted fields")
                with gr.Column():
                    out_badge = gr.Markdown()
                    out_fields = gr.Markdown(label="Fields")
                    out_reasons = gr.Markdown(label="Review reasons")
            btn.click(upload_fn, [file_in], [out_img, out_badge, out_fields, out_reasons])
        with gr.Tab("Built-in samples (offline)"):
            sample_dd = gr.Dropdown(choices=list(SAMPLE_DOCS), value=list(SAMPLE_DOCS)[0], label="Sample")
            sbtn = gr.Button("Run sample", variant="primary")
            s_badge = gr.Markdown()
            s_fields = gr.Markdown()
            s_reasons = gr.Markdown()
            sbtn.click(sample_fn, [sample_dd], [s_badge, s_fields, s_reasons])
        gr.Markdown("> The agent never silently accepts a total that doesn't add up — it flags it for a human. "
                    "Auto-approval requires reconciliation **and** high confidence (a fine-tuned LayoutLMv3).")
    return demo


def launch(server_name: str = "0.0.0.0", server_port: int = 7860):
    build_demo().queue(max_size=16).launch(server_name=server_name, server_port=server_port)


__all__ = ["build_demo", "launch"]
