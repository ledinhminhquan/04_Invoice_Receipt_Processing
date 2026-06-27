"""Agent + regex tests that run CPU-only with NO model / OCR binary."""
import pytest

from invoice_ai.config import AppConfig
from invoice_ai.agent.invoice_agent import InvoiceAgent
from invoice_ai.agent.state import DocType
from invoice_ai.data.samples import SAMPLE_DOCS
from invoice_ai.models.baseline_regex import extract_fields, _to_amount


@pytest.fixture(scope="module")
def agent():
    cfg = AppConfig()
    cfg.serving.log_extractions = False
    return InvoiceAgent(cfg, load_model=False)


def test_to_amount_parses_plain_and_grouped():
    assert _to_amount("1800.00") == 1800.0
    assert _to_amount("1,800.00") == 1800.0
    assert _to_amount("9.35") == 9.35


def test_regex_total_not_confused_with_subtotal():
    text = "Subtotal 1800.00\nTax 360.00\nTotal 2160.00"
    f = extract_fields(text)
    assert f["subtotal"].value == 1800.0
    assert f["total"].value == 2160.0   # not 1800 from "subtotal"


def test_agent_reconciles_good_invoice(agent):
    s = agent.process_tokens(SAMPLE_DOCS["invoice_ok.png"]["tokens"], filename="ok")
    assert s.doc_type == DocType.INVOICE
    assert s.validation.reconciles is True
    assert s.fields["total"].value == 2160.0


def test_agent_flags_bad_total(agent):
    s = agent.process_tokens(SAMPLE_DOCS["invoice_bad_total.png"]["tokens"], filename="bad")
    assert s.validation.reconciles is False
    assert any("reconcile" in r for r in s.review_reasons)
    assert s.status.value == "needs_review"


def test_agent_trace_and_line_items(agent):
    s = agent.process_tokens(SAMPLE_DOCS["invoice_ok.png"]["tokens"], filename="ok")
    tools = {t.tool for t in s.trace}
    assert "classify_document" in tools and "validate" in tools
    assert len(s.line_items) >= 1
