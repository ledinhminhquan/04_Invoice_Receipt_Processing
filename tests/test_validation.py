from invoice_ai.config import AppConfig
from invoice_ai.agent.state import AgentState, FieldValue, LineItem
from invoice_ai.agent.validation import normalize, validate, _parse_date


def _state(subtotal, tax, total):
    s = AgentState(filename="x")
    s.fields = {
        "invoice_number": FieldValue("INV-1", 0.9, "rule"),
        "invoice_date": FieldValue("2024-11-03", 0.9, "rule"),
        "vendor": FieldValue("Acme", 0.9, "rule"),
        "subtotal": FieldValue(subtotal, 0.9, "rule"),
        "tax": FieldValue(tax, 0.9, "rule"),
        "total": FieldValue(total, 0.9, "rule"),
        "currency": FieldValue("GBP", 0.9, "rule"),
    }
    return s


def test_reconciles_when_total_correct():
    cfg = AppConfig().agent
    s = validate(normalize(_state(1800.0, 360.0, 2160.0), cfg), cfg)
    assert s.validation.reconciles is True
    assert abs(s.validation.reconcile_delta) < 0.01


def test_flags_wrong_total():
    cfg = AppConfig().agent
    s = validate(normalize(_state(1800.0, 360.0, 1860.0), cfg), cfg)
    assert s.validation.reconciles is False
    assert abs(s.validation.reconcile_delta + 300.0) < 0.01   # delta = total - (subtotal+tax) = -300


def test_epsilon_tolerates_penny_rounding():
    cfg = AppConfig().agent
    s = validate(normalize(_state(100.00, 0.00, 100.01), cfg), cfg)  # within eps for small totals? eps=max(0.01,0.005*100)=0.5
    assert s.validation.reconciles is True


def test_parse_date_formats():
    assert _parse_date("14/05/2026") == "2026-05-14"
    assert _parse_date("2024-11-03") == "2024-11-03"
    assert _parse_date("not a date") is None
