from invoice_ai.config import AppConfig, load_config, save_config
from invoice_ai.ocr.engine import normalize_boxes, OcrResult


def test_default_config():
    cfg = AppConfig()
    assert cfg.model.layout_model == "microsoft/layoutlmv3-base"
    assert cfg.agent.auto_approve_min == 0.85
    assert cfg.agent.llm_fallback_enabled is False


def test_config_roundtrip(tmp_path):
    cfg = AppConfig()
    p = tmp_path / "c.yaml"
    save_config(cfg, p)
    loaded = load_config(p)
    assert loaded.agent.reconcile_eps_rel == cfg.agent.reconcile_eps_rel
    assert loaded.model.max_length == cfg.model.max_length


def test_normalize_boxes_to_0_1000():
    tokens = [{"text": "x", "bbox": (0, 0, 100, 50), "conf": 1.0}]
    boxes = normalize_boxes(tokens, (200, 100))
    assert boxes == [[0, 0, 500, 500]]


def test_ocr_result_empty():
    r = OcrResult()
    assert r.ok is False
