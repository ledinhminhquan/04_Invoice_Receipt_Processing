"""OCR engines: words + bounding boxes + per-token confidence from an image.

Primary = Tesseract (``pytesseract``, easiest on Colab/Docker); optional PaddleOCR
for noisy/rotated scans. All heavy deps are imported lazily and every engine
degrades gracefully — a missing binary returns an empty result and the agent
routes the document to human review instead of crashing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from ..logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class OcrResult:
    tokens: List[dict] = field(default_factory=list)   # [{text, bbox:(x0,y0,x1,y1), conf, page}]
    text: str = ""
    mean_conf: float = 0.0
    engine: str = "none"
    image_size: Tuple[int, int] = (0, 0)               # (width, height) px

    @property
    def ok(self) -> bool:
        return len(self.tokens) > 0


def _tesseract(image, lang: str, page: int) -> OcrResult:
    import pytesseract
    from pytesseract import Output

    w, h = image.size
    data = pytesseract.image_to_data(image, lang=lang, output_type=Output.DICT)
    tokens, confs = [], []
    for i, txt in enumerate(data["text"]):
        txt = (txt or "").strip()
        conf = float(data["conf"][i]) if str(data["conf"][i]) not in ("-1", "") else -1.0
        if not txt or conf < 0:
            continue
        x, y, bw, bh = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        tokens.append({"text": txt, "bbox": (x, y, x + bw, y + bh), "conf": conf / 100.0, "page": page})
        confs.append(conf / 100.0)
    mean = sum(confs) / len(confs) if confs else 0.0
    return OcrResult(tokens=tokens, text=" ".join(t["text"] for t in tokens),
                     mean_conf=mean, engine="tesseract", image_size=(w, h))


def _paddle(image, lang: str, page: int) -> OcrResult:
    import numpy as np
    from paddleocr import PaddleOCR

    ocr = _paddle_singleton(lang)
    arr = np.array(image.convert("RGB"))
    result = ocr.ocr(arr, cls=True)
    tokens, confs = [], []
    for line in (result[0] if result else []):
        box, (txt, conf) = line
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        bbox = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
        tokens.append({"text": txt, "bbox": bbox, "conf": float(conf), "page": page})
        confs.append(float(conf))
    w, h = image.size
    mean = sum(confs) / len(confs) if confs else 0.0
    return OcrResult(tokens=tokens, text=" ".join(t["text"] for t in tokens),
                     mean_conf=mean, engine="paddle", image_size=(w, h))


_PADDLE = None


def _paddle_singleton(lang: str):
    global _PADDLE
    if _PADDLE is None:
        from paddleocr import PaddleOCR
        _PADDLE = PaddleOCR(use_angle_cls=True, lang="en" if lang.startswith("en") else lang, show_log=False)
    return _PADDLE


def ocr_image(image, engine: str = "tesseract", lang: str = "eng", page: int = 1) -> OcrResult:
    """Run OCR on a PIL image. Returns an empty result (never raises) on failure."""
    order = [engine, "tesseract", "paddle"] if engine not in ("tesseract", "paddle") else [engine]
    for eng in order:
        try:
            if eng == "paddle":
                return _paddle(image, lang, page)
            return _tesseract(image, lang, page)
        except Exception as exc:
            logger.warning("OCR engine %s unavailable (%s)", eng, exc)
            continue
    logger.warning("No OCR engine available; returning empty OCR result.")
    return OcrResult(engine="none", image_size=getattr(image, "size", (0, 0)))


def normalize_boxes(tokens: List[dict], image_size: Tuple[int, int]) -> List[List[int]]:
    """Normalise pixel boxes to LayoutLMv3's 0-1000 space (the #1 silent-bug zone)."""
    w, h = image_size
    w = max(1, w)
    h = max(1, h)
    out = []
    for t in tokens:
        x0, y0, x1, y1 = t["bbox"]
        out.append([
            min(1000, max(0, int(1000 * x0 / w))),
            min(1000, max(0, int(1000 * y0 / h))),
            min(1000, max(0, int(1000 * x1 / w))),
            min(1000, max(0, int(1000 * y1 / h))),
        ])
    return out


__all__ = ["OcrResult", "ocr_image", "normalize_boxes"]
