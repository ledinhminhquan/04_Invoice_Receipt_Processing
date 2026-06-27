"""Document loader + born-digital-vs-scanned router.

* **Images** (png/jpg/…) → a single RGB page, OCR required.
* **Born-digital PDFs** → ``pdfplumber`` extracts words **with boxes** directly
  (no OCR, near-perfect) — these become high-confidence native tokens.
* **Scanned PDFs** → ``pdf2image`` rasterises each page → OCR required.

The router tries the text layer first; a page with ~no extractable text is
treated as scanned. All heavy deps are imported lazily and degrade gracefully.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from ..logging_utils import get_logger

logger = get_logger(__name__)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


@dataclass
class LoadedDoc:
    page_images: List = field(default_factory=list)        # PIL.Image per page
    native_tokens: Optional[List[dict]] = None             # [{text,bbox,conf,page}] if born-digital
    is_digital: bool = False
    n_pages: int = 0


def load_pages(path: str | Path, dpi: int = 250) -> LoadedDoc:
    p = Path(path)
    ext = p.suffix.lower()
    if ext in IMAGE_EXTS:
        from PIL import Image
        img = Image.open(p).convert("RGB")
        return LoadedDoc(page_images=[img], native_tokens=None, is_digital=False, n_pages=1)
    if ext == ".pdf":
        return _load_pdf(p, dpi)
    # Unknown: try as image, else empty.
    try:
        from PIL import Image
        return LoadedDoc(page_images=[Image.open(p).convert("RGB")], n_pages=1)
    except Exception as exc:
        logger.warning("Unsupported document %s (%s)", p, exc)
        return LoadedDoc()


def _load_pdf(p: Path, dpi: int) -> LoadedDoc:
    # 1) Try born-digital text layer with pdfplumber.
    try:
        import pdfplumber

        images, tokens = [], []
        with pdfplumber.open(str(p)) as pdf:
            scale = dpi / 72.0
            for pi, page in enumerate(pdf.pages, start=1):
                words = page.extract_words() or []
                for w in words:
                    bbox = (int(w["x0"] * scale), int(w["top"] * scale),
                            int(w["x1"] * scale), int(w["bottom"] * scale))
                    tokens.append({"text": w["text"], "bbox": bbox, "conf": 1.0, "page": pi})
                try:
                    images.append(page.to_image(resolution=dpi).original.convert("RGB"))
                except Exception:
                    images.append(None)
        if tokens:  # born-digital
            logger.info("Born-digital PDF: %d pages, %d native tokens", len(images), len(tokens))
            return LoadedDoc(page_images=[im for im in images if im is not None] or images,
                             native_tokens=tokens, is_digital=True, n_pages=len(images))
    except Exception as exc:
        logger.info("pdfplumber failed (%s); trying rasterisation.", exc)

    # 2) Scanned: rasterise with pdf2image (Poppler).
    try:
        from pdf2image import convert_from_path
        images = convert_from_path(str(p), dpi=dpi)
        images = [im.convert("RGB") for im in images]
        logger.info("Scanned PDF rasterised: %d pages", len(images))
        return LoadedDoc(page_images=images, native_tokens=None, is_digital=False, n_pages=len(images))
    except Exception as exc:
        logger.warning("Could not rasterise PDF %s (%s); pdfplumber image fallback.", p, exc)

    # 3) Last resort: pdfplumber images only.
    try:
        import pdfplumber
        with pdfplumber.open(str(p)) as pdf:
            images = [page.to_image(resolution=dpi).original.convert("RGB") for page in pdf.pages]
        return LoadedDoc(page_images=images, n_pages=len(images))
    except Exception as exc:
        logger.warning("PDF load failed entirely for %s (%s)", p, exc)
        return LoadedDoc()


__all__ = ["LoadedDoc", "load_pages", "IMAGE_EXTS"]
