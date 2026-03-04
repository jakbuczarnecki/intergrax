# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Sequence
from langchain_core.documents import Document

try:
    import fitz  # PyMuPDF (used only for raster OCR)
except Exception:
    fitz = None

try:
    import pytesseract
except Exception:
    pytesseract = None

from PIL import Image

from intergrax.rag.document_loaders.config.document_loader_config import DEFAULT_BUILTIN_HANDLER_CONFIDENCE
from intergrax.rag.document_loaders.contracts.base_document_handler import BaseDocumentHandler


class PdfSmartDocumentHandler(BaseDocumentHandler):

    def __init__(
        self,
        enable_ocr: bool,
        ocr_lang: str,
        ocr_dpi: int,
        ocr_psm: int,
        ocr_oem: int,
        ocr_max_pages: int,
    ) -> None:
        self._enable_ocr = enable_ocr
        self._ocr_lang = ocr_lang
        self._ocr_dpi = ocr_dpi
        self._ocr_psm = ocr_psm
        self._ocr_oem = ocr_oem
        self._ocr_max_pages = ocr_max_pages

    def supports(self, source: str) -> bool:
        return source.lower().endswith(".pdf")

    def confidence(self, source: str) -> float:
        return DEFAULT_BUILTIN_HANDLER_CONFIDENCE

    def load(self, source: str) -> Sequence[Document]:
        loader = PdfSmartLoader(
            source,
            enable_ocr=self._enable_ocr,
            ocr_lang=self._ocr_lang,
            ocr_dpi=self._ocr_dpi,
            ocr_psm=self._ocr_psm,
            ocr_oem=self._ocr_oem,
            ocr_max_pages=self._ocr_max_pages,
        )
        return loader.load()
    

class PdfSmartLoader:
    """
    PDF → list of Document(s), with OCR fallback ONLY for pages that are empty after text extraction.
    - Main text source: PyMuPDFLoader (langchain community)
    - Fallback: rasterization to bitmap and pytesseract.image_to_string(...)
    """
    def __init__(
        self,
        path: str,
        *,
        enable_ocr: bool = False,
        ocr_lang: str = "eng",
        ocr_dpi: int = 200,
        ocr_psm: int | None = None,
        ocr_oem: int | None = None,
        ocr_max_pages: int | None = None,   # hard-cap on number of pages for OCR (None = no limit)
    ):
        from langchain_community.document_loaders import PyMuPDFLoader
        self.path = path
        self.enable_ocr = bool(enable_ocr)
        self.ocr_lang = ocr_lang
        self.ocr_dpi = int(ocr_dpi)
        self.ocr_psm = ocr_psm
        self.ocr_oem = ocr_oem
        self.ocr_max_pages = ocr_max_pages
        self._base = PyMuPDFLoader(path)

    def _ocr_page(self, page) -> str:
        """Rasterizes a page and performs OCR. Returns text (may be empty)."""
        if not (fitz and pytesseract and Image):
            return ""
        pix = page.get_pixmap(dpi=self.ocr_dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        cfg_parts = []
        if self.ocr_psm is not None:
            cfg_parts.append(f"--psm {int(self.ocr_psm)}")
        if self.ocr_oem is not None:
            cfg_parts.append(f"--oem {int(self.ocr_oem)}")
        config = " ".join(cfg_parts) if cfg_parts else None
        try:
            return pytesseract.image_to_string(img, lang=self.ocr_lang, config=config) or ""
        except Exception:
            return ""

    def load(self) -> list[Document]:
        # 1) first do the “normal” parsing
        docs = self._base.load()  # usually each Document = one page (metadata['page'])
        if not docs or not self.enable_ocr:
            return docs

        # 2) identify empty pages and do OCR only there
        # open the PDF with PyMuPDF once (if available)
        pdf = None
        if fitz:
            try:
                pdf = fitz.open(self.path)
            except Exception:
                pdf = None

        ocr_done = 0
        for d in docs:
            content = (d.page_content or "").strip()
            if content:
                continue
            if self.ocr_max_pages is not None and ocr_done >= self.ocr_max_pages:
                break
            # find page index (PyMuPDFLoader adds 'page' → copied below as 'page_index' in intergrax loader)
            pidx = d.metadata.get("page") or d.metadata.get("page_index")
            if pidx is None or pdf is None:
                continue
            try:
                page = pdf.load_page(int(pidx))
            except Exception:
                continue
            ocr_text = (self._ocr_page(page) or "").strip()
            if ocr_text:
                d.page_content = ocr_text
                # mark that it came from OCR (useful for debugging/metrics)
                md = dict(d.metadata or {})
                md["ocr"] = True
                md["ocr_lang"] = self.ocr_lang
                md["ocr_dpi"] = self.ocr_dpi
                d.metadata = md
                ocr_done += 1

        if pdf is not None:
            try:
                pdf.close()
            except Exception:
                pass
        return docs
