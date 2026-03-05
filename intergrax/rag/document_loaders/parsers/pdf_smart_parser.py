# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Sequence

from langchain_core.documents import Document
from intergrax.rag.document_loaders.contracts.base_document_parser import BaseDocumentParser


class PdfSmartParser(BaseDocumentParser):

    def __init__(
        self,
        enable_ocr: bool,
        ocr_lang: str,
        ocr_dpi: int,
        ocr_psm: int | None,
        ocr_oem: int | None,
        ocr_max_pages: int | None,
    ) -> None:
        from langchain_community.document_loaders import PyMuPDFLoader
        import fitz
        import pytesseract
        from PIL import Image

        self._enable_ocr = enable_ocr
        self._ocr_lang = ocr_lang
        self._ocr_dpi = ocr_dpi
        self._ocr_psm = ocr_psm
        self._ocr_oem = ocr_oem
        self._ocr_max_pages = ocr_max_pages

        self._fitz = fitz
        self._pytesseract = pytesseract
        self._Image = Image
        self._loader_cls = PyMuPDFLoader

    @classmethod
    def parser_id(cls) -> str:
        return "pymupdf"

    def is_available(self) -> bool:
        return True

    def load(self, source: str) -> Sequence[Document]:

        loader = self._loader_cls(source)
        docs = loader.load()

        if not docs or not self._enable_ocr:
            return docs

        pdf = None

        try:
            pdf = self._fitz.open(source)
        except Exception:
            pdf = None

        ocr_done = 0

        for d in docs:

            content = (d.page_content or "").strip()
            if content:
                continue

            if self._ocr_max_pages is not None and ocr_done >= self._ocr_max_pages:
                break

            pidx = d.metadata.get("page") or d.metadata.get("page_index")
            if pidx is None or pdf is None:
                continue

            try:
                page = pdf.load_page(int(pidx))
            except Exception:
                continue

            pix = page.get_pixmap(dpi=self._ocr_dpi)

            img = self._Image.frombytes(
                "RGB",
                [pix.width, pix.height],
                pix.samples
            )

            cfg_parts = []

            if self._ocr_psm is not None:
                cfg_parts.append(f"--psm {int(self._ocr_psm)}")

            if self._ocr_oem is not None:
                cfg_parts.append(f"--oem {int(self._ocr_oem)}")

            config = " ".join(cfg_parts) if cfg_parts else None

            try:
                ocr_text = self._pytesseract.image_to_string(
                    img,
                    lang=self._ocr_lang,
                    config=config
                ) or ""
            except Exception:
                ocr_text = ""

            ocr_text = ocr_text.strip()

            if not ocr_text:
                continue

            d.page_content = ocr_text

            md = dict(d.metadata or {})
            md["ocr"] = True
            md["ocr_lang"] = self._ocr_lang
            md["ocr_dpi"] = self._ocr_dpi
            d.metadata = md

            ocr_done += 1

        if pdf is not None:
            try:
                pdf.close()
            except Exception:
                pass

        return docs