# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import List

from intergrax.rag.document_loaders.config.document_loader_config import GLOBAL_DOCUMENT_LOADER_CONFIG
from intergrax.rag.document_loaders.contracts.base_document_parser import BaseDocumentParser
from intergrax.rag.document_loaders.parsers.pdf_smart_parser import PdfSmartParser
from intergrax.rag.document_loaders.contracts.base_document_handler import BaseDocumentHandler


class PdfSmartDocumentHandler(BaseDocumentHandler):

    def __init__(
        self,
        enable_ocr: bool = False,
        ocr_lang: str = "eng",
        ocr_dpi: int = 200,
        ocr_psm: int | None = None,
        ocr_oem: int | None = None,
        ocr_max_pages: int | None = None,   
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
        return GLOBAL_DOCUMENT_LOADER_CONFIG.default_builtin_handler_confidence

    def build_parsers(self) -> List[BaseDocumentParser]:
        return [
            PdfSmartParser(
                enable_ocr=self._enable_ocr,
                ocr_lang=self._ocr_lang,
                ocr_dpi=self._ocr_dpi,
                ocr_psm=self._ocr_psm,
                ocr_oem=self._ocr_oem,
                ocr_max_pages=self._ocr_max_pages,
            )
        ]
