# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Sequence
from langchain_core.documents import Document

from intergrax.rag.document_loaders.config.document_loader_config import DEFAULT_BUILTIN_HANDLER_CONFIDENCE
from intergrax.rag.document_loaders.contracts.base_document_handler import BaseDocumentHandler
from intergrax.rag.document_loaders.documents_loader import PdfSmartLoader


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