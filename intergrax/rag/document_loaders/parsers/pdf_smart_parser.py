# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Sequence

from langchain_core.documents import Document

from intergrax.rag.document_loaders.contracts.base_document_parser import BaseDocumentParser
from intergrax.rag.document_loaders.integration.catalog_parser import CatalogDocumentParser
from intergrax.rag.document_loaders.integration.resolver import resolve_document_parser


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
        self._options = {
            "enable_ocr": enable_ocr,
            "ocr_lang": ocr_lang,
            "ocr_dpi": ocr_dpi,
            "ocr_psm": ocr_psm,
            "ocr_oem": ocr_oem,
            "ocr_max_pages": ocr_max_pages,
        }

    @classmethod
    def parser_id(cls) -> str:
        return "pymupdf"

    def is_available(self) -> bool:
        return resolve_document_parser("pymupdf", **self._options).is_available()

    def load(self, source: str) -> Sequence[Document]:
        backend = resolve_document_parser("pymupdf", **self._options)
        return CatalogDocumentParser(backend).load(source)
