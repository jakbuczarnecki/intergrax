# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Literal, Sequence

from langchain_core.documents import Document

from intergrax.rag.document_loaders.contracts.base_document_parser import BaseDocumentParser
from intergrax.rag.document_loaders.integration.catalog_parser import CatalogDocumentParser
from intergrax.rag.document_loaders.integration.resolver import resolve_document_parser

EXTRACTION_STRATEGY = Literal["auto", "fulltext", "paragraphs", "headings"]


class DocSmartParser(BaseDocumentParser):
    def __init__(self, strategy: EXTRACTION_STRATEGY):
        self._strategy = strategy

    @classmethod
    def parser_id(cls) -> str:
        return "python_docx"

    def is_available(self) -> bool:
        return resolve_document_parser(
            "python_docx",
            strategy=self._strategy,
        ).is_available()

    def load(self, source: str) -> Sequence[Document]:
        backend = resolve_document_parser(
            "python_docx",
            strategy=self._strategy,
        )
        return CatalogDocumentParser(backend).load(source)
