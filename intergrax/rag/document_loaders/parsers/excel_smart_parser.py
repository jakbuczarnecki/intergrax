# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Literal, Sequence

from langchain_core.documents import Document

from intergrax.rag.document_loaders.contracts.base_document_parser import BaseDocumentParser
from intergrax.rag.document_loaders.contracts.metadata_contract import build_loader_metadata
from intergrax.rag.document_loaders.integration.catalog_parser import CatalogDocumentParser
from intergrax.rag.document_loaders.integration.resolver import resolve_document_parser

EXTRACTION_STRATEGY = Literal["rows", "sheets", "markdown"]


class ExcelSmartParser(BaseDocumentParser):

    def __init__(
        self,
        *,
        mode: EXTRACTION_STRATEGY,
        header: int,
        sheet: str | int | None,
        na_filter: bool,
        max_rows_per_sheet: int | None,
        encoding: str | None,
        delimiter: str | None,
    ):
        self._options = {
            "mode": mode,
            "header": header,
            "sheet": sheet,
            "na_filter": na_filter,
            "max_rows_per_sheet": max_rows_per_sheet,
            "encoding": encoding,
            "delimiter": delimiter,
        }

    @classmethod
    def parser_id(cls) -> str:
        return "openpyxl"

    def is_available(self) -> bool:
        return resolve_document_parser("openpyxl", **self._options).is_available()

    def load(self, source: str) -> Sequence[Document]:
        backend = resolve_document_parser("openpyxl", **self._options)
        docs = CatalogDocumentParser(backend).load(source)
        result: list[Document] = []
        for i, doc in enumerate(docs):
            metadata = build_loader_metadata(
                source=source,
                parser=self.parser_id(),
                position=i,
            )
            metadata.update(doc.metadata or {})
            result.append(Document(page_content=doc.page_content, metadata=metadata))
        return result
