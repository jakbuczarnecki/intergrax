# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Literal, Sequence

from intergrax.integrations.contracts.document_parser import ParsedDocumentFragment
from intergrax.rag.document_loaders.contracts.base_document_parser import BaseDocumentParser
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

    def load(self, source: str) -> Sequence[ParsedDocumentFragment]:
        backend = resolve_document_parser("openpyxl", **self._options)
        return CatalogDocumentParser(backend).load(source)
