# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import List, Literal

from intergrax.rag.document_loaders.config.document_loader_config import GLOBAL_DOCUMENT_LOADER_CONFIG, DoclingMode
from intergrax.rag.document_loaders.contracts.base_document_parser import BaseDocumentParser
from intergrax.rag.document_loaders.parsers.docling_local_parser import DoclingLocalParser
from intergrax.rag.document_loaders.parsers.docling_server_parser import DoclingServerParser
from intergrax.rag.document_loaders.parsers.excel_smart_parser import EXTRACTION_STRATEGY, ExcelSmartParser

from intergrax.rag.document_loaders.contracts.base_document_handler import BaseDocumentHandler

class ExcelSmartDocumentHandler(BaseDocumentHandler):

    def __init__(
        self,
        *,
        mode: EXTRACTION_STRATEGY = "rows",
        header: int = 0,
        sheet: str | int | None = None,
        na_filter: bool = True,
        max_rows_per_sheet: int | None = None,
        encoding: str | None = None,
        delimiter: str | None = None,
    ) -> None:

        self._mode = mode
        self._header = header
        self._sheet = sheet
        self._na_filter = na_filter
        self._max_rows_per_sheet = max_rows_per_sheet
        self._encoding = encoding
        self._delimiter = delimiter

    def supports(self, source: str) -> bool:

        s = source.lower()

        return (
            s.endswith(".xlsx")
            or s.endswith(".xls")
            or s.endswith(".csv")
            or s.endswith(".tsv")
        )

    def confidence(self, source: str) -> float:
        return GLOBAL_DOCUMENT_LOADER_CONFIG.default_builtin_handler_confidence

    def build_parsers(self) -> List[BaseDocumentParser]:

        parsers: List[BaseDocumentParser] = []

        parsers.append(
            ExcelSmartParser(
                mode=self._mode,
                header=self._header,
                sheet=self._sheet,
                na_filter=self._na_filter,
                max_rows_per_sheet=self._max_rows_per_sheet,
                encoding=self._encoding,
                delimiter=self._delimiter,
            )
        )

        mode = GLOBAL_DOCUMENT_LOADER_CONFIG.docling_mode

        if mode is DoclingMode.LOCAL:
            parsers.append(DoclingLocalParser())

        elif mode is DoclingMode.SERVER:
            parsers.append(DoclingServerParser())

        return parsers