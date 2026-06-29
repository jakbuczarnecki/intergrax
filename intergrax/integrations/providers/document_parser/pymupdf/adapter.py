# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Sequence

from intergrax.integrations.contracts.document_parser import DocumentParser, ParsedDocumentFragment
from intergrax.integrations.providers.document_parser.pymupdf.config import PymupdfIntegrationConfig
from intergrax.integrations.providers.document_parser.pymupdf.opens import parse_pymupdf_file


class _PymupdfDocumentParser:
    def __init__(self, config: PymupdfIntegrationConfig) -> None:
        self._config = config

    def parser_id(self) -> str:
        return "pymupdf"

    def is_available(self) -> bool:
        return True

    def parse_file(self, source: str) -> Sequence[ParsedDocumentFragment]:
        return parse_pymupdf_file(self._config, source)
