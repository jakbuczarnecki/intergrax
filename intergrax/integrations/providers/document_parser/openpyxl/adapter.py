# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Sequence

from intergrax.integrations.contracts.document_parser import DocumentParser, ParsedDocumentFragment
from intergrax.integrations.providers.document_parser.openpyxl.config import OpenpyxlIntegrationConfig
from intergrax.integrations.providers.document_parser.openpyxl.opens import parse_openpyxl_file


class OpenpyxlDocumentParser:
    def __init__(self, config: OpenpyxlIntegrationConfig) -> None:
        self._config = config

    def parser_id(self) -> str:
        return "openpyxl"

    def is_available(self) -> bool:
        try:
            import pandas  # noqa: F401

            return True
        except Exception:
            return False

    def parse_file(self, source: str) -> Sequence[ParsedDocumentFragment]:
        return parse_openpyxl_file(self._config, source)
