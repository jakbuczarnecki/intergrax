# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Sequence

from intergrax.integrations.contracts.document_parser import DocumentParser, ParsedDocumentFragment
from intergrax.integrations.providers.document_parser.python_docx.config import PythonDocxIntegrationConfig
from intergrax.integrations.providers.document_parser.python_docx.opens import parse_python_docx_file


class PythonDocxDocumentParser:
    def __init__(self, config: PythonDocxIntegrationConfig) -> None:
        self._config = config

    def parser_id(self) -> str:
        return "python_docx"

    def is_available(self) -> bool:
        return True

    def parse_file(self, source: str) -> Sequence[ParsedDocumentFragment]:
        return parse_python_docx_file(self._config, source)


def create_python_docx_document_parser(**config_overrides: object) -> DocumentParser:
    return PythonDocxDocumentParser(PythonDocxIntegrationConfig.from_env(**config_overrides))
