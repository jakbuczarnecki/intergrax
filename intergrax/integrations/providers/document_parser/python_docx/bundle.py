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

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.document_parser.python_docx.integration import (
    PYTHON_DOCX_DOCUMENT_PARSER_PROVIDER_ID,
    PythonDocxDocumentParserIntegration,
    PythonDocxDocumentParserIntegrationConfig,
    PythonDocxDocumentParserClient,
)


def create_python_docx_document_parser_integration(
    *,
    client: PythonDocxDocumentParserClient | None = None,
    enabled: bool = False,
) -> PythonDocxDocumentParserIntegration:
    """
    Build a contract-based Python Docx document parser integration.

    The legacy facade (create_python_docx_document_parser) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Python Docx document parser integration requires an injected client when enabled=True",
        )
    if client is not None:
        return PythonDocxDocumentParserIntegration.from_client(client, enabled=enabled)
    return PythonDocxDocumentParserIntegration.for_provider(
        provider_id=PYTHON_DOCX_DOCUMENT_PARSER_PROVIDER_ID,
        display_name="Python Docx",
        config=PythonDocxDocumentParserIntegrationConfig(enabled=enabled),
    )
