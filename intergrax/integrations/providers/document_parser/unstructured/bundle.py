# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Literal, Sequence

from intergrax.integrations.contracts.document_parser import DocumentParser, ParsedDocumentFragment
from intergrax.integrations.providers.document_parser.unstructured.opens import parse_unstructured_html


class UnstructuredDocumentParser:
    def parser_id(self) -> str:
        return "unstructured"

    def is_available(self) -> bool:
        return True

    def parse_file(self, source: str) -> Sequence[ParsedDocumentFragment]:
        return parse_unstructured_html(source)


def create_unstructured_document_parser(**_: object) -> UnstructuredDocumentParserIntegration:
    return UnstructuredDocumentParserIntegration.from_client(UnstructuredDocumentParser())

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.document_parser.unstructured.integration import (
    UNSTRUCTURED_DOCUMENT_PARSER_PROVIDER_ID,
    UnstructuredDocumentParserIntegration,
    UnstructuredDocumentParserIntegrationConfig,
    UnstructuredDocumentParserClient,
)


def create_unstructured_document_parser_integration(
    *,
    client: UnstructuredDocumentParserClient | None = None,
    enabled: bool = False,
) -> UnstructuredDocumentParserIntegration:
    """
    Build a contract-based Unstructured document parser integration.

    The legacy facade (create_unstructured_document_parser) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Unstructured document parser integration requires an injected client when enabled=True",
        )
    if client is not None:
        return UnstructuredDocumentParserIntegration.from_client(client, enabled=enabled)
    return UnstructuredDocumentParserIntegration.for_provider(
        provider_id=UNSTRUCTURED_DOCUMENT_PARSER_PROVIDER_ID,
        display_name="Unstructured",
        config=UnstructuredDocumentParserIntegrationConfig(enabled=enabled),
    )
