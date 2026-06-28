# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.integrations.contracts.document_parser import DocumentParser
from intergrax.integrations.providers.document_parser.pymupdf.adapter import PymupdfDocumentParser
from intergrax.integrations.providers.document_parser.pymupdf.config import PymupdfIntegrationConfig


def create_pymupdf_document_parser(**config_overrides: object) -> DocumentParser:
    return PymupdfDocumentParser(PymupdfIntegrationConfig.from_env(**config_overrides))

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.document_parser.pymupdf.integration import (
    PYMUPDF_DOCUMENT_PARSER_PROVIDER_ID,
    PymupdfDocumentParserIntegration,
    PymupdfDocumentParserIntegrationConfig,
    PymupdfDocumentParserClient,
)


def create_pymupdf_document_parser_integration(
    *,
    client: PymupdfDocumentParserClient | None = None,
    enabled: bool = False,
) -> PymupdfDocumentParserIntegration:
    """
    Build a contract-based Pymupdf document parser integration.

    The legacy facade (create_pymupdf_document_parser) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Pymupdf document parser integration requires an injected client when enabled=True",
        )
    if client is not None:
        return PymupdfDocumentParserIntegration.from_client(client, enabled=enabled)
    return PymupdfDocumentParserIntegration.for_provider(
        provider_id=PYMUPDF_DOCUMENT_PARSER_PROVIDER_ID,
        display_name="Pymupdf",
        config=PymupdfDocumentParserIntegrationConfig(enabled=enabled),
    )
