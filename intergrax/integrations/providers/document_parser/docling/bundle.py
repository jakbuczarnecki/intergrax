# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Optional

from intergrax.integrations.contracts.document_parser import DocumentParser
from intergrax.integrations.providers.document_parser.docling.adapter import _DoclingDocumentParser
from intergrax.integrations.providers.document_parser.docling.config import DoclingIntegrationConfig


def create_docling_document_parser(**config_overrides: object) -> DoclingDocumentParserIntegration:
    config = DoclingIntegrationConfig.from_env(**config_overrides)
    return DoclingDocumentParserIntegration.from_client(_DoclingDocumentParser(config))

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.document_parser.docling.integration import (
    DOCLING_DOCUMENT_PARSER_PROVIDER_ID,
    DoclingDocumentParserIntegration,
    DoclingDocumentParserIntegrationConfig,
    DoclingDocumentParserClient,
)


def create_docling_document_parser_integration(
    *,
    client: DoclingDocumentParserClient | None = None,
    enabled: bool = False,
) -> DoclingDocumentParserIntegration:
    """
    Build a contract-based Docling document parser integration.

    Compatibility shim — constructs Integration via from_store (create_docling_document_parser) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Docling document parser integration requires an injected client when enabled=True",
        )
    if client is not None:
        return DoclingDocumentParserIntegration.from_client(client, enabled=enabled)
    return DoclingDocumentParserIntegration.for_provider(
        provider_id=DOCLING_DOCUMENT_PARSER_PROVIDER_ID,
        display_name="Docling",
        config=DoclingDocumentParserIntegrationConfig(enabled=enabled),
    )
