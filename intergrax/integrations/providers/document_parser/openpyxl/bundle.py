# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.integrations.contracts.document_parser import DocumentParser
from intergrax.integrations.providers.document_parser.openpyxl.adapter import _OpenpyxlDocumentParser
from intergrax.integrations.providers.document_parser.openpyxl.config import OpenpyxlIntegrationConfig


def create_openpyxl_document_parser(**config_overrides: object) -> OpenpyxlDocumentParserIntegration:
    return OpenpyxlDocumentParserIntegration.from_client(_OpenpyxlDocumentParser(OpenpyxlIntegrationConfig.from_env(**config_overrides)))

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.document_parser.openpyxl.integration import (
    OPENPYXL_DOCUMENT_PARSER_PROVIDER_ID,
    OpenpyxlDocumentParserIntegration,
    OpenpyxlDocumentParserIntegrationConfig,
    OpenpyxlDocumentParserClient,
)


def create_openpyxl_document_parser_integration(
    *,
    client: OpenpyxlDocumentParserIntegrationClient | None = None,
    enabled: bool = False,
) -> OpenpyxlDocumentParserIntegration:
    """
    Build a contract-based Openpyxl document parser integration.

    Compatibility shim — constructs Integration via from_store (create_openpyxl_document_parser) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Openpyxl document parser integration requires an injected client when enabled=True",
        )
    if client is not None:
        return OpenpyxlDocumentParserIntegration.from_client(client, enabled=enabled)
    return OpenpyxlDocumentParserIntegration.for_provider(
        provider_id=OPENPYXL_DOCUMENT_PARSER_PROVIDER_ID,
        display_name="Openpyxl",
        config=OpenpyxlDocumentParserIntegrationConfig(enabled=enabled),
    )
