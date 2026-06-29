# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.integrations.contracts.document_parser import DocumentParser
from intergrax.integrations.providers.document_parser.whisper.adapter import _WhisperDocumentParser
from intergrax.integrations.providers.document_parser.whisper.config import WhisperIntegrationConfig


def create_whisper_document_parser(**config_overrides: object) -> WhisperDocumentParserIntegration:
    return WhisperDocumentParserIntegration.from_client(_WhisperDocumentParser(WhisperIntegrationConfig.from_env(**config_overrides)))

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.document_parser.whisper.integration import (
    WHISPER_DOCUMENT_PARSER_PROVIDER_ID,
    WhisperDocumentParserIntegration,
    WhisperDocumentParserIntegrationConfig,
    WhisperDocumentParserClient,
)


def create_whisper_document_parser_integration(
    *,
    client: WhisperDocumentParserIntegrationClient | None = None,
    enabled: bool = False,
) -> WhisperDocumentParserIntegration:
    """
    Build a contract-based Whisper document parser integration.

    Compatibility shim — constructs Integration via from_store (create_whisper_document_parser) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Whisper document parser integration requires an injected client when enabled=True",
        )
    if client is not None:
        return WhisperDocumentParserIntegration.from_client(client, enabled=enabled)
    return WhisperDocumentParserIntegration.for_provider(
        provider_id=WHISPER_DOCUMENT_PARSER_PROVIDER_ID,
        display_name="Whisper",
        config=WhisperDocumentParserIntegrationConfig(enabled=enabled),
    )
