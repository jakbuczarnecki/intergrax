# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Whisper document parser integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.ai import DocumentParserIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

WHISPER_DOCUMENT_PARSER_PROVIDER_ID = "whisper"


class WhisperDocumentParserIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Whisper document parser integration."""

    pass


@runtime_checkable
class WhisperDocumentParserClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class WhisperDocumentParserIntegration(DocumentParserIntegrationContract):
    """
    Whisper document parser integration.

    The legacy facade (create_whisper_document_parser) remains separate and backward-compatible.
    """

    config: WhisperDocumentParserIntegrationConfig = WhisperDocumentParserIntegrationConfig()
    _client: WhisperDocumentParserClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: WhisperDocumentParserClient,
        *,
        enabled: bool = False,
    ) -> WhisperDocumentParserIntegration:
        integration = cls.for_provider(
            provider_id=WHISPER_DOCUMENT_PARSER_PROVIDER_ID,
            display_name="Whisper",
            config=WhisperDocumentParserIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> WhisperDocumentParserClient | None:
        return self._client
