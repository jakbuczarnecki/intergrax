# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Pymupdf document parser integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.ai import DocumentParserIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

PYMUPDF_DOCUMENT_PARSER_PROVIDER_ID = "pymupdf"


class PymupdfDocumentParserIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Pymupdf document parser integration."""

    pass


@runtime_checkable
class PymupdfDocumentParserClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class PymupdfDocumentParserIntegration(DocumentParserIntegrationContract):
    """
    Pymupdf document parser integration.

    The legacy facade (create_pymupdf_document_parser) remains separate and backward-compatible.
    """

    config: PymupdfDocumentParserIntegrationConfig = PymupdfDocumentParserIntegrationConfig()
    _client: PymupdfDocumentParserClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: PymupdfDocumentParserClient,
        *,
        enabled: bool = False,
    ) -> PymupdfDocumentParserIntegration:
        integration = cls.for_provider(
            provider_id=PYMUPDF_DOCUMENT_PARSER_PROVIDER_ID,
            display_name="Pymupdf",
            config=PymupdfDocumentParserIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> PymupdfDocumentParserClient | None:
        return self._client
