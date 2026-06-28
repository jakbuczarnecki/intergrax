# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Docling document parser integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.ai import DocumentParserIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

DOCLING_DOCUMENT_PARSER_PROVIDER_ID = "docling"


class DoclingDocumentParserIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Docling document parser integration."""

    pass


@runtime_checkable
class DoclingDocumentParserClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class DoclingDocumentParserIntegration(DocumentParserIntegrationContract):
    """
    Docling document parser integration.

    The legacy facade (create_docling_document_parser) remains separate and backward-compatible.
    """

    config: DoclingDocumentParserIntegrationConfig = DoclingDocumentParserIntegrationConfig()
    _client: DoclingDocumentParserClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: DoclingDocumentParserClient,
        *,
        enabled: bool = False,
    ) -> DoclingDocumentParserIntegration:
        integration = cls.for_provider(
            provider_id=DOCLING_DOCUMENT_PARSER_PROVIDER_ID,
            display_name="Docling",
            config=DoclingDocumentParserIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> DoclingDocumentParserClient | None:
        return self._client
