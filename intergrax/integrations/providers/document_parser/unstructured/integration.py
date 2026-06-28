# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unstructured document parser integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.ai import DocumentParserIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

UNSTRUCTURED_DOCUMENT_PARSER_PROVIDER_ID = "unstructured"


class UnstructuredDocumentParserIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Unstructured document parser integration."""

    pass


@runtime_checkable
class UnstructuredDocumentParserClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class UnstructuredDocumentParserIntegration(DocumentParserIntegrationContract):
    """
    Unstructured document parser integration.

    The legacy facade (create_unstructured_document_parser) remains separate and backward-compatible.
    """

    config: UnstructuredDocumentParserIntegrationConfig = UnstructuredDocumentParserIntegrationConfig()
    _client: UnstructuredDocumentParserClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: UnstructuredDocumentParserClient,
        *,
        enabled: bool = False,
    ) -> UnstructuredDocumentParserIntegration:
        integration = cls.for_provider(
            provider_id=UNSTRUCTURED_DOCUMENT_PARSER_PROVIDER_ID,
            display_name="Unstructured",
            config=UnstructuredDocumentParserIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> UnstructuredDocumentParserClient | None:
        return self._client
