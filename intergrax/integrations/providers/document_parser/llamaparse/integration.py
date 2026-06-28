# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Llamaparse document parser integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.ai import DocumentParserIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

LLAMAPARSE_DOCUMENT_PARSER_PROVIDER_ID = "llamaparse"


class LlamaparseDocumentParserIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Llamaparse document parser integration."""

    pass


@runtime_checkable
class LlamaparseDocumentParserClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class LlamaparseDocumentParserIntegration(DocumentParserIntegrationContract):
    """
    Llamaparse document parser integration.

    The legacy facade (create_llamaparse_document_parser) remains separate and backward-compatible.
    """

    config: LlamaparseDocumentParserIntegrationConfig = LlamaparseDocumentParserIntegrationConfig()
    _client: LlamaparseDocumentParserClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: LlamaparseDocumentParserClient,
        *,
        enabled: bool = False,
    ) -> LlamaparseDocumentParserIntegration:
        integration = cls.for_provider(
            provider_id=LLAMAPARSE_DOCUMENT_PARSER_PROVIDER_ID,
            display_name="Llamaparse",
            config=LlamaparseDocumentParserIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> LlamaparseDocumentParserClient | None:
        return self._client
