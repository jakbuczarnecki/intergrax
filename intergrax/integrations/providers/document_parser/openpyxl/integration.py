# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Openpyxl document parser integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.ai import DocumentParserIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

OPENPYXL_DOCUMENT_PARSER_PROVIDER_ID = "openpyxl"


class OpenpyxlDocumentParserIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Openpyxl document parser integration."""

    pass


@runtime_checkable
class OpenpyxlDocumentParserClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class OpenpyxlDocumentParserIntegration(DocumentParserIntegrationContract):
    """
    Openpyxl document parser integration.

    The legacy facade (create_openpyxl_document_parser) remains separate and backward-compatible.
    """

    config: OpenpyxlDocumentParserIntegrationConfig = OpenpyxlDocumentParserIntegrationConfig()
    _client: OpenpyxlDocumentParserClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: OpenpyxlDocumentParserClient,
        *,
        enabled: bool = False,
    ) -> OpenpyxlDocumentParserIntegration:
        integration = cls.for_provider(
            provider_id=OPENPYXL_DOCUMENT_PARSER_PROVIDER_ID,
            display_name="Openpyxl",
            config=OpenpyxlDocumentParserIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> OpenpyxlDocumentParserClient | None:
        return self._client
