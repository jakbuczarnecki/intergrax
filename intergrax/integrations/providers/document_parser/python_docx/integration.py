# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Python Docx document parser integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.ai import DocumentParserIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

PYTHON_DOCX_DOCUMENT_PARSER_PROVIDER_ID = "python_docx"


class PythonDocxDocumentParserIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Python Docx document parser integration."""

    pass


@runtime_checkable
class PythonDocxDocumentParserClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class PythonDocxDocumentParserIntegration(DocumentParserIntegrationContract):
    """
    Python Docx document parser integration.

    The legacy facade (create_python_docx_document_parser) remains separate and backward-compatible.
    """

    config: PythonDocxDocumentParserIntegrationConfig = PythonDocxDocumentParserIntegrationConfig()
    _client: PythonDocxDocumentParserClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: PythonDocxDocumentParserClient,
        *,
        enabled: bool = False,
    ) -> PythonDocxDocumentParserIntegration:
        integration = cls.for_provider(
            provider_id=PYTHON_DOCX_DOCUMENT_PARSER_PROVIDER_ID,
            display_name="Python Docx",
            config=PythonDocxDocumentParserIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> PythonDocxDocumentParserClient | None:
        return self._client
