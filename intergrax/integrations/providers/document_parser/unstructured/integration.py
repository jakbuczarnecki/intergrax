# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unstructured document parser integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.document_parser import DocumentParser
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
    Single public Unstructured document parser entrypoint.

    Legacy catalog factory (create_unstructured_document_parser) delegates to this class.
    """

    config: UnstructuredDocumentParserIntegrationConfig = UnstructuredDocumentParserIntegrationConfig()
    _client: UnstructuredDocumentParserClient | None = PrivateAttr(default=None)
    _runtime: Any | None = PrivateAttr(default=None)

    @classmethod
    def from_runtime(cls, runtime: Any, *, enabled: bool = True) -> UnstructuredDocumentParserIntegration:
        integration = cls.for_provider(
            provider_id=UNSTRUCTURED_DOCUMENT_PARSER_PROVIDER_ID,
            display_name="Unstructured",
            config=UnstructuredDocumentParserIntegrationConfig(enabled=enabled),
        )
        integration._runtime = runtime
        return integration

    def _require_runtime(self) -> Any:
        if self._runtime is None:
            raise IntegrationConfigurationError("Unstructured integration requires a runtime delegate")
        return self._runtime



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
    def __getattr__(self, name: str) -> object:
        if name.startswith("_"):
            private = object.__getattribute__(self, "__pydantic_private__")
            if name in private:
                return private[name]
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        return getattr(self._require_runtime(), name)

DocumentParser.register(UnstructuredDocumentParserIntegration)
