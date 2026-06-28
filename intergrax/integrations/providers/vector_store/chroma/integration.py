# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Chroma vector store integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.storage import VectorStoreIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

CHROMA_VECTOR_STORE_PROVIDER_ID = "chroma"


class ChromaVectorStoreIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Chroma vector store integration."""

    pass


@runtime_checkable
class ChromaVectorStoreClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class ChromaVectorStoreIntegration(VectorStoreIntegrationContract):
    """
    Chroma vector store integration.

    The legacy facade (create_chroma_integration) remains separate and backward-compatible.
    """

    config: ChromaVectorStoreIntegrationConfig = ChromaVectorStoreIntegrationConfig()
    _client: ChromaVectorStoreClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: ChromaVectorStoreClient,
        *,
        enabled: bool = False,
    ) -> ChromaVectorStoreIntegration:
        integration = cls.for_provider(
            provider_id=CHROMA_VECTOR_STORE_PROVIDER_ID,
            display_name="Chroma",
            config=ChromaVectorStoreIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> ChromaVectorStoreClient | None:
        return self._client
