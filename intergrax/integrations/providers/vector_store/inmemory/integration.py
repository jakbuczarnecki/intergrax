# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Inmemory vector store integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.storage import VectorStoreIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

INMEMORY_VECTOR_STORE_PROVIDER_ID = "inmemory"


class InmemoryVectorStoreIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Inmemory vector store integration."""

    pass


@runtime_checkable
class InmemoryVectorStoreClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class InmemoryVectorStoreIntegration(VectorStoreIntegrationContract):
    """
    Inmemory vector store integration.

    The legacy facade (create_inmemory_vector_store) remains separate and backward-compatible.
    """

    config: InmemoryVectorStoreIntegrationConfig = InmemoryVectorStoreIntegrationConfig()
    _client: InmemoryVectorStoreClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: InmemoryVectorStoreClient,
        *,
        enabled: bool = False,
    ) -> InmemoryVectorStoreIntegration:
        integration = cls.for_provider(
            provider_id=INMEMORY_VECTOR_STORE_PROVIDER_ID,
            display_name="Inmemory",
            config=InmemoryVectorStoreIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> InmemoryVectorStoreClient | None:
        return self._client
