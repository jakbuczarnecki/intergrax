# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typesense vector store integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.storage import VectorStoreIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

TYPESENSE_VECTOR_STORE_PROVIDER_ID = "typesense"


class TypesenseVectorStoreIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Typesense vector store integration."""

    pass


@runtime_checkable
class TypesenseVectorStoreClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class TypesenseVectorStoreIntegration(VectorStoreIntegrationContract):
    """
    Typesense vector store integration.

    The legacy facade (create_typesense_vector_store) remains separate and backward-compatible.
    """

    config: TypesenseVectorStoreIntegrationConfig = TypesenseVectorStoreIntegrationConfig()
    _client: TypesenseVectorStoreClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: TypesenseVectorStoreClient,
        *,
        enabled: bool = False,
    ) -> TypesenseVectorStoreIntegration:
        integration = cls.for_provider(
            provider_id=TYPESENSE_VECTOR_STORE_PROVIDER_ID,
            display_name="Typesense",
            config=TypesenseVectorStoreIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> TypesenseVectorStoreClient | None:
        return self._client
