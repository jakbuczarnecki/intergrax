# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Lancedb vector store integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.health_probe import IntegrationHealthProbe
from intergrax.integrations.contracts.vector_store import MetadataFilter, VectorStore, VectorStoreHit
from intergrax.runtime.integrations.categories.storage import VectorStoreIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

LANCEDB_VECTOR_STORE_PROVIDER_ID = "lancedb"


class LancedbVectorStoreIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Lancedb vector store integration."""

    pass


@runtime_checkable
class LancedbVectorStoreClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class LancedbVectorStoreIntegration(VectorStoreIntegrationContract):
    """
    Single public Lancedb vector store entrypoint.

    Legacy catalog factory (create_lancedb_vector_store) delegates to this class.
    """

    config: LancedbVectorStoreIntegrationConfig = LancedbVectorStoreIntegrationConfig()
    _client: LancedbVectorStoreClient | None = PrivateAttr(default=None)
    _store_config: Any | None = PrivateAttr(default=None)
    _inner: Any | None = PrivateAttr(default=None)

    @classmethod
    def from_store(
        cls,
        store_config: Any,
        inner: Any,
        *,
        enabled: bool = True,
    ) -> LancedbVectorStoreIntegration:
        integration = cls.for_provider(
            provider_id=LANCEDB_VECTOR_STORE_PROVIDER_ID,
            display_name="Lancedb",
            config=LancedbVectorStoreIntegrationConfig(enabled=enabled),
        )
        integration._store_config = store_config
        integration._inner = inner
        return integration

    @classmethod
    def from_runtime(
        cls,
        runtime: Any,
        *,
        enabled: bool = True,
        store_config: Any | None = None,
    ) -> LancedbVectorStoreIntegration:
        return cls.from_store(runtime, enabled=enabled, store_config=store_config)

    @property
    def store_config(self) -> Any | None:
        return self._store_config

    @property
    def rag_store(self) -> VectorStore:
        return self._require_runtime()
    def add_documents(
        self,
        documents: Sequence[Any],
        embeddings: Sequence[Sequence[float]],
        *,
        ids: Sequence[str] | None = None,
    ) -> None:
        self._require_runtime().add_documents(documents, embeddings, ids=ids)

    def query(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int,
        metadata_filter: MetadataFilter | None = None,
        include_embeddings: bool = False,
    ) -> list[VectorStoreHit]:
        return self._require_runtime().query(
            query_embedding,
            top_k=top_k,
            metadata_filter=metadata_filter,
            include_embeddings=include_embeddings,
        )

    def delete(self, ids: Sequence[str]) -> None:
        self._require_runtime().delete(ids)

    def count(self) -> int:
        return self._require_runtime().count()


    def _require_runtime(self) -> Any:
        private = object.__getattribute__(self, "__pydantic_private__")
        runtime = private.get("_runtime")
        if runtime is None:
            runtime = private.get("_backend")
        if runtime is None:
            runtime = private.get("_inner")
        if runtime is None:
            raise IntegrationConfigurationError(
                f"{type(self).__name__} requires a runtime delegate for catalog operations",
            )
        return runtime


    @classmethod
    def from_client(
        cls,
        client: LancedbVectorStoreClient,
        *,
        enabled: bool = False,
    ) -> LancedbVectorStoreIntegration:
        integration = cls.for_provider(
            provider_id=LANCEDB_VECTOR_STORE_PROVIDER_ID,
            display_name="Lancedb",
            config=LancedbVectorStoreIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> LancedbVectorStoreClient | None:
        return self._client

VectorStore.register(LancedbVectorStoreIntegration)
