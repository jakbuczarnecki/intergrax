# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.rag.vectorstore.contracts.native_vectorstore import (
    MetadataFilter,
    VectorStoreHit,
    VectorStoreRecord,
    VectorStoreScope,
)
from intergrax.rag.vectorstore.providers.base_vector_store import BaseVectorStore


@dataclass(frozen=True)
class MilvusConfig:
    collection_name: str
    tenant_id: str


class MilvusVectorStore(BaseVectorStore):
    """Milvus-backed vector store — refuses CRUD until native SDK mapping is wired."""

    def __init__(self, cfg: MilvusConfig, *, client: Any = None) -> None:
        self.cfg = cfg
        self.collection_name = f"{cfg.collection_name}__tenant__{cfg.tenant_id}"
        self._client = client

    def _unsupported(self) -> IntegrationConfigurationError:
        return IntegrationConfigurationError("Milvus vector store CRUD is not configured for native persistence")

    def add_records(
        self,
        records: Sequence[VectorStoreRecord],
        *,
        scope: VectorStoreScope,
    ) -> Sequence[str]:
        del records, scope
        raise self._unsupported()

    def query(
        self,
        query_embedding: Sequence[float],
        *,
        scope: VectorStoreScope,
        top_k: int,
        metadata_filter: MetadataFilter | None = None,
        include_embeddings: bool = False,
    ) -> list[VectorStoreHit]:
        del query_embedding, scope, top_k, metadata_filter, include_embeddings
        raise self._unsupported()

    def delete(self, ids: Sequence[str], *, scope: VectorStoreScope) -> None:
        del ids, scope
        raise self._unsupported()

    def count(self, *, scope: VectorStoreScope) -> int:
        del scope
        raise self._unsupported()

    def list_collections(self) -> list[str]:
        if self._client is None:
            return [self.collection_name]
        try:
            names = self._client.list_collections()
            return [str(name) for name in names]
        except Exception:
            return [self.collection_name]
