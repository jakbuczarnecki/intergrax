# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Shared catalog wrapper delegating ``VectorStore`` calls to a RAG backend."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from intergrax.integrations._shared.config import BaseIntegrationConfig
from intergrax.integrations.contracts.base import HealthStatus
from intergrax.integrations.contracts.health_probe import IntegrationHealthProbe
from intergrax.integrations.contracts.vector_store import (
    MetadataFilter,
    VectorStore,
    VectorStoreHit,
    VectorStoreRecord,
    VectorStoreScope,
)


class VectorStoreBridge(VectorStore):
    """
    Thin ``VectorStore`` wrapper over an inner store (typically ``intergrax/rag/``).

    Construct only from provider ``opens.py`` factories.
    """

    def __init__(
        self,
        config: BaseIntegrationConfig,
        inner: VectorStore,
    ) -> None:
        self._config = config
        self._inner = inner

    @property
    def config(self) -> BaseIntegrationConfig:
        return self._config

    @property
    def rag_store(self) -> VectorStore:
        """Underlying RAG vector store instance."""
        return self._inner

    def add_records(
        self,
        records: Sequence[VectorStoreRecord],
        *,
        scope: VectorStoreScope,
    ) -> Sequence[str] | None:
        return self._inner.add_records(records, scope=scope)

    def query(
        self,
        query_embedding: NDArray[np.float32] | Sequence[float],
        *,
        scope: VectorStoreScope,
        top_k: int,
        metadata_filter: MetadataFilter | None = None,
        include_embeddings: bool = False,
    ) -> Sequence[VectorStoreHit]:
        return self._inner.query(
            query_embedding,
            scope=scope,
            top_k=top_k,
            metadata_filter=metadata_filter,
            include_embeddings=include_embeddings,
        )

    def delete(self, ids: Sequence[str], *, scope: VectorStoreScope) -> None:
        self._inner.delete(ids, scope=scope)

    def count(self, *, scope: VectorStoreScope) -> int:
        return self._inner.count(scope=scope)

    def list_source_record_ids(
        self,
        *,
        source_id: str,
        scope: VectorStoreScope,
        root_document_id: str | None = None,
    ) -> Sequence[str]:
        return self._inner.list_source_record_ids(
            source_id=source_id,
            scope=scope,
            root_document_id=root_document_id,
        )

    def health(self) -> HealthStatus | bool:
        if isinstance(self._inner, IntegrationHealthProbe):
            return self._inner.health()
        return True
