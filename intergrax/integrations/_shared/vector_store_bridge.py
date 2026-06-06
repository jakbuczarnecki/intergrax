# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Shared catalog wrapper delegating ``VectorStore`` calls to a RAG backend."""

from __future__ import annotations

from typing import Optional, Sequence

from langchain_core.documents import Document

from intergrax.integrations._shared.config import BaseIntegrationConfig
from intergrax.integrations.contracts.base import HealthStatus, IntegrationConfigurationError
from intergrax.integrations.contracts.health_probe import IntegrationHealthProbe
from intergrax.integrations.contracts.vector_store import MetadataFilter, VectorStore, VectorStoreHit


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

    def add_documents(
        self,
        documents: Sequence[Document],
        embeddings: Sequence[Sequence[float]],
        *,
        ids: Optional[Sequence[str]] = None,
    ) -> None:
        self._inner.add_documents(documents, embeddings, ids=ids)

    def query(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int,
        metadata_filter: Optional[MetadataFilter] = None,
        include_embeddings: bool = False,
    ) -> list[VectorStoreHit]:
        return self._inner.query(
            query_embedding,
            top_k=top_k,
            metadata_filter=metadata_filter,
            include_embeddings=include_embeddings,
        )

    def delete(self, ids: Sequence[str]) -> None:
        self._inner.delete(ids)

    def count(self) -> int:
        return self._inner.count()

    def health(self) -> HealthStatus | bool:
        if isinstance(self._inner, IntegrationHealthProbe):
            return self._inner.health()
        return True
