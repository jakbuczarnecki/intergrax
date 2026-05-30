# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Pinecone vector store adapter — catalog facade delegating to ``rag/`` (no SDK here)."""

from __future__ import annotations

from typing import Optional, Sequence

from langchain_core.documents import Document

from intergrax.integrations.contracts.vector_store import MetadataFilter, VectorStore, VectorStoreHit
from intergrax.integrations.providers.pinecone.config import PineconeIntegrationConfig


class PineconeVectorStoreIntegration(VectorStore):
    """
    Thin ``VectorStore`` wrapper over the RAG ``PineconeVectorStore``.

    The underlying store is constructed only in ``opens.open_pinecone_vector_store()``.
    Tier-3 code MUST use ``create_pinecone_vector_store()`` or ``profile.resolve()``.
    """

    def __init__(
        self,
        config: PineconeIntegrationConfig,
        inner: VectorStore,
    ) -> None:
        self._config = config
        self._inner = inner

    @property
    def config(self) -> PineconeIntegrationConfig:
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
