# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Optional, Sequence

from langchain_core.documents import Document

from intergrax.rag.vectorstore.contracts.vector_store import VectorStore
from intergrax.rag.vectorstore.contracts.vector_store import MetadataFilter
from intergrax.rag.vectorstore.contracts.vector_store import VectorStoreHit
from intergrax.logging import IntergraxLogging

logger = IntergraxLogging.get_logger(__name__, component="rag")


class VectorstoreManager:
    """
    Thin delegation layer for VectorStore providers.

    This class no longer implements any backend logic.
    It delegates all operations to the injected VectorStore instance.
    """

    def __init__(self, store: VectorStore) -> None:
        self._store = store

    def add_documents(
        self,
        documents: Sequence[Document],
        embeddings: Sequence[Sequence[float]],
        *,
        ids: Optional[Sequence[str]] = None,
    ) -> None:
        self._store.add_documents(
            documents=documents,
            embeddings=embeddings,
            ids=ids,
        )

    def query(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int,
        metadata_filter: Optional[MetadataFilter] = None,
        include_embeddings: bool = False,
    ) -> Sequence[VectorStoreHit]:
        return self._store.query(
            query_embedding=query_embedding,
            top_k=top_k,
            metadata_filter=metadata_filter,
            include_embeddings=include_embeddings,
        )

    def delete(self, ids: Sequence[str]) -> None:
        self._store.delete(ids)

    def count(self) -> int:
        return self._store.count()