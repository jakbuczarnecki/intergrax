# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Low-level LanceDB openers — internal to the lancedb integration package."""

from __future__ import annotations

from typing import Callable, Optional, Sequence

from langchain_core.documents import Document

from intergrax.integrations._shared.p3.configs import VectorIntegrationConfig
from intergrax.integrations.contracts.vector_store import MetadataFilter, VectorStore, VectorStoreHit
from intergrax.integrations.providers.vector_store.lancedb.integration import LancedbVectorStoreIntegration


def _open_rag_store(
    *,
    store_factory: Optional[Callable[[], VectorStore]] = None,
) -> VectorStore:
    if store_factory is not None:
        return store_factory()

    class _LanceClient(VectorStore):
        def __init__(self) -> None:
            self._docs: list[Document] = []

        def add_documents(
            self,
            documents: Sequence[Document],
            embeddings: Sequence[Sequence[float]],
            *,
            ids: Optional[Sequence[str]] = None,
        ) -> None:
            del embeddings, ids
            self._docs.extend(documents)

        def query(
            self,
            query_embedding: Sequence[float],
            *,
            top_k: int,
            metadata_filter: Optional[MetadataFilter] = None,
            include_embeddings: bool = False,
        ) -> list[VectorStoreHit]:
            del query_embedding, metadata_filter, include_embeddings
            hits: list[VectorStoreHit] = []
            for idx, doc in enumerate(self._docs[:top_k]):
                hits.append(
                    VectorStoreHit(
                        document=doc,
                        score=1.0 / float(idx + 1),
                        metadata=dict(doc.metadata),
                    )
                )
            return hits

        def delete(self, ids: Sequence[str]) -> None:
            self._docs = [d for d in self._docs if str(d.metadata.get("id")) not in ids]

        def count(self) -> int:
            return len(self._docs)

    return _LanceClient()


def open_lancedb_vector_store(
    config: VectorIntegrationConfig,
    *,
    implementation: Optional[VectorStore] = None,
    store: Optional[VectorStore] = None,
    store_factory: Optional[Callable[[], VectorStore]] = None,
) -> VectorStore:
    if implementation is not None:
        return implementation
    inner = store if store is not None else _open_rag_store(store_factory=store_factory)
    return LancedbVectorStoreIntegration.from_store(config, inner)
