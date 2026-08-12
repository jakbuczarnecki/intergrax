# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from intergrax.rag.vectorstore.contracts.native_vectorstore import (
    MetadataFilter,
    VectorStoreHit,
    VectorStoreScope,
)


@runtime_checkable
class HybridSearchCapable(Protocol):
    """Vector stores that support dense + lexical (BM25) hybrid query."""

    def query_hybrid(
        self,
        query_embedding: Sequence[float],
        query_text: str,
        *,
        scope: VectorStoreScope,
        top_k: int,
        metadata_filter: MetadataFilter | None = None,
        include_embeddings: bool = False,
        alpha: float = 0.5,
    ) -> Sequence[VectorStoreHit]: ...


@runtime_checkable
class NativeHybridSearchCapability(Protocol):
    """Explicit durable native hybrid capability — not inferred from query_hybrid."""

    def supports_native_hybrid_search(self) -> bool: ...


def provider_supports_native_hybrid_search(store: object) -> bool:
    """
    Return True when the store exposes durable native hybrid search explicitly.

    Process-local lexical caches must opt out via ``supports_native_hybrid_search()``.
    """
    if isinstance(store, NativeHybridSearchCapability):
        return store.supports_native_hybrid_search()
    return False
