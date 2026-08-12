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


def _unwrap_integration_wrapper(store: object) -> object:
    current = store
    for _ in range(8):
        inner = getattr(current, "_inner", None)
        if inner is None:
            break
        current = inner
    return current


def provider_supports_native_hybrid_search(store: object) -> bool:
    """
  Return True when the resolved provider exposes durable native hybrid search.

  Process-local lexical caches (for example LexicalHybridSupport without sparse
  vectors) must opt out via ``supports_native_hybrid_search()``.
    """
    bridge = getattr(store, "supports_native_hybrid_search", None)
    if callable(bridge):
        return bool(bridge())

    candidate = _unwrap_integration_wrapper(store)
    if not isinstance(candidate, HybridSearchCapable):
        return False

    explicit = getattr(candidate, "supports_native_hybrid_search", None)
    if callable(explicit):
        return bool(explicit())
    return True
