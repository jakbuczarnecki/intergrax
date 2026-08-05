# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from intergrax.rag.vectorstore.contracts.native_vectorstore import (
    MetadataFilter,
    VectorStoreHit,
    VectorStoreScope,
)


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
