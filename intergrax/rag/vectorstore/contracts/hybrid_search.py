# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from typing import List, Optional, Protocol, Sequence

from intergrax.rag.vectorstore.contracts.vector_store import MetadataFilter, VectorStoreHit


class HybridSearchCapable(Protocol):
    """Vector stores that support dense + lexical (BM25) hybrid query."""

    def query_hybrid(
        self,
        query_embedding: Sequence[float],
        query_text: str,
        *,
        top_k: int,
        metadata_filter: Optional[MetadataFilter] = None,
        include_embeddings: bool = False,
        alpha: float = 0.5,
    ) -> List[VectorStoreHit]: ...
