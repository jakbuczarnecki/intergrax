# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from intergrax.rag.vectorstore.contracts.native_vectorstore import VectorStoreScope


@dataclass(frozen=True)
class RetrievalRequest:
    """Canonical retrieval input for tools, Nexus, and AnswerPipeline adapters."""

    query: str
    top_k: Optional[int] = None
    """Deprecated alias for ``final_top_k`` — kept for backward compatibility."""
    final_top_k: Optional[int] = None
    prefetch_k: Optional[int] = None
    metadata_filter: Any = None
    scope: VectorStoreScope | None = None
    score_threshold: Optional[float] = None
    retriever_id: Optional[str] = None
    route_tier_override: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def resolved_final_k(self, profile_final: int) -> int:
        if self.final_top_k is not None:
            return int(self.final_top_k)
        if self.top_k is not None:
            return int(self.top_k)
        return int(profile_final)

    def resolved_prefetch_k(self, profile_prefetch: int, final_k: int) -> int:
        if self.prefetch_k is not None:
            prefetch = int(self.prefetch_k)
        else:
            prefetch = int(profile_prefetch)
        if prefetch < final_k:
            return max(final_k, int(profile_prefetch))
        return prefetch
