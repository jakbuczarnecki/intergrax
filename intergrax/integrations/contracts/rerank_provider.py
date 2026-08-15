# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Rerank provider integration contract (Phase M.7)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from intergrax.rag.rerankers.contracts.reranker_types import (
    RerankerCandidate,
    RerankerResult,
)


@runtime_checkable
class RerankProvider(Protocol):
    """Vendor reranking API (Cohere, Jina, …)."""

    def name(self) -> str:
        """Registry name, e.g. ``cohere``."""

    def rerank(
        self,
        query: str,
        candidates: Sequence[RerankerCandidate],
        *,
        top_n: int | None = None,
    ) -> Sequence[RerankerResult]:
        """Return native results ordered by relevance to ``query``."""
