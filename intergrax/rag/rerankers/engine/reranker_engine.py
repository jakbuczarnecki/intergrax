# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import List, Optional

from intergrax.rag.rerankers.contracts.base_reranker import BaseReranker
from intergrax.rag.rerankers.contracts.reranker_types import (
    Candidates,
    RerankerResult,
)
from intergrax.rag.rerankers.registry.reranker_registry import RerankerRegistry


class RerankerEngine:
    """
    Orchestrates execution of reranker strategies.
    """

    def __init__(
        self,
        registry: RerankerRegistry,
    ) -> None:

        self._registry = registry

    def rerank(
        self,
        *,
        reranker_name: Optional[str],
        query: Optional[str],
        candidates: Candidates,
        limit: Optional[int] = None,
    ) -> List[RerankerResult]:

        if reranker_name is None:
            reranker_name = self._registry.default_reranker()

        reranker: BaseReranker = self._registry.get(reranker_name)

        return reranker.rerank(
            query=query,
            candidates=candidates,
            limit=limit,
        )