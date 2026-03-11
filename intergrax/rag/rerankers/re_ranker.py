# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import List, Optional


from intergrax.rag.embedding.bootstrap.default_embedding_engine import create_default_embedding_manager
from intergrax.rag.rerankers.bootstrap.reranker_bootstrap import create_default_reranker_engine
from intergrax.rag.rerankers.engine.reranker_engine import RerankerEngine
from intergrax.rag.rerankers.contracts.reranker_types import (
    RerankerCandidate,
    RerankerResult,
)


class ReRanker:
    """
    Public facade for the Intergrax reranking subsystem.

    This class provides a stable entry point for reranking
    operations while delegating execution to RerankerEngine.

    Responsibilities:
        - expose a simple rerank API
        - delegate execution to RerankerEngine

    Non-responsibilities:
        - scoring algorithms
        - provider selection
        - embedding computation
        - caching
        - pipeline orchestration
    """

    def __init__(
        self,
        engine: Optional[RerankerEngine] = None,
    ) -> None:

        self._engine = engine or create_default_reranker_engine(embedding_manager=create_default_embedding_manager())

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def rerank(
        self,
        *,
        query: Optional[str],
        candidates: List[RerankerCandidate],
        limit: Optional[int] = None,
    ) -> List[RerankerResult]:

        return self._engine.rerank(
            query=query,
            candidates=candidates,
            limit=limit,
        )

    # ------------------------------------------------------------------
    # Callable interface
    # ------------------------------------------------------------------

    def __call__(
        self,
        *,
        query: Optional[str],
        candidates: List[RerankerCandidate],
        limit: Optional[int] = None,
    ) -> List[RerankerResult]:

        return self.rerank(
            query=query,
            candidates=candidates,
            limit=limit,
        )