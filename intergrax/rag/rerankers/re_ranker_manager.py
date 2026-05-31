# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import List, Optional

from intergrax.rag.rerankers.contracts.base_reranker_manager import BaseRerankerManager
from intergrax.rag.rerankers.engine.reranker_engine import RerankerEngine
from intergrax.rag.rerankers.contracts.reranker_types import (
    RerankerCandidate,
    RerankerResult,
)


class ReRankerManager(BaseRerankerManager):
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
        engine: RerankerEngine = None,
    ) -> None:

        self._engine = engine

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def rerank(
        self,
        *,
        query: Optional[str],
        candidates: List[RerankerCandidate],
        limit: Optional[int] = None,
        reranker_id: Optional[str] = None,
    ) -> List[RerankerResult]:

        return self._engine.rerank(
            reranker_name=reranker_id,
            query=query,
            candidates=candidates,
            limit=limit,
        )