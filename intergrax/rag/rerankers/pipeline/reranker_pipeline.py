# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import List, Optional

from intergrax.rag.rerankers.contracts.reranker_types import (
    Candidates,
    RerankerResult,
)

from intergrax.rag.rerankers.engine.reranker_engine import RerankerEngine


class RerankerPipeline:
    """
    Pipeline orchestrating reranking process.
    """

    def __init__(
        self,
        engine: RerankerEngine,
    ) -> None:

        self._engine = engine

    def rerank(
        self,
        *,
        query: Optional[str],
        candidates: Candidates,
        reranker_name: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[RerankerResult]:

        return self._engine.rerank(
            reranker_name=reranker_name,
            query=query,
            candidates=candidates,
            limit=limit,
        )