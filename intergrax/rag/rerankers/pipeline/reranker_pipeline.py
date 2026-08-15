# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from collections.abc import Sequence

from intergrax.rag.rerankers.contracts.reranker_types import (
    RerankerCandidate,
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
        query: str,
        candidates: Sequence[RerankerCandidate],
        reranker_name: str | None = None,
        limit: int | None = None,
    ) -> Sequence[RerankerResult]:

        return self._engine.rerank(
            reranker_name=reranker_name,
            query=query,
            candidates=candidates,
            limit=limit,
        )