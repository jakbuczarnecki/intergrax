# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import List, Optional, Sequence

from intergrax.rag.rerankers.contracts.base_reranker import BaseReranker
from intergrax.rag.rerankers.contracts.reranker_types import (
    Candidates,
    RerankerCandidate,
    RerankerResult,
)


class EnsembleReranker(BaseReranker):

    def __init__(
        self,
        rerankers: Sequence[BaseReranker],
        *,
        weights: Optional[Sequence[float]] = None,
    ) -> None:

        self._rerankers = list(rerankers)

        if not self._rerankers:
            raise ValueError("EnsembleReranker requires at least one reranker.")

        if weights is None:
            weights = [1.0] * len(self._rerankers)

        if len(weights) != len(self._rerankers):
            raise ValueError("weights must match number of rerankers.")

        self._weights = list(weights)

    @classmethod
    def name(cls) -> str:
        return "ensemble"

    def rerank(
        self,
        *,
        query: Optional[str],
        candidates: Candidates,
        limit: Optional[int] = None,
    ) -> List[RerankerResult]:

        if not candidates:
            return []

        aggregated: dict[str, float] = {}
        candidate_map: dict[str, RerankerCandidate] = {}

        for reranker, weight in zip(self._rerankers, self._weights):

            results = reranker.rerank(
                query=query,
                candidates=candidates,
                limit=None,
            )

            for r in results:

                cid = r.candidate.id or r.candidate.text

                candidate_map[cid] = r.candidate

                score = r.rerank_score

                aggregated[cid] = aggregated.get(cid, 0.0) + weight * score

        results: List[RerankerResult] = []

        for cid, score in aggregated.items():

            candidate = candidate_map[cid]

            results.append(
                RerankerResult(
                    candidate=candidate,
                    rerank_score=score,
                    fusion_score=score,
                    rank=0,
                )
            )

        results.sort(
            key=lambda r: r.rerank_score,
            reverse=True,
        )

        if limit:
            results = results[:limit]

        for i, r in enumerate(results, start=1):
            r.rank = i

        return results