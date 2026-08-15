# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import List, Optional, Sequence

from intergrax.rag.rerankers.contracts.base_reranker import BaseReranker
from intergrax.rag.rerankers.contracts.reranker_types import (
    RerankerCandidate,
    RerankerResult,
    validate_candidates,
    validate_limit,
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
        query: str,
        candidates: Sequence[RerankerCandidate],
        limit: int | None = None,
    ) -> Sequence[RerankerResult]:
        candidates = validate_candidates(candidates)
        validate_limit(limit)
        if not candidates:
            return ()

        positions = {id(candidate): position for position, candidate in enumerate(candidates)}
        aggregated: dict[int, float] = {}
        candidate_map: dict[int, RerankerCandidate] = {}

        for reranker, weight in zip(self._rerankers, self._weights):

            results = reranker.rerank(
                query=query,
                candidates=candidates,
                limit=None,
            )

            for r in results:

                cid = id(r.candidate)
                if cid not in positions:
                    raise ValueError("ensemble reranker returned an unknown candidate")

                candidate_map[cid] = r.candidate

                score = r.rerank_score

                aggregated[cid] = aggregated.get(cid, 0.0) + weight * score

        results: List[RerankerResult] = []

        if set(aggregated) != set(positions):
            raise ValueError("ensemble reranker returned an incomplete candidate batch")

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

        results.sort(key=lambda r: (-r.rerank_score, positions[id(r.candidate)]))
        selected = results[:limit] if limit is not None else results
        return tuple(
            RerankerResult(
                candidate=result.candidate,
                rerank_score=result.rerank_score,
                fusion_score=result.fusion_score,
                rank=rank,
            )
            for rank, result in enumerate(selected)
        )