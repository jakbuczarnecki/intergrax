# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import List, Sequence

from intergrax.rag.rerankers.contracts.base_reranker import BaseReranker
from intergrax.rag.rerankers.contracts.reranker_types import (
    RerankerCandidate,
    RerankerResult,
    validate_candidates,
    validate_limit,
)


class RRFReranker(BaseReranker):

    def __init__(
        self,
        rerankers: Sequence[BaseReranker],
        *,
        k: int = 60,
    ) -> None:

        if not rerankers:
            raise ValueError("RRFReranker requires at least one reranker.")

        self._rerankers = list(rerankers)
        self._k = int(k)

    @classmethod
    def name(cls) -> str:
        return "rrf"

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
        scores: dict[int, float] = {}
        candidate_map: dict[int, RerankerCandidate] = {}

        for reranker in self._rerankers:

            results = reranker.rerank(
                query=query,
                candidates=candidates,
                limit=None,
            )

            for r in results:

                cid = id(r.candidate)
                if cid not in positions:
                    raise ValueError("RRF reranker returned an unknown candidate")

                candidate_map[cid] = r.candidate

                rank = r.rank if r.rank > 0 else 1

                scores[cid] = scores.get(cid, 0.0) + 1.0 / (self._k + rank)

        if set(scores) != set(positions):
            raise ValueError("RRF reranker returned an incomplete candidate batch")

        fused: List[RerankerResult] = []

        for cid, score in scores.items():

            fused.append(
                RerankerResult(
                    candidate=candidate_map[cid],
                    rerank_score=score,
                    fusion_score=score,
                    rank=0,
                )
            )

        fused.sort(key=lambda r: (-r.rerank_score, positions[id(r.candidate)]))
        selected = fused[:limit] if limit is not None else fused
        return tuple(
            RerankerResult(
                candidate=result.candidate,
                rerank_score=result.rerank_score,
                fusion_score=result.fusion_score,
                rank=rank,
            )
            for rank, result in enumerate(selected)
        )