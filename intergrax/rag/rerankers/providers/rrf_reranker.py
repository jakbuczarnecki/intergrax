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
        query: Optional[str],
        candidates: Candidates,
        limit: Optional[int] = None,
    ) -> List[RerankerResult]:

        if not candidates:
            return []

        scores: dict[str, float] = {}
        candidate_map: dict[str, RerankerCandidate] = {}

        for reranker in self._rerankers:

            results = reranker.rerank(
                query=query,
                candidates=candidates,
                limit=None,
            )

            for r in results:

                cid = r.candidate.id or r.candidate.text

                candidate_map[cid] = r.candidate

                rank = r.rank if r.rank > 0 else 1

                scores[cid] = scores.get(cid, 0.0) + 1.0 / (self._k + rank)

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

        fused.sort(
            key=lambda r: r.rerank_score,
            reverse=True,
        )

        if limit:
            fused = fused[:limit]

        for i, r in enumerate(fused, start=1):
            r.rank = i

        return fused