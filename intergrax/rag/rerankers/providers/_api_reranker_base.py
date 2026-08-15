# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import List

from intergrax.rag.rerankers.contracts.base_reranker import BaseReranker
from intergrax.rag.rerankers.contracts.reranker_types import (
    validate_candidates,
    validate_limit,
    RerankerCandidate,
    RerankerResult,
)


class _APIRerankerBase(BaseReranker, ABC):

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

        if not query.strip():
            return ()

        normalized = candidates
        texts: List[str] = [c.document.content for c in normalized]

        scores = self._score(query, texts)
        if len(scores) != len(normalized):
            raise ValueError("reranker returned an invalid score shape")

        scored = list(zip(normalized, scores))
        scored.sort(key=lambda item: float(item[1]), reverse=True)
        selected = scored[:limit] if limit is not None else scored
        return tuple(
            RerankerResult(
                candidate=candidate,
                rerank_score=score,
                fusion_score=None,
                rank=rank,
            )
            for rank, (candidate, score) in enumerate(selected)
        )

    @abstractmethod
    def _score(
        self,
        query: str,
        texts: List[str],
    ) -> List[float]:
        ...