# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from collections.abc import Sequence
from typing import List, Optional

from intergrax.integrations.providers.rerank_provider.cohere_rerank.config import CohereRerankIntegrationConfig
from intergrax.integrations.providers.rerank_provider.cohere_rerank.opens import cohere_rerank_scores
from intergrax.rag.rerankers.contracts.reranker_types import (
    RerankerCandidate,
    RerankerResult,
    validate_candidates,
    validate_limit,
)


class _CohereRerankProvider:
    def __init__(self, config: CohereRerankIntegrationConfig) -> None:
        self._config = config

    def name(self) -> str:
        return "cohere"

    def rerank(
        self,
        query: str,
        candidates: Sequence[RerankerCandidate],
        *,
        top_n: int | None = None,
    ) -> Sequence[RerankerResult]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        candidates = validate_candidates(candidates)
        validate_limit(top_n)
        if not candidates:
            return ()
        texts = [candidate.document.content for candidate in candidates]
        scores = cohere_rerank_scores(self._config, query, texts, top_n=top_n)
        if len(scores) != len(candidates):
            raise ValueError("Cohere returned an invalid score shape")
        ordered = sorted(
            zip(candidates, scores),
            key=lambda pair: pair[1],
            reverse=True,
        )
        if top_n is not None:
            ordered = ordered[:top_n]
        return tuple(
            RerankerResult(
                candidate=candidate,
                rerank_score=score,
                fusion_score=None,
                rank=rank,
            )
            for rank, (candidate, score) in enumerate(ordered)
        )

    def score(self, query: str, texts: List[str], *, top_n: Optional[int] = None) -> List[float]:
        if not texts:
            return []
        return cohere_rerank_scores(self._config, query, texts, top_n=top_n)
