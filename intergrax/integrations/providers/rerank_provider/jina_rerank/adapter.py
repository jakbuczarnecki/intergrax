# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from collections.abc import Sequence
from typing import List

from intergrax.integrations.providers.rerank_provider.jina_rerank.config import JinaRerankIntegrationConfig
from intergrax.integrations.providers.rerank_provider.jina_rerank.opens import jina_rerank_scores
from intergrax.rag.rerankers.contracts.reranker_types import (
    RerankerCandidate,
    RerankerResult,
    validate_candidates,
    validate_limit,
)


class _JinaRerankProvider:
    def __init__(self, config: JinaRerankIntegrationConfig) -> None:
        self._config = config

    def name(self) -> str:
        return "jina"

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
        scores = jina_rerank_scores(self._config, query, texts)
        if len(scores) != len(candidates):
            raise ValueError("Jina returned an invalid score shape")
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

    def score(self, query: str, texts: List[str]) -> List[float]:
        if not texts:
            return []
        return jina_rerank_scores(self._config, query, texts)
