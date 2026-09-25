# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from intergrax.integrations.contracts.rerank_provider import RerankProvider
from intergrax.rag.rerankers.contracts.base_reranker import BaseReranker
from intergrax.rag.rerankers.contracts.reranker_types import (
    validate_candidates,
    validate_limit,
    RerankerCandidate,
    RerankerResult,
)


def _validate_rerank_provider_results(results: object) -> Sequence[RerankerResult]:
    if isinstance(results, (str, bytes)):
        raise TypeError("rerank provider returned an invalid result type")
    if not isinstance(results, Sequence):
        raise TypeError("rerank provider returned an invalid result type")
    validated: list[RerankerResult] = []
    for item in results:
        if not isinstance(item, RerankerResult):
            raise TypeError("rerank provider returned a non-RerankerResult entry")
        validated.append(item)
    return tuple(validated)


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

        provider = self._resolve_provider()
        results = provider.rerank(query, candidates, top_n=limit)
        return _validate_rerank_provider_results(results)

    @abstractmethod
    def _resolve_provider(self) -> RerankProvider:
        ...
