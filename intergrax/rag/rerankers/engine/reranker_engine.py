# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import List, Optional

from intergrax.rag.rerankers.cache.rerank_cache import RerankCache
from intergrax.rag.rerankers.contracts.base_reranker import BaseReranker
from intergrax.rag.rerankers.contracts.reranker_types import (
    Candidates,
    RerankerResult,
)
from intergrax.rag.rerankers.registry.reranker_registry import RerankerRegistry


class RerankerEngine:
    """
    Orchestrates execution of reranker strategies.
    """

    def __init__(
        self,
        registry: RerankerRegistry,
        cache: Optional[RerankCache] = None,
    ) -> None:

        self._registry = registry
        self._cache = cache

    def rerank(
        self,
        *,
        reranker_name: Optional[str],
        query: Optional[str],
        candidates: Candidates,
        max_candidates: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[RerankerResult]:

        if not candidates:
            return []

        if reranker_name is None:
            reranker_name = self._registry.default_reranker()

        reranker: BaseReranker = self._registry.get(reranker_name)

        if max_candidates is not None:
            if len(candidates) > max_candidates:
                candidates = candidates[: max_candidates]

        # Cache lookup
        if self._cache is not None and query is not None:

            texts = [c.text for c in candidates]

            cached_scores = self._cache.get(
                reranker=reranker_name,
                query=query,
                texts=texts,
            )

            if cached_scores is not None:

                results: List[RerankerResult] = []

                for rank, (candidate, score) in enumerate(
                    zip(candidates, cached_scores),
                    start=1,
                ):
                    results.append(
                        RerankerResult(
                            candidate=candidate,
                            rerank_score=score,
                            fusion_score=None,
                            rank=rank,
                        )
                    )

                if limit is not None:
                    return results[:limit]

                return results


        results = reranker.rerank(
            query=query,
            candidates=candidates,
            limit=limit,
        )

        # Cache store
        if self._cache is not None and query is not None:

            texts = [c.text for c in candidates]
            scores = [r.rerank_score for r in results]

            self._cache.set(
                reranker=reranker_name,
                query=query,
                texts=texts,
                scores=scores,
            )

        return results