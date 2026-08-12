# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from collections.abc import Sequence

from intergrax.rag.rerankers.cache.base_rerank_cache import BaseRerankCache
from intergrax.rag.rerankers.contracts.base_reranker import BaseReranker
from intergrax.rag.rerankers.contracts.reranker_types import (
    RerankerCandidate,
    RerankerResult,
    validate_candidates,
    validate_limit,
)
from intergrax.rag.rerankers.registry.reranker_registry import RerankerRegistry


class RerankerEngine:
    """
    Orchestrates execution of reranker strategies.
    """

    def __init__(
        self,
        registry: RerankerRegistry,
        cache: BaseRerankCache | None = None,
    ) -> None:

        self._registry = registry
        self._cache = cache

    def rerank(
        self,
        *,
        reranker_name: str | None = None,
        query: str = "",
        candidates: Sequence[RerankerCandidate] = (),
        max_candidates: int | None = None,
        limit: int | None = None,
    ) -> Sequence[RerankerResult]:
        candidates = validate_candidates(candidates)
        validate_limit(limit)
        if max_candidates is not None and (
            type(max_candidates) is not int or max_candidates <= 0
        ):
            raise ValueError("max_candidates must be an exact positive int or None")
        if not candidates:
            return ()

        if reranker_name is None:
            reranker_name = self._registry.default_reranker()

        reranker: BaseReranker = self._registry.get(reranker_name)

        if max_candidates is not None:
            if len(candidates) > max_candidates:
                candidates = candidates[:max_candidates]

        # Cache lookup
        if self._cache is not None and query is not None:

            texts = [c.document.content for c in candidates]
            identity_keys = [c.identity_key for c in candidates]

            cached_scores = self._cache.get(
                reranker=reranker_name,
                query=query,
                texts=texts,
                identity_keys=identity_keys,
            )

            if cached_scores is not None:

                if len(cached_scores) != len(candidates):
                    raise ValueError("reranker cache returned an invalid score shape")
                results = [
                    RerankerResult(
                        candidate=candidate,
                        rerank_score=score,
                        fusion_score=None,
                        rank=0,
                    )
                    for candidate, score in zip(candidates, cached_scores)
                ]
                results.sort(key=lambda result: result.rerank_score, reverse=True)
                results = [
                    RerankerResult(
                        candidate=result.candidate,
                        rerank_score=result.rerank_score,
                        fusion_score=result.fusion_score,
                        rank=rank,
                    )
                    for rank, result in enumerate(
                        results[:limit] if limit is not None else results
                    )
                ]
                return tuple(results)


        results = reranker.rerank(
            query=query,
            candidates=candidates,
            limit=limit,
        )

        results = tuple(results)
        input_ids = {id(candidate) for candidate in candidates}
        result_ids = [id(result.candidate) for result in results]
        if len(result_ids) != len(set(result_ids)) or not set(result_ids) <= input_ids:
            raise ValueError("reranker returned an unknown or duplicate candidate")
        if limit is None and set(result_ids) != input_ids:
            raise ValueError("reranker returned an incomplete candidate batch")

        # Cache store — only when the reranker scored the full input batch.
        if self._cache is not None and query is not None:

            texts = [c.document.content for c in candidates]
            identity_keys = [c.identity_key for c in candidates]
            scores_by_id = {id(result.candidate): result.rerank_score for result in results}
            if len(scores_by_id) == len(candidates):
                scores = [scores_by_id[id(candidate)] for candidate in candidates]

                self._cache.set(
                    reranker=reranker_name,
                    query=query,
                    texts=texts,
                    identity_keys=identity_keys,
                    scores=scores,
                )

        return results