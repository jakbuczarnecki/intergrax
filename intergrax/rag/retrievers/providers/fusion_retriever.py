# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import replace

from intergrax.rag.retrievers.contracts.base_retriever import (
    BaseRetriever,
    RetrievalHit,
    RetrieverQuery,
)
from intergrax.rag.retrievers.registry.retriever_registry import RetrieverRegistry


class FusionRetriever(BaseRetriever):

    requires_query_embedding = False

    def __init__(
        self,
        registry: RetrieverRegistry,
        *,
        retrievers: list[str],
        rrf_k: int = 60,
    ) -> None:        
        self._registry = registry
        self._retrievers = list(retrievers)
        self._rrf_k = int(rrf_k)

    @classmethod
    def name(cls) -> str:
        return "fusion"

    def retrieve(
        self,
        query: RetrieverQuery,
    ) -> tuple[RetrievalHit, ...]:

        if not query.query_text:
            return ()

        results: dict[tuple[str, str | None, str | None, str], RetrievalHit] = {}
        scores: dict[tuple[str, str | None, str | None, str], float] = {}

        for retriever_name in self._retrievers:

            retriever = self._registry.get(retriever_name)

            candidates = retriever.retrieve(query)

            for rank, cand in enumerate(candidates):
                key = (
                    cand.document.scope.tenant_id,
                    cand.document.scope.namespace,
                    cand.document.scope.workspace_id,
                    cand.vector_id or cand.document.identity.document_id,
                )
                rrf_score = 1.0 / (self._rrf_k + rank + 1)
                scores[key] = scores.get(key, 0.0) + rrf_score
                if key not in results:
                    results[key] = cand

        fused = list(results.values())

        fused.sort(
            key=lambda c: scores[
                (
                    c.document.scope.tenant_id,
                    c.document.scope.namespace,
                    c.document.scope.workspace_id,
                    c.vector_id or c.document.identity.document_id,
                )
            ],
            reverse=True,
        )

        top_k = int(query.top_k)

        return tuple(
            replace(
                cand,
                score=scores[
                    (
                        cand.document.scope.tenant_id,
                        cand.document.scope.namespace,
                        cand.document.scope.workspace_id,
                        cand.vector_id or cand.document.identity.document_id,
                    )
                ],
                rank=rank,
                channel="hybrid",
                retriever_name=self.name(),
            )
            for rank, cand in enumerate(fused[:top_k])
        )