# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Dict, List

from intergrax.rag.retrievers.contracts.base_retriever import (
    BaseRetriever,
    RetrieverCandidate,
    RetrieverQuery,
)
from intergrax.rag.retrievers.registry.retriever_registry import RetrieverRegistry


class FusionRetriever(BaseRetriever):

    def __init__(
        self,
        registry: RetrieverRegistry,
        *,
        retrievers: List[str],
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
    ) -> List[RetrieverCandidate]:

        if not query.query_text:
            return []

        results: Dict[str, RetrieverCandidate] = {}
        scores: Dict[str, float] = {}

        for retriever_name in self._retrievers:

            retriever = self._registry.get(retriever_name)

            candidates = retriever.retrieve(query)

            for rank, cand in enumerate(candidates):

                rrf_score = 1.0 / (self._rrf_k + rank + 1)

                scores[cand.id] = scores.get(cand.id, 0.0) + rrf_score

                if cand.id not in results:
                    results[cand.id] = cand

        fused = list(results.values())

        fused.sort(
            key=lambda c: scores.get(c.id, 0.0),
            reverse=True,
        )

        top_k = int(query.top_k)

        return fused[:top_k]