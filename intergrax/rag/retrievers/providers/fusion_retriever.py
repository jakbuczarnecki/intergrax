# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import replace

from intergrax.rag.retrieval.fusion import (
    RankFusionConfiguration,
    RankFusionStrategyPort,
    RankedRetrievalCandidate,
    RankedRetrievalChannel,
    ReciprocalRankFusionStrategy,
)
from intergrax.rag.retrievers.contracts.base_retriever import (
    BaseRetriever,
    RetrievalHit,
    RetrieverQuery,
)
from intergrax.rag.retrievers.registry.retriever_registry import RetrieverRegistry


def _retrieval_hit_identity(hit: RetrievalHit) -> str:
    return "|".join(
        (
            hit.document.scope.tenant_id,
            hit.document.scope.namespace or "",
            hit.document.scope.workspace_id or "",
            hit.vector_id or hit.document.identity.document_id,
        )
    )


class FusionRetriever(BaseRetriever):

    requires_query_embedding = False

    def __init__(
        self,
        registry: RetrieverRegistry,
        *,
        retrievers: list[str],
        rrf_k: int = 60,
        fusion_strategy: RankFusionStrategyPort[RetrievalHit] | None = None,
    ) -> None:
        self._registry = registry
        self._retrievers = list(retrievers)
        configuration = RankFusionConfiguration(rrf_k=int(rrf_k))
        self._fusion_strategy = fusion_strategy or ReciprocalRankFusionStrategy(
            configuration=configuration,
        )

    @classmethod
    def name(cls) -> str:
        return "fusion"

    def retrieve(
        self,
        query: RetrieverQuery,
    ) -> tuple[RetrievalHit, ...]:

        if not query.query_text:
            return ()

        channels: list[RankedRetrievalChannel[RetrievalHit]] = []

        for retriever_name in self._retrievers:
            retriever = self._registry.get(retriever_name)
            candidates = retriever.retrieve(query)
            ranked: list[RankedRetrievalCandidate[RetrievalHit]] = []
            for rank, cand in enumerate(candidates):
                ranked.append(
                    RankedRetrievalCandidate(
                        candidate_id=_retrieval_hit_identity(cand),
                        rank=rank,
                        payload=cand,
                    )
                )
            channels.append(
                RankedRetrievalChannel(
                    channel_key=retriever_name,
                    candidates=tuple(ranked),
                )
            )

        fused = self._fusion_strategy.fuse(tuple(channels), limit=int(query.top_k))

        return tuple(
            replace(
                item.payload,
                score=item.fusion_score,
                rank=item.fused_rank,
                channel="hybrid",
                retriever_name=self.name(),
            )
            for item in fused.candidates
            if item.payload is not None
        )
