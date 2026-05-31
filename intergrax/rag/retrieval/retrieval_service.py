# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""Unified retrieval entry point for ``rag.retrieve``, Nexus, and diagnostics."""

from __future__ import annotations

import time
from typing import Any, List, Optional

from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter

from intergrax.rag.profiles.rag_profile import RagProfile
from intergrax.rag.retrieval.retrieval_request import RetrievalRequest
from intergrax.rag.retrieval.retrieval_result import RetrievalChunk, RetrievalResult, RetrievalTrace
from intergrax.rag.retrievers.contracts.base_retriever_manager import BaseRetrieverManager
from intergrax.rag.rerankers.contracts.base_reranker_manager import BaseRerankerManager
from intergrax.rag.rerankers.contracts.reranker_types import RerankerCandidate
from intergrax.rag.routing.query_router import QueryRouter
class RetrievalService:
    """
    Single Tier-0 retrieval pipeline: route → retrieve → optional rerank → filter.

    Depends on injected managers; does not import vector-store SDKs or fixed parsers.
    """

    def __init__(
        self,
        *,
        retriever_manager: BaseRetrieverManager,
        reranker_manager: Optional[BaseRerankerManager] = None,
        profile: Optional[RagProfile] = None,
        query_router: Optional[QueryRouter] = None,
        llm_for_agentic: Optional[LLMAdapter] = None,
    ) -> None:
        self._retriever_manager = retriever_manager
        self._reranker_manager = reranker_manager
        self._profile = profile or RagProfile()
        self._router = query_router or QueryRouter(self._profile)
        self._llm_for_agentic = llm_for_agentic

    @property
    def profile(self) -> RagProfile:
        return self._profile

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        query = (request.query or "").strip()
        if not query:
            trace = RetrievalTrace()
            return RetrievalResult(chunks=[], used=False, reason="empty_query", trace=trace)

        tier = request.route_tier_override or self._router.route(query)
        if tier == "deep" and self._profile.agentic_enabled:
            from intergrax.rag.retrieval.agentic_loop import AgenticRetrievalLoop

            loop = AgenticRetrievalLoop(self, self._profile, llm=self._llm_for_agentic)
            return loop.run(request)

        return self.retrieve_single_pass(request, route_tier=str(tier))

    def retrieve_single_pass(
        self,
        request: RetrievalRequest,
        *,
        route_tier: Optional[str] = None,
    ) -> RetrievalResult:
        trace = RetrievalTrace()
        query = (request.query or "").strip()
        if not query:
            return RetrievalResult(chunks=[], used=False, reason="empty_query", trace=trace)

        tier = route_tier or request.route_tier_override or self._router.route(query)
        trace.route_tier = str(tier)

        retriever_id = request.retriever_id or self._profile.effective_retriever(route_tier=str(tier))
        trace.retriever_id = retriever_id

        prefetch_k = int(request.top_k or self._profile.prefetch_top_k)
        final_k = int(request.top_k or self._profile.final_top_k)
        if prefetch_k < final_k:
            prefetch_k = max(final_k, self._profile.prefetch_top_k)

        t0 = time.perf_counter()
        candidates = self._retriever_manager.retrieve(
            query,
            retriever_id=retriever_id,
            top_k=prefetch_k,
            metadata_filter=request.metadata_filter,
            include_embeddings=False,
        )
        trace.retrieval_latency_ms = (time.perf_counter() - t0) * 1000.0
        trace.candidates_before_rerank = len(candidates)

        if not candidates:
            return RetrievalResult(chunks=[], used=False, reason="no_hits", trace=trace)

        chunks = _candidates_to_chunks(candidates)

        use_rerank = self._profile.enable_rerank and self._reranker_manager is not None
        trace.rerank_enabled = use_rerank
        if use_rerank and self._reranker_manager is not None:
            reranker_id = self._profile.reranker_id
            trace.reranker_id = reranker_id
            rerank_candidates = [
                RerankerCandidate(
                    id=c.id,
                    text=c.content,
                    metadata=c.metadata,
                    original_score=c.score,
                )
                for c in candidates
            ]
            t1 = time.perf_counter()
            reranked = self._reranker_manager.rerank(
                query=query,
                candidates=rerank_candidates,
                limit=final_k,
                reranker_id=reranker_id,
            )
            trace.rerank_latency_ms = (time.perf_counter() - t1) * 1000.0
            chunks = [
                RetrievalChunk(
                    id=r.candidate.id,
                    text=r.candidate.text,
                    score=float(r.score),
                    metadata=dict(r.candidate.metadata or {}),
                )
                for r in reranked
            ]
            trace.candidates_after_rerank = len(chunks)
        else:
            chunks = chunks[:final_k]
            trace.candidates_after_rerank = len(chunks)

        threshold = request.score_threshold
        if threshold is None:
            threshold = self._profile.score_threshold
        if threshold is not None:
            chunks = [c for c in chunks if c.score >= float(threshold)]

        if not chunks:
            return RetrievalResult(chunks=[], used=False, reason="below_score_threshold", trace=trace)

        return RetrievalResult(chunks=chunks, used=True, reason="ok", trace=trace)


def _candidates_to_chunks(candidates: List[Any]) -> List[RetrievalChunk]:
    out: List[RetrievalChunk] = []
    for c in candidates:
        text = (getattr(c, "content", None) or "").strip()
        if not text:
            continue
        out.append(
            RetrievalChunk(
                id=str(getattr(c, "id", "unknown")),
                text=text,
                score=float(getattr(c, "score", 0.0) or 0.0),
                metadata=dict(getattr(c, "metadata", None) or {}),
            )
        )
    return out
