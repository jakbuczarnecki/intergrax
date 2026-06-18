# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""Unified retrieval entry point for ``rag.retrieve``, Nexus, and diagnostics."""

from __future__ import annotations
from intergrax.utils import attribute_access

import time
from typing import Any, List, Optional

from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter

from intergrax.rag.profiles.rag_profile import RagProfile
from intergrax.rag.governance.embedding_version_policy import filter_chunks_by_embedding_version
from intergrax.rag.retrieval.citation import citations_from_chunks
from intergrax.rag.retrieval.retrieval_errors import RetrievalError
from intergrax.rag.retrieval.retrieval_request import RetrievalRequest
from intergrax.rag.retrieval.retrieval_result import RetrievalChunk, RetrievalResult, RetrievalTrace
from intergrax.rag.retrievers.contracts.base_retriever_manager import BaseRetrieverManager
from intergrax.rag.rerankers.contracts.base_reranker_manager import BaseRerankerManager
from intergrax.rag.rerankers.contracts.reranker_types import RerankerCandidate
from intergrax.rag.routing.query_router import QueryRouter
from intergrax.rag.tracking.metrics import record_retrieval
from intergrax.rag.tracking.rag_spans import rag_span


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
        llm_for_routing: Optional[LLMAdapter] = None,
    ) -> None:
        self._retriever_manager = retriever_manager
        self._reranker_manager = reranker_manager
        self._profile = profile or RagProfile()
        router_llm = llm_for_routing or llm_for_agentic
        self._router = query_router or QueryRouter(self._profile, llm=router_llm)
        self._llm_for_agentic = llm_for_agentic

    @property
    def profile(self) -> RagProfile:
        return self._profile

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        query = (request.query or "").strip()
        with rag_span(
            "rag.retrieve",
            attributes={
                "rag.query.length": len(query),
                "rag.tenant_id": attribute_access.optional(request, "tenant_id", None),
            },
        ):
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
        query = (request.query or "").strip()
        with rag_span(
            "rag.retrieve.single_pass",
            attributes={
                "rag.query.length": len(query),
                "rag.tenant_id": attribute_access.optional(request, "tenant_id", None),
            },
        ):
            trace = RetrievalTrace()
            if not query:
                return RetrievalResult(chunks=[], used=False, reason="empty_query", trace=trace)

            tier = route_tier or request.route_tier_override or self._router.route(query)
            trace.route_tier = str(tier)
            trace.route_classifier = self._router.last_route_classifier

            retriever_id = request.retriever_id or self._profile.effective_retriever(route_tier=str(tier))
            trace.retriever_id = retriever_id
            trace.hybrid_used = retriever_id in ("hybrid", "graph_rag") or self._profile.native_hybrid_enabled

            final_k = request.resolved_final_k(self._profile.final_top_k)
            prefetch_k = request.resolved_prefetch_k(self._profile.prefetch_top_k, final_k)

            t0 = time.perf_counter()
            try:
                candidates = self._retriever_manager.retrieve(
                    query,
                    retriever_id=retriever_id,
                    top_k=prefetch_k,
                    metadata_filter=request.metadata_filter,
                    include_embeddings=False,
                )
            except RetrievalError as exc:
                trace.retrieval_error_kind = exc.kind.value
                trace.attempted_retriever_ids = list(exc.attempted_retriever_ids)
                trace.retrieval_latency_ms = (time.perf_counter() - t0) * 1000.0
                return RetrievalResult(
                    chunks=[],
                    used=False,
                    reason="retriever_failed",
                    trace=trace,
                )
            trace.retrieval_latency_ms = (time.perf_counter() - t0) * 1000.0
            _apply_retriever_execution_trace(self._retriever_manager, trace)
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
                        score=float(r.rerank_score),
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

            chunks, filtered_count, version_warnings = filter_chunks_by_embedding_version(
                chunks,
                profile=self._profile,
            )
            trace.embedding_version_filtered_count = filtered_count
            trace.embedding_version_warnings = list(version_warnings)

            if not chunks:
                if filtered_count > 0:
                    return RetrievalResult(
                        chunks=[],
                        used=False,
                        reason="embedding_version_mismatch",
                        trace=trace,
                    )
                return RetrievalResult(chunks=[], used=False, reason="below_score_threshold", trace=trace)

            result = RetrievalResult(
                chunks=chunks,
                used=True,
                reason="ok",
                trace=trace,
                citations=citations_from_chunks(chunks),
            )
            _record_retrieval_metrics(
                request=request,
                trace=trace,
                hits=len(chunks),
                tenant_id=attribute_access.optional(request, "tenant_id", None),
            )
            return result


def _apply_retriever_execution_trace(
    retriever_manager: BaseRetrieverManager,
    trace: RetrievalTrace,
) -> None:
    execution = attribute_access.optional(retriever_manager, "last_execution", None)
    if execution is None:
        return
    trace.retriever_id = execution.used_retriever_id
    trace.attempted_retriever_ids = list(execution.attempted_retriever_ids)
    trace.fallback_applied = execution.fallback_applied
    trace.hybrid_used = (
        execution.used_retriever_id in ("hybrid", "graph_rag")
        or trace.hybrid_used
    )
    if execution.channel_contributions:
        trace.channel_contributions = dict(execution.channel_contributions)
    if execution.graph_expanded_node_ids:
        trace.graph_expanded_node_ids = list(execution.graph_expanded_node_ids)
    if execution.graph_provenance_summary:
        trace.graph_provenance_summary = execution.graph_provenance_summary
    if execution.graph_provenance_records:
        trace.graph_provenance_records = list(execution.graph_provenance_records)


def _record_retrieval_metrics(
    *,
    request: RetrievalRequest,
    trace: RetrievalTrace,
    hits: int,
    tenant_id: Optional[str] = None,
) -> None:
    record_retrieval(
        tenant_id=tenant_id or "_platform",
        retriever_id=trace.retriever_id or "unknown",
        route_tier=trace.route_tier or "standard",
        retrieval_latency_ms=float(trace.retrieval_latency_ms or 0.0),
        rerank_latency_ms=float(trace.rerank_latency_ms or 0.0),
        agentic_iterations=int(trace.agentic_iteration or 0),
        hybrid_used=bool(trace.hybrid_used),
        hits=hits,
        recall_at_k=trace.recall_at_k,
    )


def _candidates_to_chunks(candidates: List[Any]) -> List[RetrievalChunk]:
    out: List[RetrievalChunk] = []
    for c in candidates:
        text = (attribute_access.optional(c, "content", None) or "").strip()
        if not text:
            continue
        out.append(
            RetrievalChunk(
                id=str(attribute_access.optional(c, "id", "unknown")),
                text=text,
                score=float(attribute_access.optional(c, "score", 0.0) or 0.0),
                metadata=dict(attribute_access.optional(c, "metadata", None) or {}),
            )
        )
    return out
