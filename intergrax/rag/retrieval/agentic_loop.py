# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""Budgeted agentic retrieval loop for deep-tier queries (M-RAG.13)."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Optional

from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.rag.profiles.rag_profile import RagProfile
from intergrax.rag.retrieval.query_refiner import QueryRefiner, resolve_query_refiner
from intergrax.rag.retrieval.retrieval_request import RetrievalRequest
from intergrax.rag.retrieval.retrieval_result import RetrievalResult

if TYPE_CHECKING:
    from intergrax.rag.retrieval.retrieval_service import RetrievalService


class AgenticRetrievalLoop:
    """
    Iteratively refine the query when initial retrieval is insufficient.

    Uses deterministic query refinement (no mandatory LLM) to stay Tier-0 pure.
  Optional ``LLMAdapter`` can be added later via profile extras.
    """

    def __init__(
        self,
        service: RetrievalService,
        profile: RagProfile,
        *,
        llm: Optional[LLMAdapter] = None,
        query_refiner: Optional[QueryRefiner] = None,
    ) -> None:  # noqa: F821
        self._service = service
        self._profile = profile
        self._refiner = query_refiner or resolve_query_refiner(profile, llm=llm)

    def run(self, request: RetrievalRequest) -> RetrievalResult:
        max_iters = max(1, int(self._profile.agentic_max_iterations))
        min_chunks = max(1, int(self._profile.agentic_min_chunks))
        min_score = float(self._profile.agentic_min_score)

        current_query = request.query
        last: Optional[RetrievalResult] = None
        t0 = time.perf_counter()

        for iteration in range(max_iters):
            step = RetrievalRequest(
                query=current_query,
                top_k=request.top_k,
                metadata_filter=request.metadata_filter,
                score_threshold=request.score_threshold,
                retriever_id=request.retriever_id,
                route_tier_override=request.route_tier_override or "deep",
            )
            result = self._service.retrieve_single_pass(step, route_tier="deep")
            result.trace.agentic_iteration = iteration + 1
            last = result

            if not result.used:
                current_query = self._refiner.refine(current_query, result)
                continue

            strong = [
                c for c in result.chunks if c.score >= min_score
            ]
            if len(strong) >= min_chunks:
                result.trace.agentic_stopped = "sufficient_context"
                result.trace.agentic_total_latency_ms = (time.perf_counter() - t0) * 1000.0
                return result

            current_query = self._refiner.refine(current_query, result)

        if last is not None:
            last.trace.agentic_stopped = "max_iterations"
            last.trace.agentic_total_latency_ms = (time.perf_counter() - t0) * 1000.0
            return last
        return RetrievalResult(chunks=[], used=False, reason="agentic_no_result")
