# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""Budgeted agentic retrieval loop for deep-tier queries (M-RAG.13)."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Optional

from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.rag.profiles.rag_profile import RagProfile
from intergrax.rag.retrieval.agentic_policy import (
    latency_budget_exceeded,
    resolve_agentic_retriever_id,
)
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
        per_retriever_ids: list[str] = []
        per_latencies_ms: list[float] = []
        refine_calls = 0

        for iteration in range(max_iters):
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if iteration > 0 and latency_budget_exceeded(
                self._profile, elapsed_ms=elapsed_ms
            ):
                break

            retriever_id = resolve_agentic_retriever_id(
                self._profile,
                iteration_index=iteration,
                request_retriever_id=request.retriever_id,
            )
            step = RetrievalRequest(
                query=current_query,
                top_k=request.top_k,
                metadata_filter=request.metadata_filter,
                score_threshold=request.score_threshold,
                retriever_id=retriever_id,
                route_tier_override=request.route_tier_override or "deep",
            )
            iter_t0 = time.perf_counter()
            result = self._service.retrieve_single_pass(step, route_tier="deep")
            iter_ms = (time.perf_counter() - iter_t0) * 1000.0
            used_retriever = result.trace.retriever_id or retriever_id or ""
            per_retriever_ids.append(used_retriever)
            per_latencies_ms.append(
                result.trace.retrieval_latency_ms
                if result.trace.retrieval_latency_ms is not None
                else iter_ms
            )
            result.trace.agentic_iteration = iteration + 1
            last = result

            if not result.used:
                current_query = self._refiner.refine(current_query, result)
                refine_calls += 1
                if latency_budget_exceeded(
                    self._profile,
                    elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                ):
                    return self._finalize(
                        last,
                        t0=t0,
                        per_retriever_ids=per_retriever_ids,
                        per_latencies_ms=per_latencies_ms,
                        refine_calls=refine_calls,
                        stopped="latency_budget",
                    )
                continue

            strong = [c for c in result.chunks if c.score >= min_score]
            if len(strong) >= min_chunks:
                return self._finalize(
                    last,
                    t0=t0,
                    per_retriever_ids=per_retriever_ids,
                    per_latencies_ms=per_latencies_ms,
                    refine_calls=refine_calls,
                    stopped="sufficient_context",
                )

            current_query = self._refiner.refine(current_query, result)
            refine_calls += 1
            if latency_budget_exceeded(
                self._profile,
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            ):
                return self._finalize(
                    last,
                    t0=t0,
                    per_retriever_ids=per_retriever_ids,
                    per_latencies_ms=per_latencies_ms,
                    refine_calls=refine_calls,
                    stopped="latency_budget",
                )

        if last is not None:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            stopped = (
                "latency_budget"
                if latency_budget_exceeded(self._profile, elapsed_ms=elapsed_ms)
                and len(per_retriever_ids) < max_iters
                else "max_iterations"
            )
            return self._finalize(
                last,
                t0=t0,
                per_retriever_ids=per_retriever_ids,
                per_latencies_ms=per_latencies_ms,
                refine_calls=refine_calls,
                stopped=stopped,
            )
        return RetrievalResult(chunks=[], used=False, reason="agentic_no_result")

    def _finalize(
        self,
        result: RetrievalResult,
        *,
        t0: float,
        per_retriever_ids: list[str],
        per_latencies_ms: list[float],
        refine_calls: int,
        stopped: str,
    ) -> RetrievalResult:
        result.trace.agentic_stopped = stopped
        result.trace.agentic_total_latency_ms = (time.perf_counter() - t0) * 1000.0
        result.trace.agentic_per_iteration_retriever_ids = list(per_retriever_ids)
        result.trace.agentic_per_iteration_latency_ms = list(per_latencies_ms)
        result.trace.agentic_refine_calls = refine_calls
        result.trace.agentic_latency_budget_ms = self._profile.agentic_max_total_latency_ms
        return result
