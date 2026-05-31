# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""Budgeted agentic retrieval loop for deep-tier queries (M-RAG.13)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from intergrax.rag.profiles.rag_profile import RagProfile
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

    def __init__(self, service: RetrievalService, profile: RagProfile) -> None:  # noqa: F821
        self._service = service
        self._profile = profile

    def run(self, request: RetrievalRequest) -> RetrievalResult:
        max_iters = max(1, int(self._profile.agentic_max_iterations))
        min_chunks = max(1, int(self._profile.agentic_min_chunks))
        min_score = float(self._profile.agentic_min_score)

        current_query = request.query
        last: Optional[RetrievalResult] = None

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
                current_query = self._refine_query(current_query, result)
                continue

            strong = [
                c for c in result.chunks if c.score >= min_score
            ]
            if len(strong) >= min_chunks:
                result.trace.agentic_stopped = "sufficient_context"
                return result

            current_query = self._refine_query(current_query, result)

        if last is not None:
            last.trace.agentic_stopped = "max_iterations"
            return last
        return RetrievalResult(chunks=[], used=False, reason="agentic_no_result")

    @staticmethod
    def _refine_query(query: str, result: RetrievalResult) -> str:
        if result.chunks:
            terms = []
            for chunk in result.chunks[:2]:
                words = [w for w in chunk.text.split() if len(w) > 4][:3]
                terms.extend(words)
            if terms:
                return f"{query} {' '.join(dict.fromkeys(terms))}"
        return query
