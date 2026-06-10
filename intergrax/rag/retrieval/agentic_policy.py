# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""Agentic retrieval schedule and latency budget helpers (M-RAG.34)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from intergrax.rag.profiles.rag_profile import RagProfile


def resolve_agentic_retriever_id(
    profile: RagProfile,
    *,
    iteration_index: int,
    request_retriever_id: Optional[str],
) -> Optional[str]:
    """
    Return per-iteration retriever override when ``agentic_iteration_retriever_ids`` is set.

    Uses the last schedule entry when the iteration index exceeds the schedule length.
    """
    schedule = profile.agentic_iteration_retriever_ids
    if not schedule:
        return request_retriever_id
    idx = min(max(0, iteration_index), len(schedule) - 1)
    return schedule[idx]


def latency_budget_exceeded(
    profile: RagProfile,
    *,
    elapsed_ms: float,
) -> bool:
    budget = profile.agentic_max_total_latency_ms
    if budget is None:
        return False
    return elapsed_ms >= float(budget)
