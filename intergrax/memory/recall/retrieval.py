# © Artur Czarnecki. All rights reserved.

"""Build typed recall candidates from user memory capabilities (MEM-ENT-6)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.memory.contracts.memory_control import UserMemoryRecallCapabilityResult
from intergrax.memory.contracts.memory_models import UserProfileMemoryEntry
from intergrax.memory.strategies.recall_models import MemoryRecallCandidate, MemoryRetrievalSource


@dataclass(frozen=True, slots=True)
class UserMemoryRecallRetrievalConfig:
    retrieval_candidate_multiplier: int = 3


def candidates_from_semantic_search(
    search_result: UserMemoryRecallCapabilityResult,
) -> tuple[MemoryRecallCandidate, ...]:
    candidates: list[MemoryRecallCandidate] = []
    for index, entry in enumerate(search_result.entries):
        score = (
            search_result.scores[index] if index < len(search_result.scores) else None
        )
        candidates.append(
            MemoryRecallCandidate(
                record=entry,
                retrieval_source=MemoryRetrievalSource.SEMANTIC,
                retrieval_score=score,
                retrieval_reason=search_result.reason,
            )
        )
    return tuple(candidates)


def candidates_from_profile_scan(
    entries: tuple[UserProfileMemoryEntry, ...],
    *,
    query: str,
    candidate_limit: int,
) -> tuple[MemoryRecallCandidate, ...]:
    needle = query.strip().lower()
    candidates: list[MemoryRecallCandidate] = []
    for entry in entries:
        if needle and needle not in (entry.content or "").lower():
            continue
        retrieval_score = 1.0 if needle else 0.5
        candidates.append(
            MemoryRecallCandidate(
                record=entry,
                retrieval_source=MemoryRetrievalSource.PROFILE_SCAN,
                retrieval_score=retrieval_score,
                retrieval_reason="keyword" if needle else "profile_scan",
            )
        )
        if len(candidates) >= candidate_limit:
            break
    return tuple(candidates)


def semantic_retrieval_top_k(final_top_k: int, config: UserMemoryRecallRetrievalConfig) -> int:
    multiplier = max(1, config.retrieval_candidate_multiplier)
    return max(final_top_k, final_top_k * multiplier)
