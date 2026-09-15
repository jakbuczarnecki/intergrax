# © Artur Czarnecki. All rights reserved.

"""Validation helpers for recall strategy contracts (MEM-ENT-6)."""

from __future__ import annotations

import math

from intergrax.memory.strategies.errors import MemoryStrategyContractError
from intergrax.memory.strategies.recall_models import (
    MemoryConflict,
    MemoryConflictResolutionDecision,
    MemoryRankingRequest,
    MemoryRankingResult,
    MemoryRecallCandidate,
    MemoryRankedCandidate,
    MemorySupersessionIntent,
)


def _candidate_ids(candidates: tuple[MemoryRecallCandidate, ...]) -> set[str]:
    return {c.record.entry_id for c in candidates}


def validate_ranking_result(
    request: MemoryRankingRequest,
    result: MemoryRankingResult,
) -> None:
    allowed = _candidate_ids(request.candidates)
    seen: set[str] = set()
    for ranked in result.ranked:
        entry_id = ranked.candidate.record.entry_id
        if entry_id not in allowed:
            raise MemoryStrategyContractError(
                f"ranking result contains unknown memory id: {entry_id}"
            )
        if entry_id in seen:
            raise MemoryStrategyContractError(
                f"ranking result contains duplicate memory id: {entry_id}"
            )
        seen.add(entry_id)
        total = ranked.score.total
        if not math.isfinite(total):
            raise MemoryStrategyContractError("ranking score must be finite")


def validate_conflict_records(
    conflicts: tuple[MemoryConflict, ...],
    ranked: tuple[MemoryRankedCandidate, ...],
) -> None:
    allowed = {r.candidate.record.entry_id for r in ranked}
    for conflict in conflicts:
        for record in conflict.records:
            if record.entry_id not in allowed:
                raise MemoryStrategyContractError(
                    f"conflict references unknown memory id: {record.entry_id}"
                )


def validate_supersession_intent(
    intent: MemorySupersessionIntent,
    allowed_ids: set[str],
) -> None:
    if intent.superseded_memory_id == intent.superseding_memory_id:
        raise MemoryStrategyContractError("supersession cannot target the same memory id")
    if intent.superseded_memory_id not in allowed_ids:
        raise MemoryStrategyContractError("superseded memory id not in candidate set")
    if intent.superseding_memory_id not in allowed_ids:
        raise MemoryStrategyContractError("superseding memory id not in candidate set")


def validate_resolution_decisions(
    decisions: tuple[MemoryConflictResolutionDecision, ...],
    conflicts: tuple[MemoryConflict, ...],
    ranked: tuple[MemoryRankedCandidate, ...],
) -> None:
    conflict_ids = {c.conflict_id for c in conflicts}
    allowed = {r.candidate.record.entry_id for r in ranked}
    for decision in decisions:
        if decision.conflict_id not in conflict_ids:
            raise MemoryStrategyContractError(
                f"resolution references unknown conflict id: {decision.conflict_id}"
            )
        if decision.supersession_intent is not None:
            validate_supersession_intent(decision.supersession_intent, allowed)
            a = decision.supersession_intent.superseded_memory_id
            b = decision.supersession_intent.superseding_memory_id
            reverse = any(
                d.supersession_intent is not None
                and d.supersession_intent.superseded_memory_id == b
                and d.supersession_intent.superseding_memory_id == a
                for d in decisions
                if d is not decision
            )
            if reverse:
                raise MemoryStrategyContractError(
                    "conflicting reciprocal supersession in one resolution batch"
                )
