# © Artur Czarnecki. All rights reserved.

"""Conservative conflict detection and fail-safe resolution (MEM-ENT-6)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from intergrax.memory.contracts.enterprise_memory_record import parse_memory_record_timestamp
from intergrax.memory.user_profile_memory import UserProfileMemoryEntry
from intergrax.memory.strategies.recall_models import (
    MemoryConflict,
    MemoryConflictDetectionRequest,
    MemoryConflictDetectionResult,
    MemoryConflictKind,
    MemoryConflictResolutionAction,
    MemoryConflictResolutionDecision,
    MemoryConflictResolutionRequest,
    MemoryConflictResolutionResult,
    MemoryRankedCandidate,
    MemorySupersessionIntent,
)
from intergrax.memory.strategies.recall_validation import (
    validate_conflict_records,
    validate_resolution_decisions,
)


def _normalized_title(entry: MemoryRankedCandidate) -> str:
    title = (entry.candidate.record.title or "").strip().lower()
    return title


def _same_subject_key(a: MemoryRankedCandidate, b: MemoryRankedCandidate) -> bool:
    title_a = _normalized_title(a)
    title_b = _normalized_title(b)
    if title_a and title_b and title_a == title_b:
        return True
    meta_a = a.candidate.record.metadata or {}
    meta_b = b.candidate.record.metadata or {}
    key_a = str(meta_a.get("subject_key") or "").strip().lower()
    key_b = str(meta_b.get("subject_key") or "").strip().lower()
    return bool(key_a and key_b and key_a == key_b)


def _valid_from_instant(record: UserProfileMemoryEntry) -> datetime | None:
    value = record.valid_from
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return parse_memory_record_timestamp("valid_from", text)
    except ValueError:
        return None


def _semantic_supersession_evidence(
    left: UserProfileMemoryEntry,
    right: UserProfileMemoryEntry,
) -> tuple[UserProfileMemoryEntry, UserProfileMemoryEntry] | None:
    """Return (older, newer) when valid_from provides unambiguous ordering."""
    left_vf = _valid_from_instant(left)
    right_vf = _valid_from_instant(right)
    if left_vf is None or right_vf is None:
        return None
    if left_vf.tzinfo is not None and right_vf.tzinfo is None:
        return None
    if left_vf.tzinfo is None and right_vf.tzinfo is not None:
        return None
    if left_vf == right_vf:
        return None
    if left_vf < right_vf:
        return left, right
    return right, left


@dataclass(frozen=True, slots=True)
class ConservativeConflictDetectionConfig:
    max_pairwise_candidates: int = 32


class ConservativeMemoryConflictDetectionStrategy:
    strategy_id = "conservative_conflict_detection_v1"

    def __init__(self, config: ConservativeConflictDetectionConfig | None = None) -> None:
        self._config = config or ConservativeConflictDetectionConfig()

    def detect(self, request: MemoryConflictDetectionRequest) -> MemoryConflictDetectionResult:
        limit = min(request.max_pairwise_candidates, self._config.max_pairwise_candidates)
        pool = request.ranked[:limit]
        conflicts: list[MemoryConflict] = []
        for i, left in enumerate(pool):
            for right in pool[i + 1 :]:
                left_rec = left.candidate.record
                right_rec = right.candidate.record
                if left_rec.kind != right_rec.kind:
                    continue
                if not _same_subject_key(left, right):
                    continue
                if (left_rec.content or "").strip() == (right_rec.content or "").strip():
                    continue
                conflict_id = f"{left_rec.entry_id}:{right_rec.entry_id}"
                kind = MemoryConflictKind.CONTRADICTION
                if _semantic_supersession_evidence(left_rec, right_rec) is not None:
                    kind = MemoryConflictKind.POTENTIAL_SUPERSESSION
                conflicts.append(
                    MemoryConflict(
                        conflict_id=conflict_id,
                        records=(left_rec, right_rec),
                        kind=kind,
                        reason="same_kind_and_subject_with_different_content",
                    )
                )
        result = MemoryConflictDetectionResult(conflicts=tuple(conflicts))
        validate_conflict_records(result.conflicts, request.ranked)
        return result


class FailSafeMemoryConflictResolutionStrategy:
    strategy_id = "fail_safe_conflict_resolution_v1"

    def resolve(self, request: MemoryConflictResolutionRequest) -> MemoryConflictResolutionResult:
        decisions: list[MemoryConflictResolutionDecision] = []
        ranked_by_id = {r.candidate.record.entry_id: r for r in request.ranked}
        for conflict in request.conflicts:
            if len(conflict.records) != 2:
                decisions.append(
                    MemoryConflictResolutionDecision(
                        conflict_id=conflict.conflict_id,
                        action=MemoryConflictResolutionAction.ESCALATE,
                        reason="unsupported_conflict_cardinality",
                    )
                )
                continue
            first, second = conflict.records
            first_ranked = ranked_by_id.get(first.entry_id)
            second_ranked = ranked_by_id.get(second.entry_id)
            if first_ranked is None or second_ranked is None:
                decisions.append(
                    MemoryConflictResolutionDecision(
                        conflict_id=conflict.conflict_id,
                        action=MemoryConflictResolutionAction.KEEP_BOTH,
                        reason="missing_ranked_evidence",
                    )
                )
                continue
            if conflict.kind is MemoryConflictKind.POTENTIAL_SUPERSESSION:
                ordered = _semantic_supersession_evidence(first, second)
                if ordered is not None:
                    older_rec, newer_rec = ordered
                    decisions.append(
                        MemoryConflictResolutionDecision(
                            conflict_id=conflict.conflict_id,
                            action=MemoryConflictResolutionAction.SUPERSEDE_EXISTING,
                            supersession_intent=MemorySupersessionIntent(
                                superseded_memory_id=older_rec.entry_id,
                                superseding_memory_id=newer_rec.entry_id,
                                reason="valid_from_semantic_ordering",
                            ),
                            reason="clear_supersession_evidence",
                        )
                    )
                    continue
            decisions.append(
                MemoryConflictResolutionDecision(
                    conflict_id=conflict.conflict_id,
                    action=MemoryConflictResolutionAction.KEEP_BOTH,
                    reason="insufficient_evidence_for_destructive_resolution",
                )
            )
        result = MemoryConflictResolutionResult(decisions=tuple(decisions))
        validate_resolution_decisions(result.decisions, request.conflicts, request.ranked)
        return result
