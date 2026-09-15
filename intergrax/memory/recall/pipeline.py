# © Artur Czarnecki. All rights reserved.

"""Recall decision pipeline: rank → detect → resolve (read-only) (MEM-ENT-6)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.memory.strategies.protocols import (
    MemoryConflictDetectionStrategy,
    MemoryConflictResolutionStrategy,
    MemoryRankingStrategy,
)
from intergrax.memory.strategies.recall_models import (
    MemoryConflict,
    MemoryConflictDetectionRequest,
    MemoryConflictResolutionAction,
    MemoryConflictResolutionRequest,
    MemoryConflictResolutionResult,
    MemoryRankingRequest,
    MemoryRecallCandidate,
    MemoryRecallReasonCode,
    MemoryRankedCandidate,
)
from intergrax.memory.strategies.recall_validation import validate_ranking_result


@dataclass(frozen=True, slots=True)
class MemoryRecallPipelineResult:
    ranked: tuple[MemoryRankedCandidate, ...]
    conflicts: tuple[MemoryConflict, ...]
    resolution: MemoryConflictResolutionResult | None
    unresolved_conflict_entry_ids: frozenset[str]


def run_recall_decision_pipeline(
    *,
    candidates: tuple[MemoryRecallCandidate, ...],
    query: str,
    top_k: int,
    ranking: MemoryRankingStrategy,
    conflict_detection: MemoryConflictDetectionStrategy | None = None,
    conflict_resolution: MemoryConflictResolutionStrategy | None = None,
    max_pairwise_candidates: int = 32,
) -> MemoryRecallPipelineResult:
    ranking_request = MemoryRankingRequest(
        candidates=candidates,
        query=query,
        top_k=top_k,
    )
    ranking_result = ranking.rank(ranking_request)
    validate_ranking_result(ranking_request, ranking_result)

    conflicts = ()
    resolution: MemoryConflictResolutionResult | None = None
    unresolved_ids: set[str] = set()

    if conflict_detection is not None and ranking_result.ranked:
        detection_request = MemoryConflictDetectionRequest(
            ranked=ranking_result.ranked,
            max_pairwise_candidates=max_pairwise_candidates,
        )
        detection_result = conflict_detection.detect(detection_request)
        conflicts = detection_result.conflicts
        if conflict_resolution is not None and detection_result.conflicts:
            resolution_request = MemoryConflictResolutionRequest(
                conflicts=detection_result.conflicts,
                ranked=ranking_result.ranked,
            )
            resolution = conflict_resolution.resolve(resolution_request)
            for decision in resolution.decisions:
                if decision.action in {
                    MemoryConflictResolutionAction.KEEP_BOTH,
                    MemoryConflictResolutionAction.ESCALATE,
                    MemoryConflictResolutionAction.MERGE_REQUIRED,
                }:
                    for conflict in detection_result.conflicts:
                        if conflict.conflict_id == decision.conflict_id:
                            for record in conflict.records:
                                unresolved_ids.add(record.entry_id)

    for ranked in ranking_result.ranked:
        if MemoryRecallReasonCode.CONFLICT_UNRESOLVED in ranked.reason_codes:
            unresolved_ids.add(ranked.candidate.record.entry_id)

    return MemoryRecallPipelineResult(
        ranked=ranking_result.ranked,
        conflicts=conflicts,
        resolution=resolution,
        unresolved_conflict_entry_ids=frozenset(unresolved_ids),
    )
