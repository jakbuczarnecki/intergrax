# © Artur Czarnecki. All rights reserved.

"""Typed models for recall ranking and conflict strategies (MEM-ENT-6)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from intergrax.memory.contracts.memory_models import UserProfileMemoryEntry
from intergrax.memory.contracts.memory_recall import (
    MemoryRankingScore,
    MemoryRecallReasonCode,
    MemorySupersessionIntent,
)

__all__ = [
    "MemoryConflict",
    "MemoryConflictDetectionRequest",
    "MemoryConflictDetectionResult",
    "MemoryConflictKind",
    "MemoryConflictResolutionAction",
    "MemoryConflictResolutionDecision",
    "MemoryConflictResolutionRequest",
    "MemoryConflictResolutionResult",
    "MemoryRankingRequest",
    "MemoryRankingResult",
    "MemoryRankingScore",
    "MemoryRankedCandidate",
    "MemoryRecallCandidate",
    "MemoryRecallReasonCode",
    "MemoryRetrievalSource",
    "MemorySupersessionIntent",
]


class MemoryRetrievalSource(str, Enum):
    SEMANTIC = "semantic"
    PROFILE_SCAN = "profile_scan"
    SESSION = "session"
    OTHER = "other"


class MemoryConflictKind(str, Enum):
    CONTRADICTION = "contradiction"
    POTENTIAL_SUPERSESSION = "potential_supersession"
    AMBIGUOUS = "ambiguous"


class MemoryConflictResolutionAction(str, Enum):
    KEEP_BOTH = "keep_both"
    PREFER_EXISTING = "prefer_existing"
    PREFER_CANDIDATE = "prefer_candidate"
    SUPERSEDE_EXISTING = "supersede_existing"
    MERGE_REQUIRED = "merge_required"
    ESCALATE = "escalate"


@dataclass(frozen=True, slots=True)
class MemoryRecallCandidate:
    record: UserProfileMemoryEntry
    retrieval_source: MemoryRetrievalSource
    retrieval_score: float | None = None
    retrieval_reason: str | None = None


@dataclass(frozen=True, slots=True)
class MemoryRankedCandidate:
    candidate: MemoryRecallCandidate
    score: MemoryRankingScore
    reason_codes: tuple[MemoryRecallReasonCode, ...] = ()


@dataclass(frozen=True, slots=True)
class MemoryRankingRequest:
    candidates: tuple[MemoryRecallCandidate, ...]
    query: str
    top_k: int
    as_of_iso: str | None = None


@dataclass(frozen=True, slots=True)
class MemoryRankingResult:
    ranked: tuple[MemoryRankedCandidate, ...]


@dataclass(frozen=True, slots=True)
class MemoryConflict:
    conflict_id: str
    records: tuple[UserProfileMemoryEntry, ...]
    kind: MemoryConflictKind
    reason: str


@dataclass(frozen=True, slots=True)
class MemoryConflictDetectionRequest:
    ranked: tuple[MemoryRankedCandidate, ...]
    max_pairwise_candidates: int = 32


@dataclass(frozen=True, slots=True)
class MemoryConflictDetectionResult:
    conflicts: tuple[MemoryConflict, ...]


@dataclass(frozen=True, slots=True)
class MemoryConflictResolutionDecision:
    conflict_id: str
    action: MemoryConflictResolutionAction
    supersession_intent: MemorySupersessionIntent | None = None
    reason: str = ""


@dataclass(frozen=True, slots=True)
class MemoryConflictResolutionRequest:
    conflicts: tuple[MemoryConflict, ...]
    ranked: tuple[MemoryRankedCandidate, ...]


@dataclass(frozen=True, slots=True)
class MemoryConflictResolutionResult:
    decisions: tuple[MemoryConflictResolutionDecision, ...]
