# © Artur Czarnecki. All rights reserved.

"""Public recall decision types for Memory Control Plane (MEM-ENT-6)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

__all__ = [
    "MemoryRankingScore",
    "MemoryRecallReasonCode",
    "MemorySupersessionIntent",
]


class MemoryRecallReasonCode(str, Enum):
    HIGH_RELEVANCE = "high_relevance"
    TRUSTED_SOURCE = "trusted_source"
    FRESHER_RECORD = "fresher_record"
    SUPERSEDED_RECORD = "superseded_record"
    CONFLICT_UNRESOLVED = "conflict_unresolved"
    TEMPORAL_INACTIVE = "temporal_inactive"


@dataclass(frozen=True, slots=True)
class MemoryRankingScore:
    total: float
    relevance: float | None = None
    trust: float | None = None
    temporal: float | None = None
    freshness: float | None = None
    supersession_adjustment: float | None = None


@dataclass(frozen=True, slots=True)
class MemorySupersessionIntent:
    superseded_memory_id: str
    superseding_memory_id: str
    reason: str
