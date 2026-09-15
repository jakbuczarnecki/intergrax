# © Artur Czarnecki. All rights reserved.

"""Typed models for memory strategy SPI (MEM-ENT-4)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from intergrax.llm.messages import ChatMessage
from intergrax.memory.session_summary_schema import SessionSummarySchema
from intergrax.memory.user_profile_memory import (
    MemoryImportance,
    MemoryKind,
    UserProfileMemoryEntry,
)


class MemoryPromotionAction(str, Enum):
    PROMOTE = "promote"
    SKIP = "skip"


@dataclass(frozen=True)
class MemoryCandidate:
    """Minimal consolidation candidate prior to profile entry materialization."""

    content: str
    kind: MemoryKind
    session_id: str
    title: str | None = None
    importance: MemoryImportance = MemoryImportance.MEDIUM
    tags: tuple[str, ...] = ()
    source: str = "session_consolidation"
    structured_summary: SessionSummarySchema | None = None
    include_episodic_from_summary: bool = False


@dataclass(frozen=True)
class MemoryExtractionRequest:
    user_id: str
    session_id: str
    messages: tuple[ChatMessage, ...]
    language: str
    max_facts: int
    max_preferences: int
    include_session_summary: bool
    default_fact_importance: MemoryImportance
    default_preference_importance: MemoryImportance
    default_summary_importance: MemoryImportance
    temperature: float | None = None
    run_id: str | None = None


@dataclass(frozen=True)
class MemoryExtractionResult:
    candidates: tuple[MemoryCandidate, ...]


@dataclass(frozen=True)
class MemoryDeduplicationRequest:
    existing: tuple[UserProfileMemoryEntry, ...]
    incoming: tuple[UserProfileMemoryEntry, ...]


@dataclass(frozen=True)
class MemoryDeduplicationResult:
    accepted: tuple[UserProfileMemoryEntry, ...]
    rejected_as_duplicate: tuple[UserProfileMemoryEntry, ...]


@dataclass(frozen=True)
class MemoryPromotionRequest:
    user_id: str
    session_id: str
    entries: tuple[UserProfileMemoryEntry, ...]


@dataclass(frozen=True)
class MemoryPromotionDecision:
    entry: UserProfileMemoryEntry
    action: MemoryPromotionAction
    reason: str | None = None


@dataclass(frozen=True)
class MemoryPromotionResult:
    decisions: tuple[MemoryPromotionDecision, ...]
