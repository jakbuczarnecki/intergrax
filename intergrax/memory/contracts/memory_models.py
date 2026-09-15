# © Artur Czarnecki. All rights reserved.

"""Neutral domain models for memory contracts (MEM-ENT domain language)."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from intergrax.globals.settings import GLOBAL_SETTINGS
from intergrax.memory.contracts.enterprise_memory_record import (
    MemoryProvenance,
    MemoryRecordGovernance,
    MemoryRecordLineage,
    MemoryRecordTrust,
    validate_memory_record_invariants,
)
from intergrax.utils.time_provider import SystemTimeProvider

__all__ = [
    "EnterpriseMemoryRecord",
    "MemoryImportance",
    "MemoryKind",
    "UserIdentity",
    "UserPreferences",
    "UserProfile",
    "UserProfileMemoryEntry",
    "UserProfileMemoryEntryNotFoundError",
]


class MemoryKind(Enum):
    USER_FACT = "user_fact"
    PREFERENCE = "preference"
    SESSION_SUMMARY = "session_summary"
    EPISODIC_EVENT = "episodic_event"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    ORG_FACT = "org_fact"
    POLICY = "policy"
    OTHER = "other"


class MemoryImportance(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class UserProfileMemoryEntryNotFoundError(LookupError):
    """Raised when a memory entry id does not exist on the user profile."""

    def __init__(self, entry_id: str) -> None:
        super().__init__(f"memory entry not found: {entry_id}")
        self.entry_id = entry_id


@dataclass
class UserProfileMemoryEntry:
    """
    Canonical enterprise memory record for user long-term memory (MEM-ENT-5).

    Stable ``entry_id`` is the memory identity; ``revision`` increments on semantic
    mutations. Typed provenance, trust, governance, and lineage are persisted
    independently of vendor storage.
    """

    entry_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    revision: int = 1
    content: str = ""
    session_id: Optional[str] = None
    kind: MemoryKind = MemoryKind.OTHER
    title: Optional[str] = None
    importance: MemoryImportance = MemoryImportance.MEDIUM
    created_at: str = field(
        default_factory=lambda: SystemTimeProvider.utc_now().isoformat()
    )
    updated_at: Optional[str] = None
    provenance: MemoryProvenance = field(default_factory=MemoryProvenance)
    trust: MemoryRecordTrust = field(default_factory=MemoryRecordTrust)
    governance: MemoryRecordGovernance = field(default_factory=MemoryRecordGovernance)
    lineage: MemoryRecordLineage = field(default_factory=MemoryRecordLineage)
    evidence_refs: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)
    valid_from: Optional[str] = None
    valid_until: Optional[str] = None
    deleted: bool = False
    modified: bool = False

    @property
    def memory_id(self) -> str:
        return self.entry_id

    def __post_init__(self) -> None:
        validate_memory_record_invariants(
            memory_id=self.entry_id,
            revision=self.revision,
            valid_from=self.valid_from,
            valid_until=self.valid_until,
            trust=self.trust,
            lineage=self.lineage,
            evidence_refs=self.evidence_refs,
        )

    def bump_revision_for_semantic_change(self) -> None:
        """Increment revision after a persisted semantic mutation."""
        self.revision += 1
        self.updated_at = SystemTimeProvider.utc_now().isoformat()
        validate_memory_record_invariants(
            memory_id=self.entry_id,
            revision=self.revision,
            valid_from=self.valid_from,
            valid_until=self.valid_until,
            trust=self.trust,
            lineage=self.lineage,
            evidence_refs=self.evidence_refs,
        )


EnterpriseMemoryRecord = UserProfileMemoryEntry


@dataclass
class UserIdentity:
    """High-level description of who the user is."""

    user_id: str
    display_name: Optional[str] = None
    role: Optional[str] = None
    domain_expertise: Optional[str] = None
    language: Optional[str] = GLOBAL_SETTINGS.default_language
    locale: Optional[str] = GLOBAL_SETTINGS.default_locale
    timezone: Optional[str] = GLOBAL_SETTINGS.default_timezone


@dataclass
class UserPreferences:
    """Stable user preferences that influence runtime and LLM behavior."""

    preferred_language: Optional[str] = None
    answer_length: Optional[str] = None
    tone: Optional[str] = None
    no_emojis_in_code: bool = False
    no_emojis_in_docs: bool = False
    prefer_markdown: bool = True
    prefer_code_blocks: bool = True
    default_project_context: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserProfile:
    """Canonical user profile aggregate."""

    identity: UserIdentity
    preferences: UserPreferences
    system_instructions: Optional[str] = None
    memory_entries: List[UserProfileMemoryEntry] = field(default_factory=list)
    version: int = 1
    entry_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    deleted: bool = False
    modified: bool = False
