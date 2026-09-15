# © Artur Czarnecki. All rights reserved.

"""Durable memory lifecycle contracts (MEM-ENT-2)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from intergrax.memory.user_profile_memory import UserProfile, UserProfileMemoryEntry

__all__ = [
    "MemoryLifecycleDisposition",
    "MemoryLifecycleOperation",
    "MemoryLifecycleOutcome",
    "MemoryProjectionFailureCategory",
    "MemoryProjectionFailureEvidence",
    "MemoryProjectionOperation",
    "MemoryProjectionOperationEvidence",
    "MemoryProjectionReconciliationDisposition",
    "MemoryProjectionReconciliationResult",
    "MemoryReconciliationDisposition",
    "MemoryReconciliationOutcome",
    "UserProfileMemoryProjection",
    "UserProfileMemoryReconciliationContext",
]


class MemoryLifecycleOperation(str, Enum):
    WRITE = "write"
    UPDATE = "update"
    DELETE_ENTRY = "delete_entry"
    CLEAR = "clear"
    DELETE_PROFILE = "delete_profile"


class MemoryLifecycleDisposition(str, Enum):
    """Aggregate lifecycle state after primary + projections."""

    UNCHANGED = "unchanged"
    CONSISTENT = "consistent"
    PRIMARY_FAILED = "primary_failed"
    PARTIAL_PROJECTION_FAILURE = "partial_projection_failure"


class MemoryProjectionOperation(str, Enum):
    UPSERT = "upsert"
    DELETE = "delete"
    REBUILD = "rebuild"


class MemoryProjectionFailureCategory(str, Enum):
    RETRYABLE = "retryable"
    PERMANENT = "permanent"


@dataclass(frozen=True, slots=True)
class MemoryProjectionFailureEvidence:
    projection_id: str
    operation: MemoryProjectionOperation
    category: MemoryProjectionFailureCategory
    message: str


@dataclass(frozen=True, slots=True)
class MemoryProjectionOperationEvidence:
    projection_id: str
    operation: MemoryProjectionOperation
    succeeded: bool
    failure: MemoryProjectionFailureEvidence | None = None


@dataclass(frozen=True, slots=True)
class MemoryLifecycleOutcome:
    operation: MemoryLifecycleOperation
    disposition: MemoryLifecycleDisposition
    user_id: str
    memory_entity_ids: tuple[str, ...]
    primary_applied: bool
    projection_evidence: tuple[MemoryProjectionOperationEvidence, ...]

    @property
    def requires_reconciliation(self) -> bool:
        return self.disposition is MemoryLifecycleDisposition.PARTIAL_PROJECTION_FAILURE


@dataclass(frozen=True, slots=True)
class UserProfileMemoryReconciliationContext:
    user_id: str
    profile: UserProfile | None
    authoritative_active_entry_ids: frozenset[str]


class MemoryProjectionReconciliationDisposition(str, Enum):
    CONSISTENT = "consistent"
    REPAIRED = "repaired"


@dataclass(frozen=True, slots=True)
class MemoryProjectionReconciliationResult:
    projection_id: str
    disposition: MemoryProjectionReconciliationDisposition


class UserProfileMemoryProjection(Protocol):
    """Pluggable derived representation for user-profile long-term memory."""

    @property
    def projection_id(self) -> str: ...

    async def upsert_memory_entry(
        self,
        user_id: str,
        entry: UserProfileMemoryEntry,
    ) -> None: ...

    async def delete_memory_entries(
        self,
        entry_ids: Sequence[str],
    ) -> None: ...

    async def reconcile(
        self,
        context: UserProfileMemoryReconciliationContext,
    ) -> MemoryProjectionReconciliationResult: ...


class MemoryReconciliationDisposition(str, Enum):
    CONSISTENT = "consistent"
    REPAIRED = "repaired"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class MemoryReconciliationOutcome:
    user_id: str
    disposition: MemoryReconciliationDisposition
    projection_evidence: tuple[MemoryProjectionOperationEvidence, ...]
