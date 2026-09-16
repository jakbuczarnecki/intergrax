# © Artur Czarnecki. All rights reserved.

"""Durable memory lifecycle contracts (MEM-ENT-2)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from intergrax.contracts.agent_run import RequestIdentity
from intergrax.memory.contracts.memory_models import UserProfile, UserProfileMemoryEntry

__all__ = [
    "MemoryLifecycleDisposition",
    "MemoryLifecycleOperation",
    "MemoryLifecycleOutcome",
    "aggregate_memory_lifecycle_outcomes",
    "UserProfileMemoryMutationResult",
    "MemoryProjectionFailureCategory",
    "MemoryProjectionFailureEvidence",
    "MemoryProjectionOperation",
    "MemoryProjectionOperationEvidence",
    "MemoryProjectionReconciliationDisposition",
    "MemoryProjectionReconciliationResult",
    "MemoryReconciliationDisposition",
    "MemoryReconciliationOutcome",
    "UserProfileMemoryProjection",
    "UserProfileMemoryProjectionContext",
    "UserProfileMemoryReconciliationContext",
    "user_profile_memory_projection_context",
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


def aggregate_memory_lifecycle_outcomes(
    *,
    operation: MemoryLifecycleOperation,
    user_id: str,
    outcomes: Sequence[MemoryLifecycleOutcome],
) -> MemoryLifecycleOutcome:
    """Merge projection evidence and disposition for multi-record mutations."""
    ordered = tuple(outcomes)
    if not ordered:
        raise ValueError("aggregate_memory_lifecycle_outcomes requires at least one outcome")
    entity_ids: list[str] = []
    for outcome in ordered:
        for memory_id in outcome.memory_entity_ids:
            if memory_id not in entity_ids:
                entity_ids.append(memory_id)
    evidence: list[MemoryProjectionOperationEvidence] = []
    for outcome in ordered:
        evidence.extend(outcome.projection_evidence)
    if any(
        outcome.disposition is MemoryLifecycleDisposition.PARTIAL_PROJECTION_FAILURE
        for outcome in ordered
    ):
        disposition = MemoryLifecycleDisposition.PARTIAL_PROJECTION_FAILURE
    else:
        disposition = MemoryLifecycleDisposition.CONSISTENT
    return MemoryLifecycleOutcome(
        operation=operation,
        disposition=disposition,
        user_id=user_id,
        memory_entity_ids=tuple(entity_ids),
        primary_applied=all(outcome.primary_applied for outcome in ordered),
        projection_evidence=tuple(evidence),
    )


@dataclass(frozen=True, slots=True)
class UserProfileMemoryMutationResult:
    """Primary mutation result with lifecycle outcome (MEM-ENT-3R)."""

    lifecycle: MemoryLifecycleOutcome
    entry: UserProfileMemoryEntry | None = None


@dataclass(frozen=True, slots=True)
class UserProfileMemoryProjectionContext:
    """Trusted request authority for projection mutations (MEM-ENT-15-R)."""

    identity: RequestIdentity

    @property
    def user_id(self) -> str:
        uid = self.identity.user_id
        if uid is None or not uid.strip():
            raise ValueError("UserProfileMemoryProjectionContext requires identity.user_id")
        return uid


def user_profile_memory_projection_context(identity: RequestIdentity) -> UserProfileMemoryProjectionContext:
    return UserProfileMemoryProjectionContext(identity=identity)


@dataclass(frozen=True, slots=True)
class UserProfileMemoryReconciliationContext:
    identity: RequestIdentity
    profile: UserProfile | None
    authoritative_active_entry_ids: frozenset[str]

    @property
    def user_id(self) -> str:
        uid = self.identity.user_id
        if uid is None or not uid.strip():
            raise ValueError("UserProfileMemoryReconciliationContext requires identity.user_id")
        return uid


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
        context: UserProfileMemoryProjectionContext,
        entry: UserProfileMemoryEntry,
    ) -> None: ...

    async def delete_memory_entries(
        self,
        context: UserProfileMemoryProjectionContext,
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
