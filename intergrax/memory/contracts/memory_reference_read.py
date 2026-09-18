# © Artur Czarnecki. All rights reserved.

"""Scoped reference-first Memory read port (MP-5F-B1).

Memory owns read/retrieval semantics. This module exposes canonical record
references only — no payload hydration and no ContextView / MP-5 types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

from intergrax.contracts.agent_run import RequestIdentity

__all__ = [
    "MEMORY_REFERENCE_READ_DEFAULT_LIMIT",
    "MEMORY_REFERENCE_READ_MAX_LIMIT",
    "MemoryRecordCanonicalRef",
    "MemoryReferenceReadOutcome",
    "MemoryReferenceReadPort",
    "MemoryReferenceReadQuery",
    "MemoryReferenceReadRequest",
    "MemoryReferenceReadResult",
    "MemoryReferenceReadScope",
    "MemoryReferenceReadScopeError",
    "MemoryScopedResourceRef",
    "memory_reference_read_scope_to_entity_scope",
    "validate_memory_reference_read_request",
]

MEMORY_REFERENCE_READ_DEFAULT_LIMIT = 50
MEMORY_REFERENCE_READ_MAX_LIMIT = 200


class MemoryReferenceReadOutcome(str, Enum):
    OK = "ok"
    ACCESS_DENIED = "access_denied"
    SCOPE_REJECTED = "scope_rejected"
    INVALID_REQUEST = "invalid_request"
    UNAVAILABLE = "unavailable"


class MemoryReferenceReadScopeError(ValueError):
    """Request scope or fields violate Memory reference-read invariants."""


@dataclass(frozen=True, slots=True)
class MemoryScopedResourceRef:
    """Neutral resource binding within Memory read scope (not a foreign-domain type)."""

    resource_kind: str
    resource_id: str

    def __post_init__(self) -> None:
        kind = (self.resource_kind or "").strip()
        ident = (self.resource_id or "").strip()
        if not kind:
            raise MemoryReferenceReadScopeError("resource_kind must be non-empty")
        if not ident:
            raise MemoryReferenceReadScopeError("resource_id must be non-empty")
        object.__setattr__(self, "resource_kind", kind)
        object.__setattr__(self, "resource_id", ident)


@dataclass(frozen=True, slots=True)
class MemoryReferenceReadScope:
    """Memory-owned least-context boundary for reference enumeration.

    ``tenant_id`` is mandatory storage authority. ``workspace_id`` is mandatory
    for this read surface (aligned with entity/temporal and vector projection
    qualifiers). Optional ``user_id`` narrows user-profile memory. Optional
    ``resource`` further bounds records when the backing surface supports it.
    """

    tenant_id: str
    workspace_id: str
    user_id: str | None = None
    resource: MemoryScopedResourceRef | None = None

    def __post_init__(self) -> None:
        tenant = (self.tenant_id or "").strip()
        workspace = (self.workspace_id or "").strip()
        if not tenant:
            raise MemoryReferenceReadScopeError("tenant_id must be non-empty")
        if not workspace:
            raise MemoryReferenceReadScopeError("workspace_id must be non-empty")
        user = self.user_id
        if user is not None:
            user = user.strip()
            if not user:
                raise MemoryReferenceReadScopeError("user_id when set must be non-empty")
        object.__setattr__(self, "tenant_id", tenant)
        object.__setattr__(self, "workspace_id", workspace)
        object.__setattr__(self, "user_id", user)


@dataclass(frozen=True, slots=True)
class MemoryReferenceReadQuery:
    """Bounded, typed query parameters — no open-ended retrieval overrides."""

    limit: int = MEMORY_REFERENCE_READ_DEFAULT_LIMIT

    def __post_init__(self) -> None:
        if self.limit < 1:
            raise MemoryReferenceReadScopeError("limit must be >= 1")
        if self.limit > MEMORY_REFERENCE_READ_MAX_LIMIT:
            raise MemoryReferenceReadScopeError(
                f"limit must be <= {MEMORY_REFERENCE_READ_MAX_LIMIT}"
            )


@dataclass(frozen=True, slots=True)
class MemoryReferenceReadRequest:
    scope: MemoryReferenceReadScope
    query: MemoryReferenceReadQuery = field(default_factory=MemoryReferenceReadQuery)


@dataclass(frozen=True, slots=True)
class MemoryRecordCanonicalRef:
    """Canonical Memory record identity (maps to ContextView ``record_ref`` externally)."""

    tenant_id: str
    memory_id: str
    revision: int

    def __post_init__(self) -> None:
        tenant = (self.tenant_id or "").strip()
        memory_id = (self.memory_id or "").strip()
        if not tenant:
            raise MemoryReferenceReadScopeError("tenant_id must be non-empty")
        if not memory_id:
            raise MemoryReferenceReadScopeError("memory_id must be non-empty")
        if self.revision < 1:
            raise MemoryReferenceReadScopeError("revision must be >= 1")
        object.__setattr__(self, "tenant_id", tenant)
        object.__setattr__(self, "memory_id", memory_id)


@dataclass(frozen=True, slots=True)
class MemoryReferenceReadResult:
    outcome: MemoryReferenceReadOutcome
    references: tuple[MemoryRecordCanonicalRef, ...] = ()
    evaluated_scope: MemoryReferenceReadScope | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        if self.outcome is not MemoryReferenceReadOutcome.OK and self.references:
            raise MemoryReferenceReadScopeError(
                "non-OK outcomes must not carry references"
            )
        if self.outcome is MemoryReferenceReadOutcome.OK:
            if self.evaluated_scope is None:
                raise MemoryReferenceReadScopeError(
                    "OK outcome requires authoritative evaluated_scope"
                )
        elif self.evaluated_scope is not None:
            raise MemoryReferenceReadScopeError(
                "evaluated_scope must be omitted when outcome is not OK"
            )


def validate_memory_reference_read_request(
    identity: RequestIdentity,
    request: MemoryReferenceReadRequest,
) -> MemoryReferenceReadOutcome | None:
    """Return a failure outcome when the request is invalid for the identity; else None."""
    tenant = (identity.tenant_id or "").strip()
    if not tenant:
        return MemoryReferenceReadOutcome.INVALID_REQUEST
    if tenant != request.scope.tenant_id:
        return MemoryReferenceReadOutcome.SCOPE_REJECTED
    return None


def memory_reference_read_scope_to_entity_scope(
    scope: MemoryReferenceReadScope,
) -> EntityMemoryScope:
    """Bridge to entity/temporal projection scope (same tenant/workspace qualifiers)."""
    from intergrax.memory.contracts.entity_temporal_memory import EntityMemoryScope

    return EntityMemoryScope(
        tenant_id=scope.tenant_id,
        user_id=scope.user_id,
        workspace_id=scope.workspace_id,
    )


@runtime_checkable
class MemoryReferenceReadPort(Protocol):
    """Provider-neutral, reference-first Memory read capability."""

    async def read_references(
        self,
        identity: RequestIdentity,
        request: MemoryReferenceReadRequest,
    ) -> MemoryReferenceReadResult: ...
