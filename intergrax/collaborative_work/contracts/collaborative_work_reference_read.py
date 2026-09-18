# © Artur Czarnecki. All rights reserved.

"""Scoped reference-first Collaborative Work read port (MP-5F-B4).

Collaborative Work owns WorkItem, WorkArtifact and WorkArtifactVersion identity,
lifecycle and canonical ownership. This module exposes scoped canonical references
only — no payload hydration and no ContextView / MP-5 types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, StrEnum
from typing import Protocol, runtime_checkable

from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.collaborative_work import WorkItemState

__all__ = [
    "COLLABORATIVE_WORK_REFERENCE_READ_DEFAULT_LIMIT",
    "COLLABORATIVE_WORK_REFERENCE_READ_MAX_LIMIT",
    "CollaborativeWorkArtifactCanonicalRef",
    "CollaborativeWorkArtifactVersionCanonicalRef",
    "CollaborativeWorkCanonicalRef",
    "CollaborativeWorkItemCanonicalRef",
    "CollaborativeWorkReferenceEntityKind",
    "CollaborativeWorkReferenceReadOutcome",
    "CollaborativeWorkReferenceReadPort",
    "CollaborativeWorkReferenceReadQuery",
    "CollaborativeWorkReferenceReadRequest",
    "CollaborativeWorkReferenceReadResult",
    "CollaborativeWorkReferenceReadScope",
    "CollaborativeWorkReferenceReadScopeError",
    "CollaborativeWorkVersionSelection",
    "validate_collaborative_work_reference_read_request",
]

COLLABORATIVE_WORK_REFERENCE_READ_DEFAULT_LIMIT = 50
COLLABORATIVE_WORK_REFERENCE_READ_MAX_LIMIT = 200


class CollaborativeWorkReferenceReadOutcome(str, Enum):
    OK = "ok"
    ACCESS_DENIED = "access_denied"
    SCOPE_REJECTED = "scope_rejected"
    INVALID_REQUEST = "invalid_request"
    UNAVAILABLE = "unavailable"


class CollaborativeWorkReferenceReadScopeError(ValueError):
    """Request scope or fields violate Collaborative Work reference-read invariants."""


class CollaborativeWorkReferenceEntityKind(StrEnum):
    WORK_ITEM = "work_item"
    WORK_ARTIFACT = "work_artifact"
    WORK_ARTIFACT_VERSION = "work_artifact_version"


class CollaborativeWorkVersionSelection(StrEnum):
    """Explicit version projection — default is current aggregate pointer only."""

    CURRENT_ONLY = "current_only"
    INCLUDE_HISTORICAL = "include_historical"


@dataclass(frozen=True, slots=True)
class CollaborativeWorkReferenceReadScope:
    """CW-owned least-context boundary for reference enumeration."""

    tenant_id: str
    workspace_id: str
    work_item_id: str | None = None
    work_artifact_id: str | None = None
    work_artifact_version_id: str | None = None

    def __post_init__(self) -> None:
        tenant = (self.tenant_id or "").strip()
        workspace = (self.workspace_id or "").strip()
        if not tenant:
            raise CollaborativeWorkReferenceReadScopeError("tenant_id must be non-empty")
        if not workspace:
            raise CollaborativeWorkReferenceReadScopeError("workspace_id must be non-empty")
        work_item = _strip_optional(self.work_item_id)
        artifact = _strip_optional(self.work_artifact_id)
        version = _strip_optional(self.work_artifact_version_id)
        object.__setattr__(self, "tenant_id", tenant)
        object.__setattr__(self, "workspace_id", workspace)
        object.__setattr__(self, "work_item_id", work_item)
        object.__setattr__(self, "work_artifact_id", artifact)
        object.__setattr__(self, "work_artifact_version_id", version)


@dataclass(frozen=True, slots=True)
class CollaborativeWorkReferenceReadQuery:
    entity_kinds: frozenset[CollaborativeWorkReferenceEntityKind]
    limit: int = COLLABORATIVE_WORK_REFERENCE_READ_DEFAULT_LIMIT
    version_selection: CollaborativeWorkVersionSelection = (
        CollaborativeWorkVersionSelection.CURRENT_ONLY
    )

    def __post_init__(self) -> None:
        kinds = frozenset(self.entity_kinds)
        if not kinds:
            raise CollaborativeWorkReferenceReadScopeError(
                "entity_kinds must be non-empty"
            )
        for kind in kinds:
            if not isinstance(kind, CollaborativeWorkReferenceEntityKind):
                raise CollaborativeWorkReferenceReadScopeError(
                    "entity_kinds must contain CollaborativeWorkReferenceEntityKind values"
                )
        if self.limit < 1:
            raise CollaborativeWorkReferenceReadScopeError("limit must be >= 1")
        if self.limit > COLLABORATIVE_WORK_REFERENCE_READ_MAX_LIMIT:
            raise CollaborativeWorkReferenceReadScopeError(
                f"limit must be <= {COLLABORATIVE_WORK_REFERENCE_READ_MAX_LIMIT}"
            )
        if not isinstance(self.version_selection, CollaborativeWorkVersionSelection):
            raise CollaborativeWorkReferenceReadScopeError(
                "version_selection must be CollaborativeWorkVersionSelection"
            )
        object.__setattr__(self, "entity_kinds", kinds)


@dataclass(frozen=True, slots=True)
class CollaborativeWorkReferenceReadRequest:
    scope: CollaborativeWorkReferenceReadScope
    query: CollaborativeWorkReferenceReadQuery = field(
        default_factory=lambda: CollaborativeWorkReferenceReadQuery(
            entity_kinds=frozenset({CollaborativeWorkReferenceEntityKind.WORK_ITEM}),
        )
    )


@dataclass(frozen=True, slots=True)
class CollaborativeWorkItemCanonicalRef:
    tenant_id: str
    workspace_id: str
    work_item_id: str
    state: WorkItemState

    def __post_init__(self) -> None:
        tenant = (self.tenant_id or "").strip()
        workspace = (self.workspace_id or "").strip()
        work_item = (self.work_item_id or "").strip()
        if not tenant:
            raise CollaborativeWorkReferenceReadScopeError("tenant_id must be non-empty")
        if not workspace:
            raise CollaborativeWorkReferenceReadScopeError("workspace_id must be non-empty")
        if not work_item:
            raise CollaborativeWorkReferenceReadScopeError("work_item_id must be non-empty")
        if not isinstance(self.state, WorkItemState):
            raise CollaborativeWorkReferenceReadScopeError("state must be WorkItemState")
        object.__setattr__(self, "tenant_id", tenant)
        object.__setattr__(self, "workspace_id", workspace)
        object.__setattr__(self, "work_item_id", work_item)


@dataclass(frozen=True, slots=True)
class CollaborativeWorkArtifactCanonicalRef:
    tenant_id: str
    workspace_id: str
    work_item_id: str
    work_artifact_id: str
    current_version_id: str

    def __post_init__(self) -> None:
        tenant = (self.tenant_id or "").strip()
        workspace = (self.workspace_id or "").strip()
        work_item = (self.work_item_id or "").strip()
        artifact = (self.work_artifact_id or "").strip()
        current = (self.current_version_id or "").strip()
        if not tenant:
            raise CollaborativeWorkReferenceReadScopeError("tenant_id must be non-empty")
        if not workspace:
            raise CollaborativeWorkReferenceReadScopeError("workspace_id must be non-empty")
        if not work_item:
            raise CollaborativeWorkReferenceReadScopeError("work_item_id must be non-empty")
        if not artifact:
            raise CollaborativeWorkReferenceReadScopeError("work_artifact_id must be non-empty")
        if not current:
            raise CollaborativeWorkReferenceReadScopeError(
                "current_version_id must be non-empty"
            )
        object.__setattr__(self, "tenant_id", tenant)
        object.__setattr__(self, "workspace_id", workspace)
        object.__setattr__(self, "work_item_id", work_item)
        object.__setattr__(self, "work_artifact_id", artifact)
        object.__setattr__(self, "current_version_id", current)


@dataclass(frozen=True, slots=True)
class CollaborativeWorkArtifactVersionCanonicalRef:
    tenant_id: str
    workspace_id: str
    work_item_id: str
    work_artifact_id: str
    work_artifact_version_id: str

    def __post_init__(self) -> None:
        tenant = (self.tenant_id or "").strip()
        workspace = (self.workspace_id or "").strip()
        work_item = (self.work_item_id or "").strip()
        artifact = (self.work_artifact_id or "").strip()
        version = (self.work_artifact_version_id or "").strip()
        if not tenant:
            raise CollaborativeWorkReferenceReadScopeError("tenant_id must be non-empty")
        if not workspace:
            raise CollaborativeWorkReferenceReadScopeError("workspace_id must be non-empty")
        if not work_item:
            raise CollaborativeWorkReferenceReadScopeError("work_item_id must be non-empty")
        if not artifact:
            raise CollaborativeWorkReferenceReadScopeError("work_artifact_id must be non-empty")
        if not version:
            raise CollaborativeWorkReferenceReadScopeError(
                "work_artifact_version_id must be non-empty"
            )
        object.__setattr__(self, "tenant_id", tenant)
        object.__setattr__(self, "workspace_id", workspace)
        object.__setattr__(self, "work_item_id", work_item)
        object.__setattr__(self, "work_artifact_id", artifact)
        object.__setattr__(self, "work_artifact_version_id", version)


CollaborativeWorkCanonicalRef = (
    CollaborativeWorkItemCanonicalRef
    | CollaborativeWorkArtifactCanonicalRef
    | CollaborativeWorkArtifactVersionCanonicalRef
)


@dataclass(frozen=True, slots=True)
class CollaborativeWorkReferenceReadResult:
    outcome: CollaborativeWorkReferenceReadOutcome
    references: tuple[CollaborativeWorkCanonicalRef, ...] = ()
    reason: str = ""

    def __post_init__(self) -> None:
        if self.outcome is not CollaborativeWorkReferenceReadOutcome.OK and self.references:
            raise CollaborativeWorkReferenceReadScopeError(
                "non-OK outcomes must not carry references"
            )


def validate_collaborative_work_reference_read_request(
    identity: RequestIdentity,
    request: CollaborativeWorkReferenceReadRequest,
) -> CollaborativeWorkReferenceReadOutcome | None:
    """Return a failure outcome when the request is invalid for the identity; else None."""
    tenant = (identity.tenant_id or "").strip()
    if not tenant:
        return CollaborativeWorkReferenceReadOutcome.INVALID_REQUEST
    if tenant != request.scope.tenant_id:
        return CollaborativeWorkReferenceReadOutcome.SCOPE_REJECTED
    return None


def _strip_optional(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized if normalized else None


@runtime_checkable
class CollaborativeWorkReferenceReadPort(Protocol):
    """Provider-neutral, reference-first Collaborative Work read capability."""

    def read_references(
        self,
        identity: RequestIdentity,
        request: CollaborativeWorkReferenceReadRequest,
    ) -> CollaborativeWorkReferenceReadResult: ...
