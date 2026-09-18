# © Artur Czarnecki. All rights reserved.

"""Scoped reference-first UCL lifecycle read port (MP-5F-B3).

UCL owns artifact lifecycle, revisions, validity and durability. This module
exposes canonical optimization-artifact references only — no payload hydration
and no ContextView / MP-5 types.

Artifact identity is ``artifact_id`` (immutable stored record). Compatibility
and reuse identity is ``artifact_lookup_key_hash``. ``context_scope_id`` is the
canonical UCL lifecycle scope on ``ArtifactLookupKey``. It is not workspace_id;
workspace-scoped reads enforce canonical persisted ``UclArtifactOwnership`` via
the scoped reference catalog (tenant, workspace, and context_scope are independent).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, StrEnum
from typing import Protocol, runtime_checkable

from intergrax.contracts.agent_run import RequestIdentity

__all__ = [
    "UCL_REFERENCE_READ_DEFAULT_LIMIT",
    "UCL_REFERENCE_READ_MAX_LIMIT",
    "UclOptimizationArtifactCanonicalRef",
    "UclReferenceLifecycleSelection",
    "UclReferenceReadOutcome",
    "UclReferenceReadPort",
    "UclReferenceReadQuery",
    "UclReferenceReadRequest",
    "UclReferenceReadResult",
    "UclReferenceReadScope",
    "UclReferenceReadScopeError",
    "UclScopedResourceRef",
    "format_ucl_artifact_locator",
    "validate_ucl_reference_read_request",
]

UCL_REFERENCE_READ_DEFAULT_LIMIT = 50
UCL_REFERENCE_READ_MAX_LIMIT = 200


class UclReferenceReadOutcome(str, Enum):
    OK = "ok"
    ACCESS_DENIED = "access_denied"
    SCOPE_REJECTED = "scope_rejected"
    INVALID_REQUEST = "invalid_request"
    UNAVAILABLE = "unavailable"


class UclReferenceLifecycleSelection(StrEnum):
    """Explicit lifecycle projection — default is active validated catalog entries only."""

    ACTIVE_VALIDATED_ONLY = "active_validated_only"
    INCLUDE_HISTORICAL = "include_historical"


class UclReferenceReadScopeError(ValueError):
    """Request scope or fields violate UCL reference-read invariants."""


@dataclass(frozen=True, slots=True)
class UclScopedResourceRef:
    """Optional neutral resource binding within a scoped UCL read request."""

    resource_kind: str
    resource_id: str

    def __post_init__(self) -> None:
        kind = (self.resource_kind or "").strip()
        ident = (self.resource_id or "").strip()
        if not kind:
            raise UclReferenceReadScopeError("resource_kind must be non-empty")
        if not ident:
            raise UclReferenceReadScopeError("resource_id must be non-empty")
        object.__setattr__(self, "resource_kind", kind)
        object.__setattr__(self, "resource_id", ident)


@dataclass(frozen=True, slots=True)
class UclReferenceReadScope:
    """UCL-owned least-context boundary for lifecycle reference enumeration.

    ``tenant_id``, ``workspace_id``, and ``context_scope_id`` are mandatory for
    workspace-scoped reference read. ``context_scope_id`` is the canonical UCL
    lifecycle scope on optimization artifacts (``ArtifactLookupKey``); it is not
    workspace authority. Equal ``workspace_id`` and ``context_scope_id`` strings
    do not imply workspace authority.
    """

    tenant_id: str
    workspace_id: str
    context_scope_id: str
    resource: UclScopedResourceRef | None = None

    def __post_init__(self) -> None:
        tenant = (self.tenant_id or "").strip()
        context_scope = (self.context_scope_id or "").strip()
        if not tenant:
            raise UclReferenceReadScopeError("tenant_id must be non-empty")
        if not context_scope:
            raise UclReferenceReadScopeError("context_scope_id must be non-empty")
        workspace = (self.workspace_id or "").strip()
        if not workspace:
            raise UclReferenceReadScopeError("workspace_id must be non-empty")
        object.__setattr__(self, "tenant_id", tenant)
        object.__setattr__(self, "context_scope_id", context_scope)
        object.__setattr__(self, "workspace_id", workspace)


@dataclass(frozen=True, slots=True)
class UclReferenceReadQuery:
    """Bounded, typed query parameters — no open-ended catalog overrides."""

    limit: int = UCL_REFERENCE_READ_DEFAULT_LIMIT
    lifecycle_selection: UclReferenceLifecycleSelection = (
        UclReferenceLifecycleSelection.ACTIVE_VALIDATED_ONLY
    )

    def __post_init__(self) -> None:
        if self.limit < 1:
            raise UclReferenceReadScopeError("limit must be >= 1")
        if self.limit > UCL_REFERENCE_READ_MAX_LIMIT:
            raise UclReferenceReadScopeError(
                f"limit must be <= {UCL_REFERENCE_READ_MAX_LIMIT}"
            )
        if not isinstance(self.lifecycle_selection, UclReferenceLifecycleSelection):
            raise UclReferenceReadScopeError(
                "lifecycle_selection must be UclReferenceLifecycleSelection"
            )


@dataclass(frozen=True, slots=True)
class UclReferenceReadRequest:
    scope: UclReferenceReadScope
    query: UclReferenceReadQuery = field(default_factory=UclReferenceReadQuery)


@dataclass(frozen=True, slots=True)
class UclOptimizationArtifactCanonicalRef:
    """Canonical UCL optimization artifact identity (maps to ContextView externally).

    ``artifact_id`` is the immutable stored artifact record id. There is no
    separate revision id on optimization artifacts; supersession is modeled via
    lifecycle status and repository active-slot semantics.
    """

    tenant_id: str
    workspace_id: str
    context_scope_id: str
    artifact_id: str
    artifact_lookup_key_hash: str
    artifact_content_hash: str
    artifact_type: str
    lifecycle_status: str

    def __post_init__(self) -> None:
        tenant = (self.tenant_id or "").strip()
        workspace = (self.workspace_id or "").strip()
        scope = (self.context_scope_id or "").strip()
        artifact_id = (self.artifact_id or "").strip()
        key_hash = (self.artifact_lookup_key_hash or "").strip()
        content_hash = (self.artifact_content_hash or "").strip()
        artifact_type = (self.artifact_type or "").strip()
        lifecycle_status = (self.lifecycle_status or "").strip()
        if not tenant:
            raise UclReferenceReadScopeError("tenant_id must be non-empty")
        if not workspace:
            raise UclReferenceReadScopeError("workspace_id must be non-empty")
        if not scope:
            raise UclReferenceReadScopeError("context_scope_id must be non-empty")
        if not artifact_id:
            raise UclReferenceReadScopeError("artifact_id must be non-empty")
        if not key_hash:
            raise UclReferenceReadScopeError("artifact_lookup_key_hash must be non-empty")
        if not content_hash:
            raise UclReferenceReadScopeError("artifact_content_hash must be non-empty")
        if not artifact_type:
            raise UclReferenceReadScopeError("artifact_type must be non-empty")
        if not lifecycle_status:
            raise UclReferenceReadScopeError("lifecycle_status must be non-empty")
        object.__setattr__(self, "tenant_id", tenant)
        object.__setattr__(self, "workspace_id", workspace)
        object.__setattr__(self, "context_scope_id", scope)
        object.__setattr__(self, "artifact_id", artifact_id)
        object.__setattr__(self, "artifact_lookup_key_hash", key_hash)
        object.__setattr__(self, "artifact_content_hash", content_hash)
        object.__setattr__(self, "artifact_type", artifact_type)
        object.__setattr__(self, "lifecycle_status", lifecycle_status)


def format_ucl_artifact_locator(ref: UclOptimizationArtifactCanonicalRef) -> str:
    """Stable locator string for future ``ContextViewUclSourceRef.ucl_artifact_ref`` mapping."""
    return (
        f"ucl-opt/v1/tenant/{ref.tenant_id}/scope/{ref.context_scope_id}/"
        f"artifact/{ref.artifact_id}/lookup/{ref.artifact_lookup_key_hash}"
    )


@dataclass(frozen=True, slots=True)
class UclReferenceReadResult:
    outcome: UclReferenceReadOutcome
    references: tuple[UclOptimizationArtifactCanonicalRef, ...] = ()
    reason: str = ""

    def __post_init__(self) -> None:
        if self.outcome is not UclReferenceReadOutcome.OK and self.references:
            raise UclReferenceReadScopeError(
                "non-OK outcomes must not carry references"
            )


def validate_ucl_reference_read_request(
    identity: RequestIdentity,
    request: UclReferenceReadRequest,
) -> UclReferenceReadOutcome | None:
    """Return a failure outcome when the request is invalid for the identity; else None."""
    tenant = (identity.tenant_id or "").strip()
    if not tenant:
        return UclReferenceReadOutcome.INVALID_REQUEST
    if tenant != request.scope.tenant_id:
        return UclReferenceReadOutcome.SCOPE_REJECTED
    return None


@runtime_checkable
class UclReferenceReadPort(Protocol):
    """Provider-neutral, reference-first UCL lifecycle read capability."""

    async def read_references(
        self,
        identity: RequestIdentity,
        request: UclReferenceReadRequest,
    ) -> UclReferenceReadResult: ...
