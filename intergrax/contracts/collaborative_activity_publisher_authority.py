# © Artur Czarnecki. All rights reserved.

"""Trusted publisher identity and authority binding for MP-6C ingestion (MP-6C-C1).

``VerifiedCollaborativeActivityPublisherIdentity`` is produced only from canonical
``RequestIdentity`` at the authenticated composition boundary. Publisher kind,
namespace ownership, and workspace scope are resolved by an injected
``CollaborativeActivityPublisherContextResolver`` — never from publication fields
or raw caller strings.

Authority facts come from explicit registration (config snapshot). ``None`` from
``resolve_publisher_authority`` means no authority — never implicit PLATFORM.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from intergrax.contracts.agent_run import RequestIdentity, canonical_principal_id_from_request_identity
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.contracts.collaborative_activity_ingestion import (
    CollaborativeActivityPublisherContext,
    CollaborativeActivityPublisherKind,
)

_RESERVED_PLUGIN_NAMESPACES = frozenset({"intergrax", "platform"})


class CollaborativeActivityPublisherResolutionError(RuntimeError):
    """Fail-closed publisher authority resolution — no ingestion or store access."""


@dataclass(frozen=True, slots=True)
class VerifiedCollaborativeActivityPublisherIdentity:
    """Upstream-authenticated publisher principal — not self-declared publication claims."""

    tenant_id: str
    producer_principal_id: str

    def __post_init__(self) -> None:
        if not self.tenant_id.strip():
            raise ValueError("tenant_id required")
        if not self.producer_principal_id.strip():
            raise ValueError("producer_principal_id required")


def verified_collaborative_activity_publisher_identity_from_request_identity(
    identity: RequestIdentity,
) -> VerifiedCollaborativeActivityPublisherIdentity:
    """Map canonical run identity to verified publisher identity (fail-closed for users)."""
    if type(identity) is not RequestIdentity:
        raise TypeError("identity must be RequestIdentity")
    if identity.principal_type is PrincipalType.USER:
        raise CollaborativeActivityPublisherResolutionError(
            "human user principals cannot act as collaborative activity publishers",
        )
    tenant_id = identity.tenant_id.strip()
    producer_principal_id = canonical_principal_id_from_request_identity(identity)
    return VerifiedCollaborativeActivityPublisherIdentity(
        tenant_id=tenant_id,
        producer_principal_id=producer_principal_id,
    )


class CollaborativeActivityWorkspaceAuthorityMode(StrEnum):
    """Explicit workspace grant — missing registration denies; never inferred tenant-wide."""

    TENANT_WIDE = "tenant_wide"
    RESTRICTED = "restricted"


@dataclass(frozen=True, slots=True)
class CollaborativeActivityWorkspaceAuthority:
    """Trusted workspace scope from publisher registration (composition-root config only)."""

    mode: CollaborativeActivityWorkspaceAuthorityMode
    workspace_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        normalized = _normalize_workspace_ids(self.workspace_ids)
        object.__setattr__(self, "workspace_ids", normalized)
        if self.mode is CollaborativeActivityWorkspaceAuthorityMode.TENANT_WIDE:
            if normalized:
                raise ValueError("TENANT_WIDE workspace authority must have empty workspace_ids")
        elif self.mode is CollaborativeActivityWorkspaceAuthorityMode.RESTRICTED:
            if not normalized:
                raise ValueError("RESTRICTED workspace authority requires workspace_ids")
        else:
            raise ValueError("unknown workspace authority mode")


def tenant_wide_collaborative_activity_workspace_authority() -> CollaborativeActivityWorkspaceAuthority:
    """Explicit tenant-wide workspace grant."""
    return CollaborativeActivityWorkspaceAuthority(
        mode=CollaborativeActivityWorkspaceAuthorityMode.TENANT_WIDE,
    )


def restricted_collaborative_activity_workspace_authority(
    *workspace_ids: str,
) -> CollaborativeActivityWorkspaceAuthority:
    """Explicit workspace allow-list."""
    return CollaborativeActivityWorkspaceAuthority(
        mode=CollaborativeActivityWorkspaceAuthorityMode.RESTRICTED,
        workspace_ids=workspace_ids,
    )


@dataclass(frozen=True, slots=True)
class CollaborativeActivityPublisherRegistration:
    """One explicit authority record per (tenant_id, producer_principal_id)."""

    tenant_id: str
    producer_principal_id: str
    publisher_kind: CollaborativeActivityPublisherKind
    workspace_authority: CollaborativeActivityWorkspaceAuthority
    owned_namespace: str | None = None

    def __post_init__(self) -> None:
        tenant = self.tenant_id.strip()
        principal = self.producer_principal_id.strip()
        if not tenant:
            raise ValueError("tenant_id required")
        if not principal:
            raise ValueError("producer_principal_id required")
        object.__setattr__(self, "tenant_id", tenant)
        object.__setattr__(self, "producer_principal_id", principal)

        if self.publisher_kind is CollaborativeActivityPublisherKind.PLUGIN:
            if self.owned_namespace is None:
                raise ValueError("owned_namespace required for PLUGIN registration")
            owned = self.owned_namespace.strip().lower()
            if not owned:
                raise ValueError("owned_namespace required")
            if owned in _RESERVED_PLUGIN_NAMESPACES:
                raise ValueError("PLUGIN owned_namespace cannot be reserved")
            object.__setattr__(self, "owned_namespace", owned)
        else:
            if self.owned_namespace is not None:
                raise ValueError("owned_namespace must be omitted for PLATFORM registration")


@dataclass(frozen=True, slots=True)
class CollaborativeActivityPublisherAuthority:
    """Resolved publisher authority fact returned by ``CollaborativeActivityPublisherAuthoritySource``."""

    publisher_kind: CollaborativeActivityPublisherKind
    workspace_authority: CollaborativeActivityWorkspaceAuthority
    owned_namespace: str | None = None

    def __post_init__(self) -> None:
        if self.publisher_kind is CollaborativeActivityPublisherKind.PLUGIN:
            if self.owned_namespace is None:
                raise ValueError("PLUGIN authority requires owned_namespace")
            owned = self.owned_namespace.strip().lower()
            if owned in _RESERVED_PLUGIN_NAMESPACES:
                raise ValueError("PLUGIN owned_namespace cannot be reserved")
            object.__setattr__(self, "owned_namespace", owned)
        elif self.owned_namespace is not None:
            raise ValueError("PLATFORM authority must not carry owned_namespace")


def _normalize_workspace_ids(workspace_ids: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    canonical: list[str] = []
    for raw in workspace_ids:
        normalized = raw.strip()
        if not normalized:
            raise ValueError("workspace_id must be non-empty")
        if normalized in seen:
            continue
        seen.add(normalized)
        canonical.append(normalized)
    return tuple(sorted(canonical))


def allowed_workspace_ids_for_publisher_context(
    workspace_authority: CollaborativeActivityWorkspaceAuthority,
) -> tuple[str, ...]:
    """Map trusted workspace authority to ingestion ``allowed_workspace_ids`` (empty = tenant-wide)."""
    if workspace_authority.mode is CollaborativeActivityWorkspaceAuthorityMode.TENANT_WIDE:
        return ()
    return workspace_authority.workspace_ids


@runtime_checkable
class CollaborativeActivityPublisherAuthoritySource(Protocol):
    """Authoritative publisher facts (registry snapshot / platform config) — not caller input.

    ``resolve_publisher_authority`` returns ``None`` when the principal has no explicit
    registration. ``None`` never means PLATFORM.
    """

    def resolve_publisher_authority(
        self,
        identity: VerifiedCollaborativeActivityPublisherIdentity,
    ) -> CollaborativeActivityPublisherAuthority | None:
        """Return explicit PLATFORM/PLUGIN authority or ``None`` when unregistered."""
        ...


@runtime_checkable
class CollaborativeActivityPublisherContextResolver(Protocol):
    """Derives trusted ``CollaborativeActivityPublisherContext`` from verified identity."""

    def resolve(
        self,
        identity: VerifiedCollaborativeActivityPublisherIdentity,
    ) -> CollaborativeActivityPublisherContext:
        """Fail closed when identity cannot be bound to publisher authority."""
        ...
