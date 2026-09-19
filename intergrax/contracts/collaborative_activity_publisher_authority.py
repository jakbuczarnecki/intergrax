# © Artur Czarnecki. All rights reserved.

"""Trusted publisher identity and authority binding for MP-6C ingestion (MP-6C-C1).

``VerifiedCollaborativeActivityPublisherIdentity`` is produced only from canonical
``RequestIdentity`` at the authenticated composition boundary. Publisher kind,
namespace ownership, and workspace scope are resolved by an injected
``CollaborativeActivityPublisherContextResolver`` — never from publication fields
or raw caller strings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from intergrax.contracts.agent_run import RequestIdentity, canonical_principal_id_from_request_identity
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.contracts.collaborative_activity_ingestion import (
    CollaborativeActivityPublisherContext,
)


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


@dataclass(frozen=True, slots=True)
class CollaborativeActivityPluginPublisherRegistration:
    """Authoritative plugin producer registration — composition-root configuration only."""

    tenant_id: str
    producer_principal_id: str
    owned_namespace: str
    allowed_workspace_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.tenant_id.strip():
            raise ValueError("tenant_id required")
        if not self.producer_principal_id.strip():
            raise ValueError("producer_principal_id required")
        owned = self.owned_namespace.strip().lower()
        if not owned:
            raise ValueError("owned_namespace required")


@runtime_checkable
class CollaborativeActivityPublisherAuthoritySource(Protocol):
    """Authoritative publisher facts (registry snapshot / platform config) — not caller input."""

    def registered_plugin_namespace(
        self,
        identity: VerifiedCollaborativeActivityPublisherIdentity,
    ) -> str | None:
        """Return owned namespace when principal is a registered plugin producer; else None."""
        ...

    def allowed_workspace_ids(
        self,
        identity: VerifiedCollaborativeActivityPublisherIdentity,
        *,
        publisher_is_platform: bool,
    ) -> tuple[str, ...]:
        """Empty tuple means explicit tenant-wide workspace authority from trusted grant."""
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
