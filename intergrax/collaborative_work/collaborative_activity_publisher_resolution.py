# © Artur Czarnecki. All rights reserved.

"""Default MP-6C-C1 publisher authority resolver (configuration-backed authority source)."""

from __future__ import annotations

from intergrax.contracts.collaborative_activity_ingestion import (
    CollaborativeActivityPublisherContext,
    CollaborativeActivityPublisherKind,
)
from intergrax.contracts.collaborative_activity_publisher_authority import (
    CollaborativeActivityPublisherAuthority,
    CollaborativeActivityPublisherAuthoritySource,
    CollaborativeActivityPublisherContextResolver,
    CollaborativeActivityPublisherRegistration,
    CollaborativeActivityPublisherResolutionError,
    VerifiedCollaborativeActivityPublisherIdentity,
    allowed_workspace_ids_for_publisher_context,
)


class MappingCollaborativeActivityPublisherAuthoritySource:
    """Immutable authority facts from composition-root registry/configuration snapshot."""

    def __init__(
        self,
        *,
        registrations: tuple[CollaborativeActivityPublisherRegistration, ...] = (),
    ) -> None:
        self._registrations: dict[
            tuple[str, str],
            CollaborativeActivityPublisherRegistration,
        ] = {}
        for registration in registrations:
            key = (
                registration.tenant_id.strip(),
                registration.producer_principal_id.strip(),
            )
            if key in self._registrations:
                raise ValueError("duplicate publisher authority registration")
            self._registrations[key] = registration

    def resolve_publisher_authority(
        self,
        identity: VerifiedCollaborativeActivityPublisherIdentity,
    ) -> CollaborativeActivityPublisherAuthority | None:
        key = (identity.tenant_id.strip(), identity.producer_principal_id.strip())
        registration = self._registrations.get(key)
        if registration is None:
            return None
        return CollaborativeActivityPublisherAuthority(
            publisher_kind=registration.publisher_kind,
            workspace_authority=registration.workspace_authority,
            owned_namespace=registration.owned_namespace,
        )


class DefaultCollaborativeActivityPublisherContextResolver:
    """Platform default binder — publisher kind and namespace from authority source only."""

    def __init__(
        self,
        authority_source: CollaborativeActivityPublisherAuthoritySource,
    ) -> None:
        self._authority_source = authority_source

    def resolve(
        self,
        identity: VerifiedCollaborativeActivityPublisherIdentity,
    ) -> CollaborativeActivityPublisherContext:
        if type(identity) is not VerifiedCollaborativeActivityPublisherIdentity:
            raise TypeError("identity must be VerifiedCollaborativeActivityPublisherIdentity")

        authority = self._authority_source.resolve_publisher_authority(identity)
        if authority is None:
            raise CollaborativeActivityPublisherResolutionError(
                "publisher principal has no explicit authority registration",
            )

        workspaces = allowed_workspace_ids_for_publisher_context(authority.workspace_authority)
        if authority.publisher_kind is CollaborativeActivityPublisherKind.PLUGIN:
            return CollaborativeActivityPublisherContext(
                tenant_id=identity.tenant_id,
                producer_principal_id=identity.producer_principal_id,
                kind=CollaborativeActivityPublisherKind.PLUGIN,
                owned_namespace=authority.owned_namespace,
                allowed_workspace_ids=workspaces,
            )

        return CollaborativeActivityPublisherContext(
            tenant_id=identity.tenant_id,
            producer_principal_id=identity.producer_principal_id,
            kind=CollaborativeActivityPublisherKind.PLATFORM,
            allowed_workspace_ids=workspaces,
        )


__all__ = [
    "DefaultCollaborativeActivityPublisherContextResolver",
    "MappingCollaborativeActivityPublisherAuthoritySource",
]
