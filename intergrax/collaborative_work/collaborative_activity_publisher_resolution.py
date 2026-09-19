# © Artur Czarnecki. All rights reserved.

"""Default MP-6C-C1 publisher authority resolver (configuration-backed authority source)."""

from __future__ import annotations

from intergrax.contracts.collaborative_activity_ingestion import (
    CollaborativeActivityPublisherContext,
    CollaborativeActivityPublisherKind,
)
from intergrax.contracts.collaborative_activity_publisher_authority import (
    CollaborativeActivityPluginPublisherRegistration,
    CollaborativeActivityPublisherAuthoritySource,
    CollaborativeActivityPublisherContextResolver,
    CollaborativeActivityPublisherResolutionError,
    VerifiedCollaborativeActivityPublisherIdentity,
)

_RESERVED_NAMESPACES = frozenset({"intergrax", "platform"})


class MappingCollaborativeActivityPublisherAuthoritySource:
    """Immutable authority facts from composition-root registry/configuration snapshot."""

    def __init__(
        self,
        *,
        plugin_registrations: tuple[CollaborativeActivityPluginPublisherRegistration, ...] = (),
        platform_workspace_grants: dict[tuple[str, str], tuple[str, ...]] | None = None,
    ) -> None:
        self._plugins: dict[tuple[str, str], CollaborativeActivityPluginPublisherRegistration] = {}
        for registration in plugin_registrations:
            key = (
                registration.tenant_id.strip(),
                registration.producer_principal_id.strip(),
            )
            if key in self._plugins:
                raise ValueError("duplicate plugin publisher registration")
            self._plugins[key] = registration
        self._platform_workspace_grants = platform_workspace_grants or {}

    def registered_plugin_namespace(
        self,
        identity: VerifiedCollaborativeActivityPublisherIdentity,
    ) -> str | None:
        key = (identity.tenant_id.strip(), identity.producer_principal_id.strip())
        registration = self._plugins.get(key)
        if registration is None:
            return None
        return registration.owned_namespace.strip().lower()

    def allowed_workspace_ids(
        self,
        identity: VerifiedCollaborativeActivityPublisherIdentity,
        *,
        publisher_is_platform: bool,
    ) -> tuple[str, ...]:
        key = (identity.tenant_id.strip(), identity.producer_principal_id.strip())
        registration = self._plugins.get(key)
        if registration is not None:
            return registration.allowed_workspace_ids
        if publisher_is_platform:
            return self._platform_workspace_grants.get(key, ())
        raise CollaborativeActivityPublisherResolutionError(
            "workspace authority unavailable for unresolved publisher principal",
        )


class DefaultCollaborativeActivityPublisherContextResolver:
    """Platform default binder — plugin namespace and kind from authority source only."""

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

        plugin_namespace = self._authority_source.registered_plugin_namespace(identity)
        if plugin_namespace is not None:
            owned = plugin_namespace.strip().lower()
            if owned in _RESERVED_NAMESPACES:
                raise CollaborativeActivityPublisherResolutionError(
                    "registered plugin namespace cannot be reserved",
                )
            workspaces = self._authority_source.allowed_workspace_ids(
                identity,
                publisher_is_platform=False,
            )
            return CollaborativeActivityPublisherContext(
                tenant_id=identity.tenant_id,
                producer_principal_id=identity.producer_principal_id,
                kind=CollaborativeActivityPublisherKind.PLUGIN,
                owned_namespace=owned,
                allowed_workspace_ids=workspaces,
            )

        workspaces = self._authority_source.allowed_workspace_ids(
            identity,
            publisher_is_platform=True,
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
