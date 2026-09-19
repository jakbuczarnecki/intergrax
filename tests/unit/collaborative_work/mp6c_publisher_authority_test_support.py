# © Artur Czarnecki. All rights reserved.

"""Shared MP-6C publisher authority fixtures for unit gates (test-only)."""

from __future__ import annotations

from intergrax.collaborative_work.collaborative_activity_publisher_resolution import (
    DefaultCollaborativeActivityPublisherContextResolver,
    MappingCollaborativeActivityPublisherAuthoritySource,
)
from intergrax.contracts.collaborative_activity_ingestion import CollaborativeActivityPublisherKind
from intergrax.contracts.collaborative_activity_publisher_authority import (
    CollaborativeActivityPublisherRegistration,
    CollaborativeActivityWorkspaceAuthority,
    restricted_collaborative_activity_workspace_authority,
    tenant_wide_collaborative_activity_workspace_authority,
)


def platform_publisher_registration(
    tenant_id: str,
    producer_principal_id: str,
    *,
    workspace_authority: CollaborativeActivityWorkspaceAuthority | None = None,
) -> CollaborativeActivityPublisherRegistration:
    return CollaborativeActivityPublisherRegistration(
        tenant_id=tenant_id,
        producer_principal_id=producer_principal_id,
        publisher_kind=CollaborativeActivityPublisherKind.PLATFORM,
        workspace_authority=workspace_authority
        or tenant_wide_collaborative_activity_workspace_authority(),
    )


def plugin_publisher_registration(
    tenant_id: str,
    producer_principal_id: str,
    owned_namespace: str,
    *,
    workspace_authority: CollaborativeActivityWorkspaceAuthority | None = None,
) -> CollaborativeActivityPublisherRegistration:
    return CollaborativeActivityPublisherRegistration(
        tenant_id=tenant_id,
        producer_principal_id=producer_principal_id,
        publisher_kind=CollaborativeActivityPublisherKind.PLUGIN,
        owned_namespace=owned_namespace,
        workspace_authority=workspace_authority
        or tenant_wide_collaborative_activity_workspace_authority(),
    )


_BASELINE_PLATFORM_REGISTRATIONS: tuple[CollaborativeActivityPublisherRegistration, ...] = (
    platform_publisher_registration("tenant-a", "platform-producer-1"),
    platform_publisher_registration("tenant-a", "platform-producer"),
    platform_publisher_registration("tenant-a", "platform-service-caller"),
    platform_publisher_registration("tenant-a", "producer-a"),
    platform_publisher_registration("tenant-a", "custom-platform"),
    platform_publisher_registration("tenant-b", "platform-producer-1"),
    platform_publisher_registration("tenant-b", "producer-b"),
)


def mapping_authority_source(
    *registrations: CollaborativeActivityPublisherRegistration,
) -> MappingCollaborativeActivityPublisherAuthoritySource:
    return MappingCollaborativeActivityPublisherAuthoritySource(
        registrations=registrations,
    )


def default_ingestion_resolver(
    *extra_registrations: CollaborativeActivityPublisherRegistration,
) -> DefaultCollaborativeActivityPublisherContextResolver:
    return DefaultCollaborativeActivityPublisherContextResolver(
        MappingCollaborativeActivityPublisherAuthoritySource(
            registrations=_BASELINE_PLATFORM_REGISTRATIONS + tuple(extra_registrations),
        ),
    )


def publisher_context_resolver(
    *registrations: CollaborativeActivityPublisherRegistration,
) -> DefaultCollaborativeActivityPublisherContextResolver:
    return DefaultCollaborativeActivityPublisherContextResolver(
        mapping_authority_source(*registrations),
    )
