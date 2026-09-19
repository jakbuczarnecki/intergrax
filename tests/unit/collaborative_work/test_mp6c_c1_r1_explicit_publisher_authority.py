# © Artur Czarnecki. All rights reserved.

"""MP-6C-C1-R1 — explicit publisher authority registration and fail-closed workspace grants."""

from __future__ import annotations

import pytest

from intergrax.collaborative_work.collaborative_activity_composition import (
    build_collaborative_activity_ingestion_service,
)
from intergrax.collaborative_work.collaborative_activity_publisher_resolution import (
    DefaultCollaborativeActivityPublisherContextResolver,
    MappingCollaborativeActivityPublisherAuthoritySource,
)
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.contracts.collaborative_activity_ingestion import CollaborativeActivityPublisherKind
from intergrax.contracts.collaborative_activity_publisher_authority import (
    CollaborativeActivityPublisherRegistration,
    CollaborativeActivityPublisherResolutionError,
    restricted_collaborative_activity_workspace_authority,
    tenant_wide_collaborative_activity_workspace_authority,
    verified_collaborative_activity_publisher_identity_from_request_identity,
)
from tests.unit.collaborative_work.mp6c_publisher_authority_test_support import (
    mapping_authority_source,
    platform_publisher_registration,
    plugin_publisher_registration,
    publisher_context_resolver,
)

pytestmark = pytest.mark.unit


def _verified_service(tenant: str, principal: str) -> object:
    return verified_collaborative_activity_publisher_identity_from_request_identity(
        RequestIdentity(
            tenant_id=tenant,
            auth_subject=principal,
            principal_type=PrincipalType.SERVICE,
        ),
    )


def _verified_org_system(tenant: str, principal: str) -> object:
    return verified_collaborative_activity_publisher_identity_from_request_identity(
        RequestIdentity(
            tenant_id=tenant,
            auth_subject=principal,
            principal_type=PrincipalType.ORG_SYSTEM,
        ),
    )


def test_mp6c_c1_r1_empty_authority_source_denies_all() -> None:
    resolver = DefaultCollaborativeActivityPublisherContextResolver(mapping_authority_source())
    with pytest.raises(CollaborativeActivityPublisherResolutionError):
        resolver.resolve(_verified_service("tenant-a", "any-service"))


def test_mp6c_c1_r1_unknown_service_not_platform() -> None:
    resolver = publisher_context_resolver()
    with pytest.raises(CollaborativeActivityPublisherResolutionError):
        resolver.resolve(_verified_service("tenant-a", "unregistered-service"))


def test_mp6c_c1_r1_unknown_org_system_not_platform() -> None:
    resolver = publisher_context_resolver()
    with pytest.raises(CollaborativeActivityPublisherResolutionError):
        resolver.resolve(_verified_org_system("tenant-a", "unregistered-org-system"))


def test_mp6c_c1_r1_cross_tenant_registration_miss_denies() -> None:
    resolver = publisher_context_resolver(
        plugin_publisher_registration("tenant-a", "plugin-a", "vendor.a"),
    )
    with pytest.raises(CollaborativeActivityPublisherResolutionError):
        resolver.resolve(_verified_service("tenant-b", "plugin-a"))


def test_mp6c_c1_r1_explicit_platform_registration() -> None:
    resolver = publisher_context_resolver(
        platform_publisher_registration("tenant-a", "platform-service"),
    )
    ctx = resolver.resolve(_verified_service("tenant-a", "platform-service"))
    assert ctx.kind is CollaborativeActivityPublisherKind.PLATFORM
    assert ctx.owned_namespace is None


def test_mp6c_c1_r1_explicit_plugin_registration_namespace() -> None:
    resolver = publisher_context_resolver(
        plugin_publisher_registration("tenant-a", "plugin-a", "vendor.a"),
    )
    ctx = resolver.resolve(_verified_service("tenant-a", "plugin-a"))
    assert ctx.kind is CollaborativeActivityPublisherKind.PLUGIN
    assert ctx.owned_namespace == "vendor.a"


def test_mp6c_c1_r1_duplicate_platform_and_plugin_registration_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        MappingCollaborativeActivityPublisherAuthoritySource(
            registrations=(
                platform_publisher_registration("tenant-a", "dup-principal"),
                plugin_publisher_registration("tenant-a", "dup-principal", "vendor.a"),
            ),
        )


def test_mp6c_c1_r1_reserved_plugin_namespace_rejected_at_registration() -> None:
    with pytest.raises(ValueError, match="reserved"):
        plugin_publisher_registration("tenant-a", "bad-plugin", "platform")


def test_mp6c_c1_r1_explicit_tenant_wide_workspace_authority() -> None:
    resolver = publisher_context_resolver(
        platform_publisher_registration(
            "tenant-a",
            "platform-wide",
            workspace_authority=tenant_wide_collaborative_activity_workspace_authority(),
        ),
    )
    ctx = resolver.resolve(_verified_service("tenant-a", "platform-wide"))
    assert ctx.allowed_workspace_ids == ()


def test_mp6c_c1_r1_restricted_workspace_authority() -> None:
    resolver = publisher_context_resolver(
        plugin_publisher_registration(
            "tenant-a",
            "plugin-a",
            "vendor.a",
            workspace_authority=restricted_collaborative_activity_workspace_authority(
                "ws-b",
                "ws-a",
            ),
        ),
    )
    ctx = resolver.resolve(_verified_service("tenant-a", "plugin-a"))
    assert ctx.allowed_workspace_ids == ("ws-a", "ws-b")


def test_mp6c_c1_r1_unregistered_malicious_service_resolution_before_service() -> None:
    class _RecordingAppendStore:
        append_calls: list[object] = []

        def append_idempotent(self, intent: object) -> object:
            self.append_calls.append(intent)
            raise AssertionError("store must not be called")

        def get_by_idempotency_key(self, key: object) -> None:
            return None

    store = _RecordingAppendStore()
    with pytest.raises(CollaborativeActivityPublisherResolutionError):
        build_collaborative_activity_ingestion_service(
            verified_publisher_identity=_verified_service("tenant-a", "malicious-service"),
            publisher_context_resolver=publisher_context_resolver(),
            append_store=store,
        )
    assert len(store.append_calls) == 0


def test_mp6c_c1_r1_service_principal_type_does_not_imply_platform() -> None:
    identity = _verified_service("tenant-a", "service-only")
    assert identity.producer_principal_id == "service-only"
    resolver = publisher_context_resolver()
    with pytest.raises(CollaborativeActivityPublisherResolutionError):
        resolver.resolve(identity)


def test_mp6c_c1_r1_authority_source_none_semantics() -> None:
    source = mapping_authority_source()
    assert source.resolve_publisher_authority(_verified_service("tenant-a", "x")) is None
