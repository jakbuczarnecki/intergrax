# ┬ę Artur Czarnecki. All rights reserved.
# Intergrax framework ÔÇô proprietary and confidential.

"""Unit tests for provider-neutral remote resource discovery boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, timezone
from collections.abc import Callable
from typing import Any

import pytest
from pydantic import ValidationError

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.runtime.vendor_knowledge.connections import KnowledgeConnectionRegistry
from intergrax.runtime.vendor_knowledge.errors import VendorKnowledgeError, VendorKnowledgeErrorCode
from intergrax.runtime.vendor_knowledge.remote_resource_discovery import (
    RemoteResourceAvailabilityV1,
    RemoteResourceCandidatePageV1,
    RemoteResourceCandidateV1,
    RemoteResourceDescriptorV1,
    RemoteResourceDiscoveryPageV1,
    RemoteResourceDiscoveryRegistry,
    TenantRemoteResourceDiscoveryService,
)
from intergrax.runtime.vendor_knowledge.tenant_connection_capabilities import (
    CapabilityEffectV1,
    LiveCapabilityDescriptorV1,
    RepositoryTenantConnectionPort,
    TenantLiveCapabilityCatalog,
)
from intergrax.runtime.vendor_knowledge.tenant_connection_document_store import (
    DocumentStoreTenantConnectionRepository,
)
from intergrax.runtime.vendor_knowledge.tenant_connections import (
    SafeTenantConnectionV1,
    TenantConnection,
    TenantConnectionAdministrativeStatus,
    TenantConnectionInvalidState,
    TenantConnectionNotFound,
    TenantConnectionService,
)
from tests.unit.runtime.vendor_knowledge.test_tenant_connection_document_store import (
    ConditionalInMemoryDocumentStore,
)


def _utc_now(offset_seconds: int = 0) -> datetime:
    return datetime.now(UTC) + timedelta(seconds=offset_seconds)


def _connection(**overrides: Any) -> TenantConnection:
    created = _utc_now()
    payload = {
        "connection_ref": "conn-1",
        "tenant_id": "tenant-1",
        "provider_id": "ms365_graph",
        "integration_kind": IntegrationCategory.COLLABORATION_SUITE,
        "safe_display_name": "Example connection",
        "administrative_status": TenantConnectionAdministrativeStatus.ACTIVE,
        "credential_ref": "cred-1",
        "validated_secret_free_config": {"token_endpoint": "https://auth.example.test/oauth2/token"},
        "configuration_version": 1,
        "created_at": created,
        "updated_at": created,
    }
    payload.update(overrides)
    return TenantConnection(**payload)


def _descriptor(**overrides: Any) -> LiveCapabilityDescriptorV1:
    payload = {
        "capability_id": "mail.read",
        "provider_id": "ms365_graph",
        "integration_kind": IntegrationCategory.COLLABORATION_SUITE,
        "effect": CapabilityEffectV1.READ,
        "read_only": True,
        "resource_scope_required": False,
        "request_schema_ref": "schema://mail.read/request",
        "result_schema_ref": "schema://mail.read/result",
        "available": True,
    }
    payload.update(overrides)
    return LiveCapabilityDescriptorV1(**payload)


def _candidate(**overrides: Any) -> RemoteResourceCandidateV1:
    payload = {
        "remote_resource_id": "res-1",
        "resource_type": "mailbox",
        "safe_display_label": "Inbox",
        "availability": RemoteResourceAvailabilityV1.AVAILABLE,
        "supported_capability_ids": ("mail.read",),
    }
    payload.update(overrides)
    return RemoteResourceCandidateV1(**payload)


@dataclass
class _FakeIntegration:
    provider_id: str
    integration_kind: IntegrationCategory


@dataclass
class _RecordingProvider:
    provider_id: str
    integration_kind: IntegrationCategory
    source_kind: str
    pages: dict[str | None, RemoteResourceCandidatePageV1]
    calls: list[dict[str, Any]] = field(default_factory=list)
    fail_with: Exception | None = None

    async def list_remote_resources(
        self,
        *,
        integration: object,
        connection: SafeTenantConnectionV1,
        page_token: str | None,
        limit: int,
    ) -> RemoteResourceCandidatePageV1:
        self.calls.append(
            {"integration": integration, "connection": connection, "page_token": page_token, "limit": limit}
        )
        if self.fail_with is not None:
            raise self.fail_with
        return self.pages[page_token]


def _default_provider(**overrides: Any) -> _RecordingProvider:
    payload = {
        "provider_id": "ms365_graph",
        "integration_kind": IntegrationCategory.COLLABORATION_SUITE,
        "source_kind": "mailbox",
        "pages": {None: RemoteResourceCandidatePageV1(resources=(_candidate(),), snapshot_version="snap-1")},
    }
    payload.update(overrides)
    return _RecordingProvider(**payload)


@dataclass
class _MalformedProvider:
    provider_id: str = "ms365_graph"
    integration_kind: IntegrationCategory = IntegrationCategory.COLLABORATION_SUITE
    source_kind: str = "mailbox"
    return_value: Any = None

    async def list_remote_resources(
        self,
        *,
        integration: object,
        connection: SafeTenantConnectionV1,
        page_token: str | None,
        limit: int,
    ) -> RemoteResourceCandidatePageV1:
        return self.return_value  # type: ignore[return-value]


def _assert_invalid_provider_response(exc_info: pytest.ExceptionInfo[VendorKnowledgeError], *, secret: str = "") -> None:
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE
    if secret:
        assert secret not in str(exc_info.value)


def _clock_runtime_error() -> datetime:
    raise RuntimeError("internal-clock-secret")


def _stack(provider: Any = None, clock: Callable[[], datetime] | None = None) -> tuple[Any, ...]:
    store = ConditionalInMemoryDocumentStore()
    repo = DocumentStoreTenantConnectionRepository(store)
    port = RepositoryTenantConnectionPort(repo)
    catalog = TenantLiveCapabilityCatalog(connection_port=port)
    connection_registry = KnowledgeConnectionRegistry()
    discovery_registry = RemoteResourceDiscoveryRegistry()
    recording = provider or _default_provider()
    discovery_registry.register(recording)
    service = TenantRemoteResourceDiscoveryService(
        tenant_id="tenant-1",
        connection_port=port,
        capability_catalog=catalog,
        connection_registry=connection_registry,
        discovery_registry=discovery_registry,
        clock=clock,
    )
    return service, TenantConnectionService(tenant_id="tenant-1", repository=repo), repo, catalog, connection_registry, discovery_registry, recording, store


def _wire(service: TenantRemoteResourceDiscoveryService, admin: TenantConnectionService, catalog: TenantLiveCapabilityCatalog, connection_registry: KnowledgeConnectionRegistry, *, connection: TenantConnection | None = None, register_capability: bool = True) -> None:
    admin.create(connection or _connection())
    if register_capability:
        catalog.register(_descriptor())
    connection_registry.register(
        tenant_id="tenant-1",
        connection_ref="conn-1",
        provider_id="ms365_graph",
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        integration=_FakeIntegration("ms365_graph", IntegrationCategory.COLLABORATION_SUITE),
    )


@pytest.mark.unit
def test_valid_models_and_capability_sorting() -> None:
    assert _candidate(supported_capability_ids=(" beta.read ", "alpha.read", "alpha.read")).supported_capability_ids == ("alpha.read", "beta.read")
    descriptor = RemoteResourceDescriptorV1(
        connection_ref="conn-1",
        remote_resource_id="res-1",
        provider_id="ms365_graph",
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        source_kind="mailbox",
        resource_type="mailbox",
        safe_display_label="Inbox",
        availability=RemoteResourceAvailabilityV1.AVAILABLE,
        discovered_at=_utc_now(),
        snapshot_version="snap-1",
    )
    assert descriptor.safe_description == ""


@pytest.mark.unit
@pytest.mark.parametrize(
    ("factory", "kwargs"),
    [
        (RemoteResourceCandidateV1, {"remote_resource_id": "res-1", "resource_type": "mailbox", "safe_display_label": "Inbox", "availability": RemoteResourceAvailabilityV1.AVAILABLE, "unexpected": "nope"}),
        (RemoteResourceDescriptorV1, {"connection_ref": "conn-1", "remote_resource_id": "res-1", "provider_id": "ms365_graph", "integration_kind": IntegrationCategory.COLLABORATION_SUITE, "source_kind": "mailbox", "resource_type": "mailbox", "safe_display_label": "Inbox", "availability": RemoteResourceAvailabilityV1.AVAILABLE, "discovered_at": _utc_now(), "snapshot_version": "snap-1", "unexpected": "nope"}),
    ],
)
def test_unknown_fields_rejected(factory: type, kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        factory(**kwargs)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("remote_resource_id", ""),
        ("resource_type", ""),
        ("safe_display_label", ""),
        ("remote_resource_id", "a" * 257),
        ("resource_type", "a" * 65),
        ("safe_display_label", "a" * 257),
        ("safe_description", "a" * 1025),
        ("snapshot_version", ""),
    ],
)
def test_model_string_validation(field_name: str, value: str) -> None:
    kwargs = {
        "remote_resource_id": "res-1",
        "resource_type": "mailbox",
        "safe_display_label": "Inbox",
        "availability": RemoteResourceAvailabilityV1.AVAILABLE,
        field_name: value,
    }
    with pytest.raises(ValidationError):
        RemoteResourceCandidateV1(**kwargs) if field_name != "snapshot_version" else RemoteResourceDiscoveryPageV1(snapshot_version=value)


@pytest.mark.unit
@pytest.mark.parametrize(
    "unsafe",
    ["Authorization: secret", "Authorization=secret", "Bearer abc.def.ghi", "api_key=secret", "api-key: secret", "https://user:pass@example.test/path", "https://example.test/path?access_token=secret"],
)
def test_safe_text_secret_patterns_rejected(unsafe: str) -> None:
    with pytest.raises(ValidationError):
        _candidate(safe_display_label=unsafe)


@pytest.mark.unit
@pytest.mark.parametrize("token", ["", "   ", "a" * 4097])
def test_page_token_validation(token: str) -> None:
    with pytest.raises(ValidationError):
        RemoteResourceDiscoveryPageV1(snapshot_version="snap-1", next_page_token=token)


@pytest.mark.unit
def test_non_utc_discovered_at_and_snapshot_mismatch_rejected() -> None:
    with pytest.raises(ValidationError):
        RemoteResourceDescriptorV1(
            connection_ref="conn-1",
            remote_resource_id="res-1",
            provider_id="ms365_graph",
            integration_kind=IntegrationCategory.COLLABORATION_SUITE,
            source_kind="mailbox",
            resource_type="mailbox",
            safe_display_label="Inbox",
            availability=RemoteResourceAvailabilityV1.AVAILABLE,
            discovered_at=datetime(2020, 1, 1),
            snapshot_version="snap-1",
        )
    descriptor = RemoteResourceDescriptorV1(
        connection_ref="conn-1",
        remote_resource_id="res-1",
        provider_id="ms365_graph",
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        source_kind="mailbox",
        resource_type="mailbox",
        safe_display_label="Inbox",
        availability=RemoteResourceAvailabilityV1.AVAILABLE,
        discovered_at=_utc_now(),
        snapshot_version="other",
    )
    with pytest.raises(ValidationError):
        RemoteResourceDiscoveryPageV1(resources=(descriptor,), snapshot_version="snap-1")


@pytest.mark.unit
def test_registry_identity_duplicate_and_missing() -> None:
    registry = RemoteResourceDiscoveryRegistry()
    provider = _default_provider(pages={})
    registry.register(provider)
    assert registry.resolve(provider_id="ms365_graph", integration_kind=IntegrationCategory.COLLABORATION_SUITE, source_kind="mailbox") is provider
    with pytest.raises(ValueError, match="already registered"):
        registry.register(provider)
    with pytest.raises(VendorKnowledgeError) as exc_info:
        registry.resolve(provider_id="ms365_graph", integration_kind=IntegrationCategory.COLLABORATION_SUITE, source_kind="missing")
    assert exc_info.value.code is VendorKnowledgeErrorCode.ADAPTER_NOT_FOUND
    registry.register(_default_provider(source_kind="channel", pages={}))
    assert registry.registered_source_kinds(provider_id="ms365_graph", integration_kind=IntegrationCategory.COLLABORATION_SUITE) == ("channel", "mailbox")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("setup", "exception_type"),
    [
        ("missing", TenantConnectionNotFound),
        ("cross_tenant", TenantConnectionNotFound),
        ("disabled", TenantConnectionInvalidState),
        ("revoked", TenantConnectionInvalidState),
    ],
)
@pytest.mark.asyncio
async def test_connection_boundary(setup: str, exception_type: type[Exception]) -> None:
    service, admin, repo, *_ = _stack()
    if setup == "missing":
        pass
    elif setup == "cross_tenant":
        repo.create(_connection(tenant_id="tenant-2"))
    else:
        status = TenantConnectionAdministrativeStatus.DISABLED if setup == "disabled" else TenantConnectionAdministrativeStatus.REVOKED
        repo.create(_connection(administrative_status=status))
    with pytest.raises(exception_type):
        await service.list_remote_resources(connection_ref="conn-1", source_kind="mailbox")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_active_discovery_and_source_kinds() -> None:
    service, admin, _, catalog, connection_registry, *_ = _stack()
    _wire(service, admin, catalog, connection_registry)
    page = await service.list_remote_resources(connection_ref="conn-1", source_kind="mailbox")
    assert len(page.resources) == 1
    assert service.list_source_kinds(connection_ref="conn-1") == ("mailbox",)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_runtime_integration_and_safe_connection() -> None:
    service, admin, _, catalog, connection_registry, _, provider, _ = _stack()
    _wire(service, admin, catalog, connection_registry)
    integration = connection_registry.resolve(
        tenant_id="tenant-1",
        connection_ref="conn-1",
        provider_id="ms365_graph",
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
    )
    await service.list_remote_resources(connection_ref="conn-1", source_kind="mailbox")
    assert provider.calls[0]["integration"] is integration
    connection = provider.calls[0]["connection"]
    assert isinstance(connection, SafeTenantConnectionV1)
    assert "credential_ref" not in connection.model_dump()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_page_tokens_preserved() -> None:
    token = " opaque-token "
    provider = _default_provider(pages={token: RemoteResourceCandidatePageV1(resources=(), snapshot_version="snap-1"), None: RemoteResourceCandidatePageV1(resources=(_candidate(),), next_page_token=" next ", snapshot_version="snap-1")})
    service, admin, _, catalog, connection_registry, _, recording, _ = _stack(provider)
    _wire(service, admin, catalog, connection_registry)
    await service.list_remote_resources(connection_ref="conn-1", source_kind="mailbox", page_token=token)
    assert recording.calls[0]["page_token"] == token
    page = await service.list_remote_resources(connection_ref="conn-1", source_kind="mailbox")
    assert page.next_page_token == " next "


@pytest.mark.unit
@pytest.mark.parametrize("limit", [1, 100, 0, 101])
@pytest.mark.asyncio
async def test_limit_validation(limit: int) -> None:
    resources = tuple(_candidate(remote_resource_id=f"res-{index}") for index in range(min(limit, 2) or 1))
    provider = _default_provider(pages={None: RemoteResourceCandidatePageV1(resources=resources, snapshot_version="snap-1")})
    service, admin, _, catalog, connection_registry, _, _, _ = _stack(provider)
    if limit in (1, 100):
        _wire(service, admin, catalog, connection_registry)
        page = await service.list_remote_resources(connection_ref="conn-1", source_kind="mailbox", limit=limit)
        assert len(page.resources) == min(limit, len(resources))
    else:
        admin.create(_connection())
        connection_registry.register(
            tenant_id="tenant-1",
            connection_ref="conn-1",
            provider_id="ms365_graph",
            integration_kind=IntegrationCategory.COLLABORATION_SUITE,
            integration=_FakeIntegration("ms365_graph", IntegrationCategory.COLLABORATION_SUITE),
        )
        with pytest.raises(ValueError, match="limit"):
            await service.list_remote_resources(connection_ref="conn-1", source_kind="mailbox", limit=limit)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_provider_failures_and_malformed_output() -> None:
    over_limit = _default_provider(pages={None: RemoteResourceCandidatePageV1(resources=(_candidate(remote_resource_id="a"), _candidate(remote_resource_id="b")), snapshot_version="snap-1")})
    service, admin, _, catalog, connection_registry, _, _, _ = _stack(over_limit)
    _wire(service, admin, catalog, connection_registry)
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await service.list_remote_resources(connection_ref="conn-1", source_kind="mailbox", limit=1)
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE

    failing = _default_provider(pages={}, fail_with=RuntimeError("secret provider detail"))
    service, admin, _, catalog, connection_registry, _, _, _ = _stack(failing)
    _wire(service, admin, catalog, connection_registry, register_capability=False)
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await service.list_remote_resources(connection_ref="conn-1", source_kind="mailbox")
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE
    assert "secret provider detail" not in str(exc_info.value)

    malformed = _default_provider(
        pages={
            None: RemoteResourceCandidatePageV1.model_construct(
                resources=(RemoteResourceCandidateV1.model_construct(remote_resource_id="", resource_type="mailbox", safe_display_label="Inbox", availability=RemoteResourceAvailabilityV1.AVAILABLE),),
                snapshot_version="snap-1",
            )
        }
    )
    service, admin, _, catalog, connection_registry, _, _, _ = _stack(malformed)
    _wire(service, admin, catalog, connection_registry, register_capability=False)
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await service.list_remote_resources(connection_ref="conn-1", source_kind="mailbox")
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE

    for bad_return in (None, object()):
        service, admin, _, catalog, connection_registry, _, _, _ = _stack(_MalformedProvider(return_value=bad_return))
        _wire(service, admin, catalog, connection_registry, register_capability=False)
        with pytest.raises(VendorKnowledgeError) as exc_info:
            await service.list_remote_resources(connection_ref="conn-1", source_kind="mailbox")
        _assert_invalid_provider_response(exc_info)

    class _BrokenDump:
        def model_dump(self) -> dict[str, object]:
            raise RuntimeError("internal-provider-secret")

    service, admin, _, catalog, connection_registry, _, _, _ = _stack(_MalformedProvider(return_value=_BrokenDump()))
    _wire(service, admin, catalog, connection_registry, register_capability=False)
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await service.list_remote_resources(connection_ref="conn-1", source_kind="mailbox")
    _assert_invalid_provider_response(exc_info, secret="internal-provider-secret")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_capability_intersection_and_catalog_validation() -> None:
    provider = _default_provider(
        pages={
            None: RemoteResourceCandidatePageV1(
                resources=(_candidate(supported_capability_ids=("mail.read", "mail.write", "mail.unavailable", "mail.unknown")),),
                snapshot_version="snap-1",
            )
        }
    )
    service, admin, _, catalog, connection_registry, _, _, _ = _stack(provider)
    _wire(service, admin, catalog, connection_registry, register_capability=False)
    catalog.register(_descriptor())
    catalog.register(_descriptor(capability_id="mail.write", effect=CapabilityEffectV1.WRITE))
    catalog.register(_descriptor(capability_id="mail.unavailable", available=False))
    page = await service.list_remote_resources(connection_ref="conn-1", source_kind="mailbox")
    assert page.resources[0].supported_capability_ids == ("mail.read",)

    class _BadCatalog(TenantLiveCapabilityCatalog):
        def list_capabilities(self, **kwargs: Any) -> tuple[LiveCapabilityDescriptorV1, ...]:
            if self._mode == "provider":
                return (_descriptor(provider_id="other"),)
            if self._mode == "kind":
                return (_descriptor(integration_kind=IntegrationCategory.ISSUE_TRACKER),)
            if self._mode == "malformed_object":
                return (object(),)  # type: ignore[return-value]
            if self._mode == "model_construct":
                return (LiveCapabilityDescriptorV1.model_construct(capability_id="mail.read"),)  # type: ignore[return-value]
            if self._mode == "catalog_exception":
                raise RuntimeError("internal-catalog-secret")
            descriptor = _descriptor()
            return (descriptor, descriptor)

    for mode in ("provider", "kind", "duplicate", "malformed_object", "model_construct", "catalog_exception"):
        store = ConditionalInMemoryDocumentStore()
        repo = DocumentStoreTenantConnectionRepository(store)
        port = RepositoryTenantConnectionPort(repo)
        bad_catalog = _BadCatalog(connection_port=port)
        bad_catalog._mode = mode
        discovery_registry = RemoteResourceDiscoveryRegistry()
        discovery_registry.register(_default_provider())
        connection_registry = KnowledgeConnectionRegistry()
        bad_service = TenantRemoteResourceDiscoveryService(
            tenant_id="tenant-1",
            connection_port=port,
            capability_catalog=bad_catalog,
            connection_registry=connection_registry,
            discovery_registry=discovery_registry,
        )
        bad_admin = TenantConnectionService(tenant_id="tenant-1", repository=repo)
        _wire(bad_service, bad_admin, bad_catalog, connection_registry)
        with pytest.raises(VendorKnowledgeError) as exc_info:
            await bad_service.list_remote_resources(connection_ref="conn-1", source_kind="mailbox")
        secret = "internal-catalog-secret" if mode == "catalog_exception" else ""
        _assert_invalid_provider_response(exc_info, secret=secret)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_deduplication_sorting_and_ephemeral_behavior() -> None:
    provider = _default_provider(
        pages={
            None: RemoteResourceCandidatePageV1(
                resources=(
                    _candidate(remote_resource_id="res-b"),
                    _candidate(remote_resource_id="res-a"),
                    _candidate(),
                    _candidate(),
                ),
                snapshot_version="snap-1",
            )
        }
    )
    service, admin, repo, catalog, connection_registry, _, _, store = _stack(provider, clock=lambda: datetime(2020, 1, 1, tzinfo=UTC))
    _wire(service, admin, catalog, connection_registry)
    before = repo.get(tenant_id="tenant-1", connection_ref="conn-1")
    rows_before = len(store._rows)
    page = await service.list_remote_resources(connection_ref="conn-1", source_kind="mailbox")
    assert [item.remote_resource_id for item in page.resources] == ["res-1", "res-a", "res-b"]
    assert all(resource.discovered_at == datetime(2020, 1, 1, tzinfo=UTC) for resource in page.resources)
    assert repo.get(tenant_id="tenant-1", connection_ref="conn-1") == before
    assert len(store._rows) == rows_before

    conflict = _default_provider(
        pages={None: RemoteResourceCandidatePageV1(resources=(_candidate(safe_display_label="One"), _candidate(safe_display_label="Two")), snapshot_version="snap-1")}
    )
    service, admin, _, catalog, connection_registry, _, _, _ = _stack(conflict)
    _wire(service, admin, catalog, connection_registry)
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await service.list_remote_resources(connection_ref="conn-1", source_kind="mailbox")
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE


@pytest.mark.unit
@pytest.mark.parametrize(
    ("clock_factory", "secret"),
    [
        (lambda: "not-a-datetime", ""),
        (lambda: datetime(2020, 1, 1), ""),
        (lambda: datetime(2020, 1, 1, tzinfo=timezone(timedelta(hours=1))), ""),
        pytest.param(_clock_runtime_error, "internal-clock-secret", id="clock_exception"),
    ],
)
@pytest.mark.asyncio
async def test_clock_boundary_normalized(clock_factory: Callable[[], object], secret: str) -> None:
    clock_calls = 0

    def _clock() -> object:
        nonlocal clock_calls
        clock_calls += 1
        return clock_factory()

    provider = _default_provider(
        pages={
            None: RemoteResourceCandidatePageV1(
                resources=(_candidate(remote_resource_id="res-a"), _candidate(remote_resource_id="res-b")),
                snapshot_version="snap-1",
            )
        }
    )
    service, admin, _, catalog, connection_registry, _, recording, _ = _stack(provider, clock=_clock)
    _wire(service, admin, catalog, connection_registry)
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await service.list_remote_resources(connection_ref="conn-1", source_kind="mailbox")
    _assert_invalid_provider_response(exc_info, secret=secret)
    assert len(recording.calls) == 1
    assert clock_calls == 1
