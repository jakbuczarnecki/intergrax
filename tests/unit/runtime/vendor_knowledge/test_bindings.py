# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for knowledge source binding models and service."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.runtime.vendor_knowledge.bindings import (
    KnowledgeSourceBinding,
    KnowledgeSourceBindingService,
    KnowledgeSourceBindingStatus,
    to_source_ref,
)
from intergrax.runtime.vendor_knowledge.binding_document_store import (
    DocumentStoreKnowledgeSourceBindingRepository,
)
from intergrax.runtime.vendor_knowledge.connections import (
    ConnectionAwareVendorResolver,
    KnowledgeConnectionRegistry,
)
from intergrax.runtime.vendor_knowledge.errors import (
    VendorKnowledgeError,
    VendorKnowledgeErrorCode,
)
from intergrax.runtime.vendor_knowledge.models import KnowledgeSourceScope
from intergrax.runtime.vendor_knowledge.registry import KnowledgeAdapterRegistry
from tests.unit.runtime.vendor_knowledge._fakes import (
    FakeAdapter,
    FakeConnectionIntegration,
    FakeIntegration,
    InMemoryDocumentStore,
    RecordingResolver,
    make_source,
)


def _scope(*, remote_scope_id: str = "scope-1") -> KnowledgeSourceScope:
    return KnowledgeSourceScope(
        remote_scope_id=remote_scope_id,
        remote_scope_type="project",
        safe_display_name="Example Project",
        parameters={},
    )


def _binding(
    *,
    binding_id: str = "bind-1",
    tenant_id: str = "tenant-1",
    provider_id: str = "example",
    integration_kind: IntegrationCategory = IntegrationCategory.ISSUE_TRACKER,
    source_kind: str = "issues",
    connection_ref: str = "conn-1",
    credential_ref: str | None = "cred-1",
    status: KnowledgeSourceBindingStatus = KnowledgeSourceBindingStatus.ACTIVE,
    configuration_version: int = 1,
    broad_scope: bool = False,
    scope_approval_ref: str | None = None,
    safe_display_name: str = "Example binding",
    remote_scope_id: str = "scope-1",
) -> KnowledgeSourceBinding:
    return KnowledgeSourceBinding(
        binding_id=binding_id,
        tenant_id=tenant_id,
        provider_id=provider_id,
        integration_kind=integration_kind,
        source_kind=source_kind,
        connection_ref=connection_ref,
        credential_ref=credential_ref,
        safe_display_name=safe_display_name,
        scope=_scope(remote_scope_id=remote_scope_id),
        status=status,
        configuration_version=configuration_version,
        broad_scope=broad_scope,
        scope_approval_ref=scope_approval_ref,
    )


def _service(
    *,
    tenant_id: str = "tenant-1",
    store: InMemoryDocumentStore | None = None,
    resolver: RecordingResolver | None = None,
    registry: KnowledgeAdapterRegistry | None = None,
) -> tuple[KnowledgeSourceBindingService, InMemoryDocumentStore, RecordingResolver]:
    document_store = store or InMemoryDocumentStore()
    repository = DocumentStoreKnowledgeSourceBindingRepository(document_store)
    integration = FakeIntegration()
    resolved = resolver or RecordingResolver(integration=integration)
    adapter_registry = registry or KnowledgeAdapterRegistry()
    if registry is None:
        adapter_registry.register(FakeAdapter())
    service = KnowledgeSourceBindingService(
        tenant_id=tenant_id,
        repository=repository,
        integration_resolver=resolved,
        adapter_registry=adapter_registry,
    )
    return service, document_store, resolved


@pytest.mark.unit
def test_binding_accepts_valid_model() -> None:
    binding = _binding()
    assert binding.binding_id == "bind-1"
    assert binding.configuration_version == 1
    assert binding.status is KnowledgeSourceBindingStatus.ACTIVE


@pytest.mark.unit
@pytest.mark.parametrize(
    "field_name",
    [
        "binding_id",
        "tenant_id",
        "provider_id",
        "source_kind",
        "connection_ref",
        "safe_display_name",
    ],
)
def test_binding_rejects_empty_identifiers(field_name: str) -> None:
    kwargs = {
        "binding_id": "bind-1",
        "tenant_id": "tenant-1",
        "provider_id": "example",
        "integration_kind": IntegrationCategory.ISSUE_TRACKER,
        "source_kind": "issues",
        "connection_ref": "conn-1",
        "safe_display_name": "Example binding",
        "scope": _scope(),
        "status": KnowledgeSourceBindingStatus.ACTIVE,
        "configuration_version": 1,
    }
    kwargs[field_name] = "   "
    with pytest.raises(ValidationError):
        KnowledgeSourceBinding(**kwargs)


@pytest.mark.unit
def test_binding_requires_configuration_version_at_least_one() -> None:
    with pytest.raises(ValidationError):
        _binding(configuration_version=0)


@pytest.mark.unit
def test_broad_scope_requires_approval_ref() -> None:
    with pytest.raises(ValidationError):
        _binding(broad_scope=True, scope_approval_ref=None)


@pytest.mark.unit
def test_binding_forbids_secret_bearing_fields() -> None:
    with pytest.raises(ValidationError):
        KnowledgeSourceBinding(
            binding_id="bind-1",
            tenant_id="tenant-1",
            provider_id="example",
            integration_kind=IntegrationCategory.ISSUE_TRACKER,
            source_kind="issues",
            connection_ref="conn-1",
            safe_display_name="Example binding",
            scope=_scope(),
            status=KnowledgeSourceBindingStatus.ACTIVE,
            configuration_version=1,
            access_token="secret",  # type: ignore[call-arg]
        )


@pytest.mark.unit
def test_to_source_ref_preserves_connection_ref() -> None:
    binding = _binding(connection_ref="conn-exact")
    source = to_source_ref(binding)
    assert source.tenant_id == binding.tenant_id
    assert source.provider_id == binding.provider_id
    assert source.integration_kind == binding.integration_kind
    assert source.source_kind == binding.source_kind
    assert source.connection_ref == "conn-exact"
    assert source.scope == binding.scope


@pytest.mark.unit
def test_cross_tenant_create_get_update_resolve_fail_closed() -> None:
    service, _store, _resolver = _service(tenant_id="tenant-1")
    foreign = _binding(tenant_id="tenant-other")
    with pytest.raises(VendorKnowledgeError) as create_exc:
        service.create(foreign)
    assert create_exc.value.code is VendorKnowledgeErrorCode.TENANT_MISMATCH

    service.create(_binding())
    other_service, _other_store, _other_resolver = _service(
        tenant_id="tenant-2",
        store=InMemoryDocumentStore(),
    )
    with pytest.raises(VendorKnowledgeError) as get_exc:
        other_service.get("bind-1")
    assert get_exc.value.code is VendorKnowledgeErrorCode.INTEGRATION_NOT_FOUND

    with pytest.raises(VendorKnowledgeError) as update_exc:
        service.update(
            _binding(tenant_id="tenant-other", configuration_version=2),
            expected_configuration_version=1,
        )
    assert update_exc.value.code is VendorKnowledgeErrorCode.TENANT_MISMATCH

    with pytest.raises(VendorKnowledgeError) as resolve_exc:
        other_service.resolve_source("bind-1")
    assert resolve_exc.value.code is VendorKnowledgeErrorCode.INTEGRATION_NOT_FOUND


@pytest.mark.unit
def test_create_checks_adapter_and_integration() -> None:
    service, store, resolver = _service()
    created = service.create(_binding())
    assert created.binding_id == "bind-1"
    assert len(resolver.calls) == 1
    assert resolver.calls[0].connection_ref == "conn-1"
    assert store.get(
        "vendor_knowledge_bindings:tenant-1",
        "binding:bind-1",
    ) is not None


@pytest.mark.unit
def test_missing_adapter_does_not_persist_binding() -> None:
    store = InMemoryDocumentStore()
    service, _store, _resolver = _service(
        store=store,
        registry=KnowledgeAdapterRegistry(),
    )
    with pytest.raises(VendorKnowledgeError) as exc:
        service.create(_binding())
    assert exc.value.code is VendorKnowledgeErrorCode.ADAPTER_NOT_FOUND
    assert store.get("vendor_knowledge_bindings:tenant-1", "binding:bind-1") is None


@pytest.mark.unit
def test_missing_integration_does_not_persist_binding() -> None:
    store = InMemoryDocumentStore()
    resolver = RecordingResolver(
        integration=FakeIntegration(),
        error=VendorKnowledgeError(
            code=VendorKnowledgeErrorCode.INTEGRATION_NOT_FOUND,
            safe_message="integration missing",
        ),
    )
    service, _store, _resolver = _service(store=store, resolver=resolver)
    with pytest.raises(VendorKnowledgeError) as exc:
        service.create(_binding())
    assert exc.value.code is VendorKnowledgeErrorCode.INTEGRATION_NOT_FOUND
    assert store.get("vendor_knowledge_bindings:tenant-1", "binding:bind-1") is None


@pytest.mark.unit
def test_update_requires_next_configuration_version() -> None:
    service, _store, _resolver = _service()
    service.create(_binding(configuration_version=1))
    updated = service.update(
        _binding(
            configuration_version=2,
            safe_display_name="Renamed",
            credential_ref="cred-2",
        ),
        expected_configuration_version=1,
    )
    assert updated.configuration_version == 2
    assert updated.safe_display_name == "Renamed"

    with pytest.raises(VendorKnowledgeError) as exc:
        service.update(
            _binding(configuration_version=4, safe_display_name="Too far"),
            expected_configuration_version=2,
        )
    assert exc.value.code is VendorKnowledgeErrorCode.CONFIGURATION_ERROR


@pytest.mark.unit
def test_identity_fields_are_immutable_on_update() -> None:
    registry = KnowledgeAdapterRegistry()
    registry.register(FakeAdapter())
    registry.register(FakeAdapter(provider_id="other-provider"))
    service, _store, _resolver = _service(registry=registry)
    service.create(_binding())
    with pytest.raises(VendorKnowledgeError) as exc:
        service.update(
            _binding(
                provider_id="other-provider",
                configuration_version=2,
            ),
            expected_configuration_version=1,
        )
    assert exc.value.code is VendorKnowledgeErrorCode.CONFIGURATION_ERROR
    assert service.get("bind-1").provider_id == "example"


@pytest.mark.unit
def test_active_binding_resolves_source() -> None:
    service, _store, _resolver = _service()
    service.create(_binding(connection_ref="conn-live"))
    source = service.resolve_source("bind-1")
    assert source.connection_ref == "conn-live"
    assert source == make_source(connection_ref="conn-live")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("status", "code"),
    [
        (KnowledgeSourceBindingStatus.DISABLED, VendorKnowledgeErrorCode.CONFIGURATION_ERROR),
        (KnowledgeSourceBindingStatus.REVOKED, VendorKnowledgeErrorCode.AUTHORIZATION_DENIED),
        (KnowledgeSourceBindingStatus.EXPIRED, VendorKnowledgeErrorCode.AUTHENTICATION_FAILED),
    ],
)
def test_non_active_bindings_fail_closed(
    status: KnowledgeSourceBindingStatus,
    code: VendorKnowledgeErrorCode,
) -> None:
    service, _store, _resolver = _service()
    service.create(_binding(status=status))
    with pytest.raises(VendorKnowledgeError) as exc:
        service.resolve_source("bind-1")
    assert exc.value.code is code
    assert "conn-" not in exc.value.safe_message
    assert "cred-" not in exc.value.safe_message


@pytest.mark.unit
def test_list_is_tenant_scoped_and_deterministic() -> None:
    store = InMemoryDocumentStore()
    service_a, _store_a, _resolver_a = _service(tenant_id="tenant-a", store=store)
    service_b, _store_b, _resolver_b = _service(tenant_id="tenant-b", store=store)
    service_a.create(_binding(binding_id="bind-b", tenant_id="tenant-a"))
    service_a.create(_binding(binding_id="bind-a", tenant_id="tenant-a", connection_ref="conn-2"))
    service_b.create(_binding(binding_id="bind-z", tenant_id="tenant-b"))

    listed = service_a.list()
    assert [item.binding_id for item in listed] == ["bind-a", "bind-b"]
    assert all(item.tenant_id == "tenant-a" for item in listed)


@pytest.mark.unit
def test_one_connection_ref_can_have_many_source_bindings() -> None:
    store = InMemoryDocumentStore()
    registry = KnowledgeAdapterRegistry()
    registry.register(FakeAdapter(source_kind="issues"))
    registry.register(
        FakeAdapter(
            provider_id="example",
            integration_kind=IntegrationCategory.ISSUE_TRACKER,
            source_kind="comments",
        )
    )
    service, _store, _resolver = _service(store=store, registry=registry)
    service.create(_binding(binding_id="b1", source_kind="issues", connection_ref="shared"))
    service.create(
        _binding(
            binding_id="b2",
            source_kind="comments",
            connection_ref="shared",
            remote_scope_id="scope-2",
        )
    )
    listed = service.list()
    assert len(listed) == 2
    assert {item.connection_ref for item in listed} == {"shared"}


@pytest.mark.unit
def test_multiple_microsoft_source_kinds_share_one_connection_ref() -> None:
    store = InMemoryDocumentStore()
    connection_registry = KnowledgeConnectionRegistry()
    integration = FakeConnectionIntegration()
    connection_registry.register(
        tenant_id="tenant-1",
        connection_ref="m365-conn",
        provider_id="ms365_graph",
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        integration=integration,
    )
    fallback = RecordingResolver(integration=FakeIntegration())
    resolver = ConnectionAwareVendorResolver(
        tenant_id="tenant-1",
        connection_registry=connection_registry,
        fallback_resolver=fallback,
    )
    adapter_registry = KnowledgeAdapterRegistry()
    source_kinds = ("drive", "mail", "calendar", "teams_chat", "teams_channel")
    for kind in source_kinds:
        adapter_registry.register(
            FakeAdapter(
                provider_id="ms365_graph",
                integration_kind=IntegrationCategory.COLLABORATION_SUITE,
                source_kind=kind,
            )
        )
    service = KnowledgeSourceBindingService(
        tenant_id="tenant-1",
        repository=DocumentStoreKnowledgeSourceBindingRepository(store),
        integration_resolver=resolver,
        adapter_registry=adapter_registry,
    )
    for kind in source_kinds:
        service.create(
            _binding(
                binding_id=f"bind-{kind}",
                provider_id="ms365_graph",
                integration_kind=IntegrationCategory.COLLABORATION_SUITE,
                source_kind=kind,
                connection_ref="m365-conn",
                remote_scope_id=f"scope-{kind}",
            )
        )
    listed = service.list()
    assert len(listed) == 5
    assert {item.connection_ref for item in listed} == {"m365-conn"}
    assert {item.source_kind for item in listed} == set(source_kinds)
    assert fallback.calls == []
