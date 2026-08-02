# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for tenant connection capability catalog and safe reads."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.runtime.vendor_knowledge.tenant_connection_capabilities import (
    CapabilityEffectV1,
    LiveCapabilityDescriptorV1,
    RepositoryTenantConnectionPort,
    TenantConnectionCapabilityReadService,
    TenantLiveCapabilityCatalog,
    is_bindable_read_only_capability,
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


def _connection(
    *,
    connection_ref: str = "conn-1",
    tenant_id: str = "tenant-1",
    provider_id: str = "ms365_graph",
    integration_kind: IntegrationCategory = IntegrationCategory.COLLABORATION_SUITE,
    administrative_status: TenantConnectionAdministrativeStatus = (
        TenantConnectionAdministrativeStatus.ACTIVE
    ),
) -> TenantConnection:
    created = _utc_now()
    return TenantConnection(
        connection_ref=connection_ref,
        tenant_id=tenant_id,
        provider_id=provider_id,
        integration_kind=integration_kind,
        safe_display_name="Example connection",
        administrative_status=administrative_status,
        credential_ref="cred-1",
        validated_secret_free_config={
            "token_endpoint": "https://auth.example.test/oauth2/token",
            "secret_version": "v1",
        },
        configuration_version=1,
        created_at=created,
        updated_at=created,
    )


def _descriptor(
    *,
    capability_id: str = "mail.read",
    provider_id: str = "ms365_graph",
    integration_kind: IntegrationCategory = IntegrationCategory.COLLABORATION_SUITE,
    effect: CapabilityEffectV1 = CapabilityEffectV1.READ,
    read_only: bool = True,
    available: bool = True,
    **extra: object,
) -> LiveCapabilityDescriptorV1:
    return LiveCapabilityDescriptorV1(
        capability_id=capability_id,
        provider_id=provider_id,
        integration_kind=integration_kind,
        effect=effect,
        read_only=read_only,
        resource_scope_required=False,
        request_schema_ref="schema://mail.read/request",
        result_schema_ref="schema://mail.read/result",
        available=available,
        **extra,
    )


def _stack() -> tuple[
    RepositoryTenantConnectionPort,
    TenantLiveCapabilityCatalog,
    TenantConnectionCapabilityReadService,
    TenantConnectionService,
    DocumentStoreTenantConnectionRepository,
]:
    store = ConditionalInMemoryDocumentStore()
    repo = DocumentStoreTenantConnectionRepository(store)
    port = RepositoryTenantConnectionPort(repo)
    catalog = TenantLiveCapabilityCatalog(connection_port=port)
    service = TenantConnectionCapabilityReadService(
        tenant_id="tenant-1",
        connection_port=port,
        capability_catalog=catalog,
    )
    admin = TenantConnectionService(tenant_id="tenant-1", repository=repo)
    return port, catalog, service, admin, repo


@pytest.mark.unit
def test_valid_descriptor() -> None:
    descriptor = _descriptor(supported_resource_types=(" mailbox ", "folder", "mailbox"))
    assert descriptor.supported_resource_types == ("folder", "mailbox")


@pytest.mark.unit
def test_descriptor_unknown_fields_rejected() -> None:
    with pytest.raises(ValidationError):
        LiveCapabilityDescriptorV1(
            capability_id="mail.read",
            provider_id="ms365_graph",
            integration_kind=IntegrationCategory.COLLABORATION_SUITE,
            effect=CapabilityEffectV1.READ,
            read_only=True,
            resource_scope_required=False,
            request_schema_ref="schema://mail.read/request",
            result_schema_ref="schema://mail.read/result",
            unexpected="nope",
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("capability_id", ""),
        ("provider_id", ""),
        ("request_schema_ref", ""),
        ("result_schema_ref", ""),
    ],
)
def test_descriptor_blank_required_strings_rejected(field_name: str, value: str) -> None:
    kwargs = {
        "capability_id": "mail.read",
        "provider_id": "ms365_graph",
        "integration_kind": IntegrationCategory.COLLABORATION_SUITE,
        "effect": CapabilityEffectV1.READ,
        "read_only": True,
        "resource_scope_required": False,
        "request_schema_ref": "schema://mail.read/request",
        "result_schema_ref": "schema://mail.read/result",
        field_name: value,
    }
    with pytest.raises(ValidationError):
        LiveCapabilityDescriptorV1(**kwargs)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("capability_id", "a" * 129),
        ("provider_id", "a" * 65),
        ("request_schema_ref", "a" * 257),
        ("result_schema_ref", "a" * 257),
    ],
)
def test_descriptor_length_limits(field_name: str, value: str) -> None:
    kwargs = {
        "capability_id": "mail.read",
        "provider_id": "ms365_graph",
        "integration_kind": IntegrationCategory.COLLABORATION_SUITE,
        "effect": CapabilityEffectV1.READ,
        "read_only": True,
        "resource_scope_required": False,
        "request_schema_ref": "schema://mail.read/request",
        "result_schema_ref": "schema://mail.read/result",
        field_name: value,
    }
    with pytest.raises(ValidationError):
        LiveCapabilityDescriptorV1(**kwargs)


@pytest.mark.unit
@pytest.mark.parametrize("field_name", ["max_result_items", "max_result_bytes"])
def test_descriptor_limits_below_one_rejected(field_name: str) -> None:
    with pytest.raises(ValidationError):
        _descriptor(**{field_name: 0})


@pytest.mark.unit
@pytest.mark.parametrize(
    ("effect", "read_only", "available", "capability_id", "expected"),
    [
        (CapabilityEffectV1.READ, True, True, "mail.read", True),
        (CapabilityEffectV1.WRITE, True, True, "mail.read", False),
        (CapabilityEffectV1.EXECUTE, True, True, "job.run", False),
        (CapabilityEffectV1.ADMIN, True, True, "tenant.admin", False),
        (CapabilityEffectV1.READ, False, True, "mail.read", False),
        (CapabilityEffectV1.READ, True, False, "mail.read", False),
        (CapabilityEffectV1.READ, True, True, "mail.delete", False),
    ],
)
def test_bindable_read_only_classification(
    effect: CapabilityEffectV1,
    read_only: bool,
    available: bool,
    capability_id: str,
    expected: bool,
) -> None:
    descriptor = _descriptor(
        capability_id=capability_id,
        effect=effect,
        read_only=read_only,
        available=available,
    )
    assert is_bindable_read_only_capability(descriptor) is expected


@pytest.mark.unit
def test_non_read_effect_rejected_without_dangerous_suffix() -> None:
    descriptor = _descriptor(
        capability_id="mail.send",
        effect=CapabilityEffectV1.WRITE,
        read_only=True,
    )
    assert is_bindable_read_only_capability(descriptor) is False


@pytest.mark.unit
def test_repository_port_get_returns_safe_projection() -> None:
    port, _, _, admin, _ = _stack()
    admin.create(_connection())
    safe = port.get_connection(tenant_id="tenant-1", connection_ref="conn-1")
    assert isinstance(safe, SafeTenantConnectionV1)
    dumped = safe.model_dump()
    assert "credential_ref" not in dumped
    assert "validated_secret_free_config" not in dumped


@pytest.mark.unit
def test_repository_port_cross_tenant_get_returns_none() -> None:
    port, _, _, admin, _ = _stack()
    admin.create(_connection())
    assert port.get_connection(tenant_id="tenant-2", connection_ref="conn-1") is None


@pytest.mark.unit
def test_repository_port_list_is_deterministic_and_filterable() -> None:
    port, _, _, admin, repo = _stack()
    admin.create(_connection(connection_ref="conn-b"))
    admin.create(_connection(connection_ref="conn-a"))
    repo.create(
        _connection(
            connection_ref="conn-disabled",
            administrative_status=TenantConnectionAdministrativeStatus.DISABLED,
        )
    )
    listed = port.list_connections(tenant_id="tenant-1", limit=10)
    assert [item.connection_ref for item in listed] == ["conn-a", "conn-b", "conn-disabled"]
    active = port.list_connections(
        tenant_id="tenant-1",
        administrative_status=TenantConnectionAdministrativeStatus.ACTIVE,
    )
    assert [item.connection_ref for item in active] == ["conn-a", "conn-b"]


@pytest.mark.unit
def test_catalog_register_and_duplicate_rejected() -> None:
    _, catalog, _, _, _ = _stack()
    descriptor = _descriptor()
    catalog.register(descriptor)
    with pytest.raises(ValueError, match="already registered"):
        catalog.register(descriptor)


@pytest.mark.unit
def test_catalog_filters_provider_and_integration_kind() -> None:
    _, catalog, _, admin, _ = _stack()
    admin.create(_connection())
    catalog.register(_descriptor(capability_id="alpha.read"))
    catalog.register(
        _descriptor(
            capability_id="beta.read",
            provider_id="other_provider",
        )
    )
    catalog.register(
        _descriptor(
            capability_id="gamma.read",
            integration_kind=IntegrationCategory.ISSUE_TRACKER,
        )
    )
    listed = catalog.list_capabilities(
        tenant_id="tenant-1",
        connection_ref="conn-1",
        remote_resource_id=None,
    )
    assert [item.capability_id for item in listed] == ["alpha.read"]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("status", "exception_type"),
    [
        (TenantConnectionAdministrativeStatus.DISABLED, TenantConnectionInvalidState),
        (TenantConnectionAdministrativeStatus.REVOKED, TenantConnectionInvalidState),
    ],
)
def test_catalog_rejects_non_active_connection(
    status: TenantConnectionAdministrativeStatus,
    exception_type: type[Exception],
) -> None:
    _, catalog, _, _, repo = _stack()
    repo.create(_connection(administrative_status=status))
    catalog.register(_descriptor())
    with pytest.raises(exception_type):
        catalog.list_capabilities(
            tenant_id="tenant-1",
            connection_ref="conn-1",
            remote_resource_id=None,
        )


@pytest.mark.unit
def test_catalog_missing_connection_rejected() -> None:
    _, catalog, _, _, _ = _stack()
    with pytest.raises(TenantConnectionNotFound):
        catalog.list_capabilities(
            tenant_id="tenant-1",
            connection_ref="missing",
            remote_resource_id=None,
        )


@pytest.mark.unit
@pytest.mark.parametrize("remote_resource_id", ["", " " * 3, "a" * 257])
def test_catalog_remote_resource_id_validation(remote_resource_id: str) -> None:
    _, catalog, _, admin, _ = _stack()
    admin.create(_connection())
    catalog.register(_descriptor())
    with pytest.raises(ValueError):
        catalog.list_capabilities(
            tenant_id="tenant-1",
            connection_ref="conn-1",
            remote_resource_id=remote_resource_id,
        )


@pytest.mark.unit
def test_read_service_safe_get_and_list() -> None:
    _, _, service, admin, _ = _stack()
    admin.create(_connection(connection_ref="conn-b"))
    admin.create(_connection(connection_ref="conn-a"))
    safe = service.get_connection("conn-a")
    assert isinstance(safe, SafeTenantConnectionV1)
    assert "credential_ref" not in safe.model_dump()
    listed = service.list_connections()
    assert [item.connection_ref for item in listed] == ["conn-a", "conn-b"]


@pytest.mark.unit
def test_read_service_missing_connection() -> None:
    _, _, service, _, _ = _stack()
    with pytest.raises(TenantConnectionNotFound):
        service.get_connection("missing")


@pytest.mark.unit
def test_read_service_returns_only_bindable_read_capabilities() -> None:
    _, catalog, service, admin, _ = _stack()
    admin.create(_connection())
    catalog.register(_descriptor(capability_id="alpha.read"))
    catalog.register(
        _descriptor(
            capability_id="beta.write",
            effect=CapabilityEffectV1.WRITE,
        )
    )
    catalog.register(
        _descriptor(
            capability_id="gamma.execute",
            effect=CapabilityEffectV1.EXECUTE,
        )
    )
    catalog.register(
        _descriptor(
            capability_id="delta.admin",
            effect=CapabilityEffectV1.ADMIN,
        )
    )
    catalog.register(
        _descriptor(
            capability_id="epsilon.read",
            available=False,
        )
    )
    listed = service.list_read_only_capabilities(connection_ref="conn-1")
    assert [item.capability_id for item in listed] == ["alpha.read"]
    dumped = str([item.model_dump() for item in listed])
    assert "credential_ref" not in dumped
    assert "validated_secret_free_config" not in dumped
