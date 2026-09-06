# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for connection registry and connection-aware resolver."""

from __future__ import annotations

import pytest

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.runtime.vendor_knowledge.connections import (
    ConnectionAwareVendorResolver,
    KnowledgeConnectionRegistry,
)
from intergrax.runtime.vendor_knowledge.tenant_connection_factory_registry import (
    TenantConnectionIntegrationFactoryRegistry,
)
from intergrax.runtime.vendor_knowledge.tenant_connection_factory_contract import (
    EagerTenantConnectionIntegrationFactoryMixin,
)
from intergrax.runtime.vendor_knowledge.contracts import VendorIntegrationResolver
from intergrax.runtime.vendor_knowledge.errors import (
    VendorKnowledgeError,
    VendorKnowledgeErrorCode,
)
from tests.unit.runtime.vendor_knowledge._fakes import (
    FakeConnectionIntegration,
    FakeIntegration,
    RecordingResolver,
    make_source,
)


@pytest.mark.unit
def test_register_and_resolve_returns_same_instance() -> None:
    registry = KnowledgeConnectionRegistry()
    integration = FakeConnectionIntegration()
    registry.register(
        tenant_id="tenant-1",
        connection_ref="conn-1",
        provider_id="ms365_graph",
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        integration=integration,
    )
    resolved = registry.resolve(
        tenant_id="tenant-1",
        connection_ref="conn-1",
        provider_id="ms365_graph",
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
    )
    assert resolved is integration


@pytest.mark.unit
def test_duplicate_registration_rejected_and_does_not_overwrite() -> None:
    registry = KnowledgeConnectionRegistry()
    first = FakeConnectionIntegration(label="first")
    second = FakeConnectionIntegration(label="second")
    registry.register(
        tenant_id="tenant-1",
        connection_ref="conn-1",
        provider_id="ms365_graph",
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        integration=first,
    )
    with pytest.raises(ValueError, match="already registered"):
        registry.register(
            tenant_id="tenant-1",
            connection_ref="conn-1",
            provider_id="ms365_graph",
            integration_kind=IntegrationCategory.COLLABORATION_SUITE,
            integration=second,
        )
    resolved = registry.resolve(
        tenant_id="tenant-1",
        connection_ref="conn-1",
        provider_id="ms365_graph",
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
    )
    assert resolved is first
    assert resolved is not second


@pytest.mark.unit
def test_tenant_isolation() -> None:
    registry = KnowledgeConnectionRegistry()
    integration = FakeConnectionIntegration()
    registry.register(
        tenant_id="tenant-a",
        connection_ref="conn-1",
        provider_id="ms365_graph",
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        integration=integration,
    )
    with pytest.raises(VendorKnowledgeError) as exc:
        registry.resolve(
            tenant_id="tenant-b",
            connection_ref="conn-1",
            provider_id="ms365_graph",
            integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        )
    assert exc.value.code is VendorKnowledgeErrorCode.INTEGRATION_NOT_FOUND
    assert "conn-1" not in exc.value.safe_message


@pytest.mark.unit
def test_unknown_connection() -> None:
    registry = KnowledgeConnectionRegistry()
    with pytest.raises(VendorKnowledgeError) as exc:
        registry.resolve(
            tenant_id="tenant-1",
            connection_ref="missing",
            provider_id="ms365_graph",
            integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        )
    assert exc.value.code is VendorKnowledgeErrorCode.INTEGRATION_NOT_FOUND
    assert "missing" not in exc.value.safe_message


@pytest.mark.unit
def test_provider_mismatch() -> None:
    registry = KnowledgeConnectionRegistry()
    registry.register(
        tenant_id="tenant-1",
        connection_ref="conn-1",
        provider_id="ms365_graph",
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        integration=FakeConnectionIntegration(),
    )
    with pytest.raises(VendorKnowledgeError) as exc:
        registry.resolve(
            tenant_id="tenant-1",
            connection_ref="conn-1",
            provider_id="other-provider",
            integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        )
    assert exc.value.code is VendorKnowledgeErrorCode.INTEGRATION_NOT_FOUND


@pytest.mark.unit
def test_category_mismatch() -> None:
    registry = KnowledgeConnectionRegistry()
    registry.register(
        tenant_id="tenant-1",
        connection_ref="conn-1",
        provider_id="ms365_graph",
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        integration=FakeConnectionIntegration(),
    )
    with pytest.raises(VendorKnowledgeError) as exc:
        registry.resolve(
            tenant_id="tenant-1",
            connection_ref="conn-1",
            provider_id="ms365_graph",
            integration_kind=IntegrationCategory.ISSUE_TRACKER,
        )
    assert exc.value.code is VendorKnowledgeErrorCode.INTEGRATION_CATEGORY_MISMATCH


@pytest.mark.unit
def test_fallback_for_missing_connection_ref() -> None:
    registry = KnowledgeConnectionRegistry()
    fallback_integration = FakeIntegration()
    fallback = RecordingResolver(integration=fallback_integration)
    resolver = ConnectionAwareVendorResolver(
        tenant_id="tenant-1",
        connection_registry=registry,
        fallback_resolver=fallback,
    )
    source = make_source(connection_ref=None)
    resolved = resolver.resolve(source=source)
    assert resolved is fallback_integration
    assert fallback.calls == [source]


@pytest.mark.unit
def test_no_fallback_when_connection_ref_set() -> None:
    registry = KnowledgeConnectionRegistry()
    connection_integration = FakeConnectionIntegration()
    registry.register(
        tenant_id="tenant-1",
        connection_ref="conn-1",
        provider_id="ms365_graph",
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        integration=connection_integration,
    )
    fallback = RecordingResolver(integration=FakeIntegration())
    resolver = ConnectionAwareVendorResolver(
        tenant_id="tenant-1",
        connection_registry=registry,
        fallback_resolver=fallback,
    )
    source = make_source(
        provider_id="ms365_graph",
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        source_kind="drive",
        connection_ref="conn-1",
    )
    resolved = resolver.resolve(source=source)
    assert resolved is connection_integration
    assert fallback.calls == []


@pytest.mark.unit
def test_tenant_mismatch_before_registry_or_fallback() -> None:
    registry = KnowledgeConnectionRegistry()
    registry.register(
        tenant_id="tenant-1",
        connection_ref="conn-1",
        provider_id="ms365_graph",
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        integration=FakeConnectionIntegration(),
    )
    fallback = RecordingResolver(integration=FakeIntegration())
    resolver = ConnectionAwareVendorResolver(
        tenant_id="tenant-1",
        connection_registry=registry,
        fallback_resolver=fallback,
    )
    with pytest.raises(VendorKnowledgeError) as exc:
        resolver.resolve(
            source=make_source(
                tenant_id="tenant-other",
                provider_id="ms365_graph",
                integration_kind=IntegrationCategory.COLLABORATION_SUITE,
                source_kind="drive",
                connection_ref="conn-1",
            )
        )
    assert exc.value.code is VendorKnowledgeErrorCode.TENANT_MISMATCH
    assert fallback.calls == []

    with pytest.raises(VendorKnowledgeError) as fallback_exc:
        resolver.resolve(source=make_source(tenant_id="tenant-other", connection_ref=None))
    assert fallback_exc.value.code is VendorKnowledgeErrorCode.TENANT_MISMATCH
    assert fallback.calls == []


@pytest.mark.unit
def test_no_global_state_between_registry_instances() -> None:
    first = KnowledgeConnectionRegistry()
    second = KnowledgeConnectionRegistry()
    first.register(
        tenant_id="tenant-1",
        connection_ref="conn-1",
        provider_id="ms365_graph",
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        integration=FakeConnectionIntegration(label="first"),
    )
    with pytest.raises(VendorKnowledgeError):
        second.resolve(
            tenant_id="tenant-1",
            connection_ref="conn-1",
            provider_id="ms365_graph",
            integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        )


@pytest.mark.unit
def test_registry_has_no_secret_fields_or_network() -> None:
    registry = KnowledgeConnectionRegistry()
    assert not hasattr(registry, "secrets_store")
    assert not hasattr(registry, "credential_ref")
    assert "token" not in KnowledgeConnectionRegistry.__dict__
    assert "password" not in KnowledgeConnectionRegistry.__dict__


@pytest.mark.unit
def test_resolver_satisfies_vendor_integration_resolver_protocol() -> None:
    resolver = ConnectionAwareVendorResolver(
        tenant_id="tenant-1",
        connection_registry=KnowledgeConnectionRegistry(),
        fallback_resolver=RecordingResolver(integration=FakeIntegration()),
    )
    assert isinstance(resolver, VendorIntegrationResolver)


@pytest.mark.unit
def test_register_rejects_provider_id_mismatch() -> None:
    registry = KnowledgeConnectionRegistry()
    with pytest.raises(ValueError, match="provider_id") as exc:
        registry.register(
            tenant_id="tenant-1",
            connection_ref="conn-secret-ref",
            provider_id="other-provider",
            integration_kind=IntegrationCategory.COLLABORATION_SUITE,
            integration=FakeConnectionIntegration(),
        )
    assert "conn-secret-ref" not in str(exc.value)
    with pytest.raises(VendorKnowledgeError):
        registry.resolve(
            tenant_id="tenant-1",
            connection_ref="conn-secret-ref",
            provider_id="other-provider",
            integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        )


@pytest.mark.unit
def test_register_rejects_category_mismatch() -> None:
    registry = KnowledgeConnectionRegistry()
    with pytest.raises(ValueError, match="category") as exc:
        registry.register(
            tenant_id="tenant-1",
            connection_ref="conn-secret-ref",
            provider_id="ms365_graph",
            integration_kind=IntegrationCategory.ISSUE_TRACKER,
            integration=FakeConnectionIntegration(),
        )
    assert "conn-secret-ref" not in str(exc.value)


@pytest.mark.unit
def test_register_rejects_missing_identity_attributes() -> None:
    registry = KnowledgeConnectionRegistry()
    with pytest.raises(ValueError, match="identity") as exc:
        registry.register(
            tenant_id="tenant-1",
            connection_ref="conn-secret-ref",
            provider_id="ms365_graph",
            integration_kind=IntegrationCategory.COLLABORATION_SUITE,
            integration=object(),
        )
    assert "conn-secret-ref" not in str(exc.value)


@pytest.mark.unit
def test_failed_register_does_not_create_entry() -> None:
    registry = KnowledgeConnectionRegistry()
    with pytest.raises(ValueError):
        registry.register(
            tenant_id="tenant-1",
            connection_ref="conn-1",
            provider_id="ms365_graph",
            integration_kind=IntegrationCategory.COLLABORATION_SUITE,
            integration=FakeConnectionIntegration(provider_id="wrong"),
        )
    with pytest.raises(VendorKnowledgeError) as exc:
        registry.resolve(
            tenant_id="tenant-1",
            connection_ref="conn-1",
            provider_id="ms365_graph",
            integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        )
    assert exc.value.code is VendorKnowledgeErrorCode.INTEGRATION_NOT_FOUND


@pytest.mark.unit
def test_resolve_detects_provider_id_mutation() -> None:
    registry = KnowledgeConnectionRegistry()
    integration = FakeConnectionIntegration()
    registry.register(
        tenant_id="tenant-1",
        connection_ref="conn-1",
        provider_id="ms365_graph",
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        integration=integration,
    )
    integration.provider_id = "mutated-provider"
    with pytest.raises(VendorKnowledgeError) as exc:
        registry.resolve(
            tenant_id="tenant-1",
            connection_ref="conn-1",
            provider_id="ms365_graph",
            integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        )
    assert exc.value.code is VendorKnowledgeErrorCode.INTEGRATION_NOT_FOUND
    assert "conn-1" not in exc.value.safe_message


@pytest.mark.unit
def test_resolve_detects_category_mutation() -> None:
    registry = KnowledgeConnectionRegistry()
    integration = FakeConnectionIntegration()
    registry.register(
        tenant_id="tenant-1",
        connection_ref="conn-1",
        provider_id="ms365_graph",
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        integration=integration,
    )
    integration.integration_kind = IntegrationCategory.ISSUE_TRACKER.value
    with pytest.raises(VendorKnowledgeError) as exc:
        registry.resolve(
            tenant_id="tenant-1",
            connection_ref="conn-1",
            provider_id="ms365_graph",
            integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        )
    assert exc.value.code is VendorKnowledgeErrorCode.INTEGRATION_CATEGORY_MISMATCH
    assert "conn-1" not in exc.value.safe_message


@pytest.mark.unit
def test_resolve_returns_same_valid_instance() -> None:
    registry = KnowledgeConnectionRegistry()
    integration = FakeConnectionIntegration()
    registry.register(
        tenant_id="tenant-1",
        connection_ref="conn-1",
        provider_id="ms365_graph",
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        integration=integration,
    )
    resolved = registry.resolve(
        tenant_id="tenant-1",
        connection_ref="conn-1",
        provider_id="ms365_graph",
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
    )
    assert resolved is integration


@pytest.mark.unit
def test_error_messages_never_include_connection_ref() -> None:
    registry = KnowledgeConnectionRegistry()
    marker = "conn-must-not-leak"
    cases: list[Exception] = []
    try:
        registry.register(
            tenant_id="tenant-1",
            connection_ref=marker,
            provider_id="wrong",
            integration_kind=IntegrationCategory.COLLABORATION_SUITE,
            integration=FakeConnectionIntegration(),
        )
    except ValueError as exc:
        cases.append(exc)
    try:
        registry.register(
            tenant_id="tenant-1",
            connection_ref=marker,
            provider_id="ms365_graph",
            integration_kind=IntegrationCategory.ISSUE_TRACKER,
            integration=FakeConnectionIntegration(),
        )
    except ValueError as exc:
        cases.append(exc)
    try:
        registry.resolve(
            tenant_id="tenant-1",
            connection_ref=marker,
            provider_id="ms365_graph",
            integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        )
    except VendorKnowledgeError as exc:
        cases.append(exc)
    assert cases
    for err in cases:
        assert marker not in str(err)
        if isinstance(err, VendorKnowledgeError):
            assert marker not in err.safe_message
            assert marker not in repr(err)


@pytest.mark.unit
def test_factory_registry_routes_by_provider_and_category() -> None:
    class _Factory(EagerTenantConnectionIntegrationFactoryMixin):
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def create_integration(self, **kwargs: object) -> object:
            return self.create_integration_with_resolved_credential(**kwargs)

        def create_integration_with_resolved_credential(
            self,
            *,
            resolved_credential: str,
            **kwargs: object,
        ) -> object:
            self.calls.append({**kwargs, "resolved_credential": resolved_credential})
            return "integration"

    factory = _Factory()
    registry = TenantConnectionIntegrationFactoryRegistry()
    registry.register(
        provider_id="provider-a",
        integration_kind=IntegrationCategory.ISSUE_TRACKER,
        factory=factory,  # type: ignore[arg-type]
    )

    resolved = registry.create_integration(
        tenant_id="tenant-1",
        connection_ref="connection-1",
        provider_id="provider-a",
        integration_kind=IntegrationCategory.ISSUE_TRACKER,
        credential_ref="credentials/connection-1",
        credential="runtime-secret",
        secret_free_config={},
    )

    assert resolved == "integration"
    assert factory.calls[0]["resolved_credential"] == "runtime-secret"
    assert factory.calls[0]["credential_ref"] == "credentials/connection-1"


@pytest.mark.unit
def test_factory_registry_fails_closed_for_unknown_factory() -> None:
    registry = TenantConnectionIntegrationFactoryRegistry()

    with pytest.raises(ValueError, match="factory is unavailable"):
        registry.create_integration(
            tenant_id="tenant-1",
            connection_ref="connection-1",
            provider_id="unknown",
            integration_kind=IntegrationCategory.ISSUE_TRACKER,
            credential_ref="credentials/connection-1",
            credential="runtime-secret",
            secret_free_config={},
        )
