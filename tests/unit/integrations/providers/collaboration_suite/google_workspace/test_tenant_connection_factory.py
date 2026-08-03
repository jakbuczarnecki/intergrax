# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for Google Workspace tenant connection integration factory."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pytest

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.collaboration_suite.google_workspace.config import (
    GoogleWorkspaceCollaborationSuiteCompositionMode,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.contracts import (
    GoogleWorkspaceClientFamily,
    GoogleWorkspaceSourceKind,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.integration import (
    GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
    GoogleWorkspaceCollaborationSuiteIntegration,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.tenant_connection_factory import (
    GoogleWorkspaceTenantConnectionIntegrationFactory,
)
from intergrax.runtime.integrations.contracts import PlatformIntegrationStatus
from intergrax.runtime.vendor_knowledge.connections import KnowledgeConnectionRegistry
from intergrax.runtime.vendor_knowledge.tenant_connection_document_store import (
    DocumentStoreTenantConnectionRepository,
)
from intergrax.runtime.vendor_knowledge.tenant_connection_rehydration import (
    TenantConnectionRehydrationStatus,
    TenantConnectionRehydrator,
)
from intergrax.runtime.vendor_knowledge.tenant_connections import (
    TenantConnection,
    TenantConnectionAdministrativeStatus,
    TenantConnectionService,
)
from tests.unit.runtime.vendor_knowledge.test_tenant_connection_document_store import (
    ConditionalInMemoryDocumentStore,
)


@dataclass(frozen=True, slots=True)
class _FakeTransport:
    def get_json(
        self,
        *,
        source_kind: GoogleWorkspaceSourceKind,
        relative_path: str,
        params: Mapping[str, object] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> dict[str, object]:
        return {}


@dataclass(frozen=True, slots=True)
class _FakeClientFamily:
    _transport: _FakeTransport

    @property
    def transport(self) -> _FakeTransport:
        return self._transport


def _make_family() -> _FakeClientFamily:
    return _FakeClientFamily(_transport=_FakeTransport())


def _utc_now(offset_seconds: int = 0) -> datetime:
    return datetime.now(timezone.utc) + timedelta(seconds=offset_seconds)


def _google_connection(
    *,
    connection_ref: str = "gw-conn-1",
    tenant_id: str = "tenant-1",
    credential_ref: str = "cred-google-workspace",
    created_at: datetime | None = None,
    updated_at: datetime | None = None,
) -> TenantConnection:
    created = created_at or _utc_now()
    updated = updated_at or created
    return TenantConnection(
        connection_ref=connection_ref,
        tenant_id=tenant_id,
        provider_id=GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        safe_display_name="Google Workspace",
        administrative_status=TenantConnectionAdministrativeStatus.ACTIVE,
        credential_ref=credential_ref,
        validated_secret_free_config={},
        configuration_version=1,
        created_at=created,
        updated_at=updated,
    )


def _valid_credential_json() -> str:
    return json.dumps(
        {
            "type": "service_account",
            "client_id": "client-id-value",
            "private_key": "private-key-value",
        }
    )


class _SpyClientFactory:
    def __init__(self) -> None:
        self.calls: list[Mapping[str, str]] = []

    def create_client_family(
        self,
        *,
        credential_material: Mapping[str, str],
    ) -> GoogleWorkspaceClientFamily:
        self.calls.append(dict(credential_material))
        return _make_family()


class _RecordingSecretsStore:
    def __init__(self, *, secret: str) -> None:
        self.secret = secret
        self.calls: list[str] = []

    def get_secret(self, path: str, *, version: str | None = None) -> str:
        self.calls.append(path)
        return self.secret

    def put_secret(self, path: str, value: str) -> None:
        return None

    def delete_secret(self, path: str) -> None:
        return None


def _factory_kwargs(
    *,
    provider_id: str = GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
    integration_kind: IntegrationCategory = IntegrationCategory.COLLABORATION_SUITE,
    tenant_id: str = "tenant-1",
    connection_ref: str = "gw-conn-1",
    credential_ref: str = "cred-google-workspace",
    credential: str | None = None,
    secret_free_config: dict | None = None,
) -> dict[str, object]:
    return {
        "tenant_id": tenant_id,
        "connection_ref": connection_ref,
        "provider_id": provider_id,
        "integration_kind": integration_kind,
        "credential_ref": credential_ref,
        "credential": credential or _valid_credential_json(),
        "secret_free_config": secret_free_config or {},
    }


@pytest.mark.unit
def test_restart_rehydration_proof() -> None:
    credential_ref = "cred-google-workspace"
    credential_payload = _valid_credential_json()
    store = ConditionalInMemoryDocumentStore()
    repo_a = DocumentStoreTenantConnectionRepository(store)
    service_a = TenantConnectionService(tenant_id="tenant-1", repository=repo_a)
    service_a.create(_google_connection(credential_ref=credential_ref))

    repo_b = DocumentStoreTenantConnectionRepository(store)
    registry = KnowledgeConnectionRegistry()
    client_factory = _SpyClientFactory()
    factory = GoogleWorkspaceTenantConnectionIntegrationFactory(
        client_factory=client_factory,
    )
    secrets = _RecordingSecretsStore(secret=credential_payload)
    rehydrator = TenantConnectionRehydrator(
        repository=repo_b,
        secrets_store=secrets,
        integration_factory=factory,
        connection_registry=registry,
    )
    results = rehydrator.rehydrate_tenant(tenant_id="tenant-1")
    assert len(results) == 1
    assert results[0].status is TenantConnectionRehydrationStatus.REGISTERED
    assert results[0].error_code is None
    assert results[0].connection.tenant_id == "tenant-1"
    assert results[0].connection.connection_ref == "gw-conn-1"
    assert secrets.calls == [credential_ref]
    assert client_factory.calls == []

    integration = registry.resolve(
        tenant_id="tenant-1",
        connection_ref="gw-conn-1",
        provider_id=GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
    )
    assert isinstance(integration, GoogleWorkspaceCollaborationSuiteIntegration)
    assert integration.config.enabled is True
    assert integration.config.composition_mode is GoogleWorkspaceCollaborationSuiteCompositionMode.CREDENTIAL_REF
    assert integration.config.credential_ref == credential_ref
    assert integration.credential_resolver is not None
    assert integration.client_factory is client_factory
    resolved = integration.credential_resolver.resolve_credential(credential_ref)
    assert resolved == json.loads(credential_payload)
    integration.validate_runtime()
    health = integration.check_health()
    assert health.status is not PlatformIntegrationStatus.UNAVAILABLE

    dumped = results[0].model_dump()
    assert "client-id-value" not in str(dumped)
    assert "private-key-value" not in str(dumped)
    assert credential_ref not in dumped["connection"]
    public_view = integration.config.public_view()
    assert "client-id-value" not in str(public_view)
    assert "private-key-value" not in str(public_view)
    document = store.get(
        "vendor_knowledge_connections:tenant-1",
        "connection:gw-conn-1",
    )
    assert document is not None
    assert "client-id-value" not in str(document.data)
    assert "private-key-value" not in str(document.data)


@pytest.mark.unit
def test_wrong_provider_id_rejected() -> None:
    factory = GoogleWorkspaceTenantConnectionIntegrationFactory(
        client_factory=_SpyClientFactory(),
    )
    with pytest.raises(ValueError, match="provider_id"):
        factory.create_integration(**_factory_kwargs(provider_id="wrong_provider"))


@pytest.mark.unit
def test_wrong_integration_category_rejected() -> None:
    factory = GoogleWorkspaceTenantConnectionIntegrationFactory(
        client_factory=_SpyClientFactory(),
    )
    with pytest.raises(ValueError, match="integration_kind"):
        factory.create_integration(
            **_factory_kwargs(integration_kind=IntegrationCategory.ISSUE_TRACKER),
        )


@pytest.mark.unit
def test_blank_credential_ref_rejected() -> None:
    factory = GoogleWorkspaceTenantConnectionIntegrationFactory(
        client_factory=_SpyClientFactory(),
    )
    with pytest.raises(ValueError, match="credential_ref"):
        factory.create_integration(**_factory_kwargs(credential_ref="   "))


@pytest.mark.unit
def test_malformed_json_credential_rejected() -> None:
    factory = GoogleWorkspaceTenantConnectionIntegrationFactory(
        client_factory=_SpyClientFactory(),
    )
    with pytest.raises(ValueError, match="valid JSON"):
        factory.create_integration(**_factory_kwargs(credential="not-json"))


@pytest.mark.unit
def test_non_object_json_credential_rejected() -> None:
    factory = GoogleWorkspaceTenantConnectionIntegrationFactory(
        client_factory=_SpyClientFactory(),
    )
    with pytest.raises(ValueError, match="JSON object"):
        factory.create_integration(**_factory_kwargs(credential=json.dumps(["value"])))


@pytest.mark.unit
def test_empty_json_object_credential_rejected() -> None:
    factory = GoogleWorkspaceTenantConnectionIntegrationFactory(
        client_factory=_SpyClientFactory(),
    )
    with pytest.raises(ValueError, match="must not be empty"):
        factory.create_integration(**_factory_kwargs(credential=json.dumps({})))


@pytest.mark.unit
def test_blank_credential_key_rejected() -> None:
    factory = GoogleWorkspaceTenantConnectionIntegrationFactory(
        client_factory=_SpyClientFactory(),
    )
    with pytest.raises(ValueError, match="nonblank strings"):
        factory.create_integration(
            **_factory_kwargs(credential=json.dumps({"": "value"})),
        )


@pytest.mark.unit
def test_non_string_credential_value_rejected() -> None:
    factory = GoogleWorkspaceTenantConnectionIntegrationFactory(
        client_factory=_SpyClientFactory(),
    )
    with pytest.raises(ValueError, match="values must be strings"):
        factory.create_integration(
            **_factory_kwargs(credential=json.dumps({"type": 123})),
        )


@pytest.mark.unit
def test_non_empty_secret_free_config_rejected() -> None:
    factory = GoogleWorkspaceTenantConnectionIntegrationFactory(
        client_factory=_SpyClientFactory(),
    )
    with pytest.raises(ValueError, match="secret_free_config"):
        factory.create_integration(
            **_factory_kwargs(secret_free_config={"scope": "drive"}),
        )


@pytest.mark.unit
def test_resolver_rejects_mismatched_credential_ref() -> None:
    factory = GoogleWorkspaceTenantConnectionIntegrationFactory(
        client_factory=_SpyClientFactory(),
    )
    integration = factory.create_integration(**_factory_kwargs())
    assert isinstance(integration, GoogleWorkspaceCollaborationSuiteIntegration)
    resolver = integration.credential_resolver
    assert resolver is not None
    with pytest.raises(ValueError, match="does not match"):
        resolver.resolve_credential("other-ref")


@pytest.mark.unit
def test_malformed_google_credential_rehydration_unavailable() -> None:
    store = ConditionalInMemoryDocumentStore()
    repo = DocumentStoreTenantConnectionRepository(store)
    repo.create(_google_connection())
    client_factory = _SpyClientFactory()
    factory = GoogleWorkspaceTenantConnectionIntegrationFactory(
        client_factory=client_factory,
    )
    rehydrator = TenantConnectionRehydrator(
        repository=repo,
        secrets_store=_RecordingSecretsStore(secret="not-json"),
        integration_factory=factory,
        connection_registry=KnowledgeConnectionRegistry(),
    )
    results = rehydrator.rehydrate_tenant(tenant_id="tenant-1")
    assert results[0].status is TenantConnectionRehydrationStatus.UNAVAILABLE
    assert results[0].error_code == "tenant_connection_runtime_unavailable"
    assert client_factory.calls == []


@pytest.mark.unit
def test_no_client_factory_invocation_on_construction_or_rehydration() -> None:
    client_factory = _SpyClientFactory()
    factory = GoogleWorkspaceTenantConnectionIntegrationFactory(
        client_factory=client_factory,
    )
    integration = factory.create_integration(**_factory_kwargs())
    assert isinstance(integration, GoogleWorkspaceCollaborationSuiteIntegration)
    integration.validate_runtime()
    assert client_factory.calls == []

    store = ConditionalInMemoryDocumentStore()
    repo = DocumentStoreTenantConnectionRepository(store)
    repo.create(_google_connection())
    rehydrator = TenantConnectionRehydrator(
        repository=repo,
        secrets_store=_RecordingSecretsStore(secret=_valid_credential_json()),
        integration_factory=factory,
        connection_registry=KnowledgeConnectionRegistry(),
    )
    rehydrator.rehydrate_tenant(tenant_id="tenant-1")
    assert client_factory.calls == []


@pytest.mark.unit
def test_rehydration_and_registry_do_not_materialize_client_family() -> None:
    credential_ref = "cred-google-workspace"
    credential_payload = _valid_credential_json()
    store = ConditionalInMemoryDocumentStore()
    repo = DocumentStoreTenantConnectionRepository(store)
    repo.create(_google_connection(credential_ref=credential_ref))
    registry = KnowledgeConnectionRegistry()
    client_factory = _SpyClientFactory()
    factory = GoogleWorkspaceTenantConnectionIntegrationFactory(
        client_factory=client_factory,
    )
    rehydrator = TenantConnectionRehydrator(
        repository=repo,
        secrets_store=_RecordingSecretsStore(secret=credential_payload),
        integration_factory=factory,
        connection_registry=registry,
    )
    rehydrator.rehydrate_tenant(tenant_id="tenant-1")
    assert client_factory.calls == []

    integration = registry.resolve(
        tenant_id="tenant-1",
        connection_ref="gw-conn-1",
        provider_id=GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
    )
    assert isinstance(integration, GoogleWorkspaceCollaborationSuiteIntegration)
    integration.validate_runtime()
    assert client_factory.calls == []


@pytest.mark.unit
def test_first_require_client_family_materializes_once_from_rehydrated_integration() -> None:
    credential_ref = "cred-google-workspace"
    credential_payload = _valid_credential_json()
    expected_material = json.loads(credential_payload)
    store = ConditionalInMemoryDocumentStore()
    repo = DocumentStoreTenantConnectionRepository(store)
    repo.create(_google_connection(credential_ref=credential_ref))
    registry = KnowledgeConnectionRegistry()
    client_factory = _SpyClientFactory()
    factory = GoogleWorkspaceTenantConnectionIntegrationFactory(
        client_factory=client_factory,
    )
    rehydrator = TenantConnectionRehydrator(
        repository=repo,
        secrets_store=_RecordingSecretsStore(secret=credential_payload),
        integration_factory=factory,
        connection_registry=registry,
    )
    rehydrator.rehydrate_tenant(tenant_id="tenant-1")
    integration = registry.resolve(
        tenant_id="tenant-1",
        connection_ref="gw-conn-1",
        provider_id=GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
    )
    assert isinstance(integration, GoogleWorkspaceCollaborationSuiteIntegration)

    first = integration.require_client_family()
    second = integration.require_client_family()
    assert client_factory.calls == [expected_material]
    assert first is second

    public_view = integration.config.public_view()
    assert "client-id-value" not in str(public_view)
    assert "private-key-value" not in str(public_view)
    assert "client-id-value" not in repr(first)
    assert "private-key-value" not in repr(first)
