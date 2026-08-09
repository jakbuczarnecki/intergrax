# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from intergrax.integrations.contracts.base import (
    IntegrationCategory,
    IntegrationConfigurationError,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.integration import (
    MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
    Ms365GraphCollaborationSuiteIntegration,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.tenant_connection_factory import (
    Ms365GraphTenantConnectionIntegrationFactory,
)
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
)
from tests.unit.runtime.vendor_knowledge.test_tenant_connection_document_store import (
    ConditionalInMemoryDocumentStore,
)

pytestmark = pytest.mark.unit

_SECRET = "graph-client-secret"
_CREDENTIAL_REF = "secrets/tenant-1/msgraph"


class _FakeGraphBackend:
    pass


class _RecordingSecretsStore:
    def __init__(self, secret: str | None) -> None:
        self.secret = secret
        self.calls: list[str] = []

    def get_secret(self, path: str, *, version: str | None = None) -> str:
        del version
        self.calls.append(path)
        if self.secret is None:
            raise KeyError("missing")
        return self.secret

    def put_secret(self, path: str, value: str) -> None:
        del path, value

    def delete_secret(self, path: str) -> None:
        del path


def _kwargs(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "tenant_id": "tenant-1",
        "connection_ref": "graph-connection",
        "provider_id": MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
        "integration_kind": IntegrationCategory.COLLABORATION_SUITE,
        "credential_ref": _CREDENTIAL_REF,
        "credential": _SECRET,
        "secret_free_config": {"client_id": "graph-client-id"},
    }
    payload.update(overrides)
    return payload


def _integration_factory(
    captured_config: list[object] | None = None,
) -> Ms365GraphTenantConnectionIntegrationFactory:
    def build(config: object) -> Ms365GraphCollaborationSuiteIntegration:
        if captured_config is not None:
            captured_config.append(config)
        return Ms365GraphCollaborationSuiteIntegration.from_client(
            _FakeGraphBackend(),
            enabled=True,
        )

    return Ms365GraphTenantConnectionIntegrationFactory(runtime_builder=build)


def _connection() -> TenantConnection:
    now = datetime(2026, 8, 9, tzinfo=UTC)
    return TenantConnection(
        connection_ref="graph-connection",
        tenant_id="tenant-1",
        provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        safe_display_name="Microsoft Graph",
        administrative_status=TenantConnectionAdministrativeStatus.ACTIVE,
        credential_ref=_CREDENTIAL_REF,
        validated_secret_free_config={"client_id": "graph-client-id"},
        configuration_version=1,
        created_at=now,
        updated_at=now,
    )


def test_factory_builds_existing_graph_runtime_from_durable_secret() -> None:
    captured: list[object] = []

    integration = _integration_factory(captured).create_integration(**_kwargs())

    assert isinstance(integration, Ms365GraphCollaborationSuiteIntegration)
    config = captured[0]
    assert config.tenant_id == "tenant-1"
    assert config.client_id == "graph-client-id"
    assert config.client_secret == _SECRET


@pytest.mark.parametrize("credential", ["", "   ", None])
def test_factory_rejects_missing_credential_without_echoing_secret(
    credential: object,
) -> None:
    with pytest.raises(ValueError, match="credential"):
        _integration_factory().create_integration(**_kwargs(credential=credential))


def test_factory_rejects_malformed_secret_free_configuration_safely() -> None:
    with pytest.raises(ValueError, match="unsupported fields") as exc_info:
        _integration_factory().create_integration(
            **_kwargs(secret_free_config={"client_secret": _SECRET})
        )

    assert _SECRET not in str(exc_info.value)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("provider_id", "other-provider", "provider_id"),
        (
            "integration_kind",
            IntegrationCategory.ISSUE_TRACKER,
            "integration_kind",
        ),
    ],
)
def test_factory_rejects_wrong_identity_without_echoing_secret(
    field: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message) as exc_info:
        _integration_factory().create_integration(**_kwargs(**{field: value}))

    assert _SECRET not in str(exc_info.value)


def test_factory_normalizes_runtime_failure_without_echoing_secret() -> None:
    def fail_with_secret(config: object) -> Ms365GraphCollaborationSuiteIntegration:
        raise RuntimeError(f"runtime failed: {config.client_secret}")

    factory = Ms365GraphTenantConnectionIntegrationFactory(
        runtime_builder=fail_with_secret,
    )
    with pytest.raises(IntegrationConfigurationError) as exc_info:
        factory.create_integration(**_kwargs())

    assert _SECRET not in str(exc_info.value)


@pytest.mark.integration
def test_restart_rehydrates_graph_from_credential_ref_without_manual_registration() -> None:
    store = ConditionalInMemoryDocumentStore()
    repository = DocumentStoreTenantConnectionRepository(store)
    repository.create(_connection())
    registry = KnowledgeConnectionRegistry()
    secrets = _RecordingSecretsStore(_SECRET)

    results = TenantConnectionRehydrator(
        repository=DocumentStoreTenantConnectionRepository(store),
        secrets_store=secrets,
        integration_factory=_integration_factory(),
        connection_registry=registry,
    ).rehydrate_tenant(tenant_id="tenant-1")

    assert results[0].status is TenantConnectionRehydrationStatus.REGISTERED
    assert secrets.calls == [_CREDENTIAL_REF]
    integration = registry.resolve(
        tenant_id="tenant-1",
        connection_ref="graph-connection",
        provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
    )
    assert isinstance(integration, Ms365GraphCollaborationSuiteIntegration)
    persisted = store.get(
        "vendor_knowledge_connections:tenant-1",
        "connection:graph-connection",
    )
    assert persisted is not None
    assert _SECRET not in str(persisted.data)
    assert "client_secret" not in str(persisted.data)
    assert "access_token" not in str(persisted.data)
