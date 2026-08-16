# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for tenant connection domain model, service and rehydration."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.runtime.vendor_knowledge.connections import KnowledgeConnectionRegistry
from intergrax.runtime.vendor_knowledge.tenant_connection_document_store import (
    DocumentStoreTenantConnectionRepository,
)
from intergrax.runtime.vendor_knowledge.tenant_connection_rehydration import (
    TenantConnectionRehydrationStatus,
    TenantConnectionRehydrator,
)
from intergrax.runtime.vendor_knowledge.tenant_connections import (
    SafeTenantConnectionV1,
    TenantConnection,
    TenantConnectionAdministrativeStatus,
    TenantConnectionAlreadyExists,
    TenantConnectionInvalidState,
    TenantConnectionNotFound,
    TenantConnectionService,
    TenantConnectionVersionConflict,
    to_safe_tenant_connection,
)
from tests.unit.runtime.vendor_knowledge._fakes import FakeConnectionIntegration
from tests.unit.runtime.vendor_knowledge.test_tenant_connection_document_store import (
    ConditionalInMemoryDocumentStore,
)


def _utc_now(offset_seconds: int = 0) -> datetime:
    return datetime.now(timezone.utc) + timedelta(seconds=offset_seconds)


def _connection(
    *,
    connection_ref: str = "conn-1",
    tenant_id: str = "tenant-1",
    provider_id: str = "ms365_graph",
    integration_kind: IntegrationCategory = IntegrationCategory.COLLABORATION_SUITE,
    configuration_version: int = 1,
    administrative_status: TenantConnectionAdministrativeStatus = (
        TenantConnectionAdministrativeStatus.ACTIVE
    ),
    safe_display_name: str = "Example connection",
    credential_ref: str = "cred-1",
    config: dict | None = None,
    created_at: datetime | None = None,
    updated_at: datetime | None = None,
    connected_principal_ref: str | None = None,
) -> TenantConnection:
    created = created_at or _utc_now()
    updated = updated_at or created
    return TenantConnection(
        connection_ref=connection_ref,
        tenant_id=tenant_id,
        provider_id=provider_id,
        integration_kind=integration_kind,
        safe_display_name=safe_display_name,
        administrative_status=administrative_status,
        credential_ref=credential_ref,
        validated_secret_free_config=config or {
            "token_endpoint": "https://auth.example.test/oauth2/token",
            "secret_version": "v1",
            "authentication_mode": "client_credentials",
        },
        configuration_version=configuration_version,
        created_at=created,
        updated_at=updated,
        connected_principal_ref=connected_principal_ref,
    )


@pytest.mark.unit
def test_valid_durable_connection() -> None:
    connection = _connection()
    assert connection.connection_ref == "conn-1"
    assert connection.configuration_version == 1


@pytest.mark.unit
@pytest.mark.parametrize(
    "connection_ref",
    ["", "-bad", "bad space", "a" * 129],
)
def test_invalid_connection_ref(connection_ref: str) -> None:
    with pytest.raises(ValidationError):
        _connection(connection_ref=connection_ref)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("tenant_id", ""),
        ("provider_id", ""),
        ("safe_display_name", ""),
        ("credential_ref", ""),
    ],
)
def test_blank_required_fields_rejected(field_name: str, value: str) -> None:
    with pytest.raises(ValidationError):
        _connection(**{field_name: value})


@pytest.mark.unit
def test_naive_timestamps_rejected() -> None:
    naive = datetime(2026, 1, 1, 12, 0, 0)
    with pytest.raises(ValidationError):
        _connection(created_at=naive, updated_at=naive)


@pytest.mark.unit
@pytest.mark.parametrize(
    "offset",
    [
        timezone(timedelta(hours=2)),
        timezone(timedelta(hours=-5)),
    ],
)
def test_non_utc_timestamps_rejected(offset: timezone) -> None:
    aware = datetime(2026, 8, 2, 12, 0, 0, tzinfo=offset)
    with pytest.raises(ValidationError):
        _connection(created_at=aware, updated_at=aware)


@pytest.mark.unit
def test_utc_plus_zero_offset_accepted() -> None:
    utc = datetime(2026, 8, 2, 12, 0, 0, tzinfo=timezone.utc)
    connection = _connection(created_at=utc, updated_at=utc)
    assert connection.created_at.utcoffset() == timedelta(0)


@pytest.mark.unit
def test_configuration_version_below_one_rejected() -> None:
    with pytest.raises(ValidationError):
        _connection(configuration_version=0)


@pytest.mark.unit
def test_unknown_fields_rejected() -> None:
    with pytest.raises(ValidationError):
        TenantConnection(
            connection_ref="conn-1",
            tenant_id="tenant-1",
            provider_id="example",
            integration_kind=IntegrationCategory.ISSUE_TRACKER,
            safe_display_name="Example",
            administrative_status=TenantConnectionAdministrativeStatus.ACTIVE,
            credential_ref="cred-1",
            validated_secret_free_config={},
            configuration_version=1,
            created_at=_utc_now(),
            updated_at=_utc_now(),
            unexpected_field="nope",
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "config",
    [
        {"api_key": "leak"},
        {"nested": {"password": "leak"}},
        {"items": [{"client_secret": "leak"}]},
    ],
)
def test_raw_secret_key_rejected(config: dict) -> None:
    with pytest.raises(ValidationError):
        _connection(config=config)


@pytest.mark.unit
def test_credential_bearing_url_rejected() -> None:
    with pytest.raises(ValidationError):
        _connection(config={"endpoint": "https://user:password@example.test/path"})


@pytest.mark.unit
def test_access_token_key_rejected() -> None:
    with pytest.raises(ValidationError):
        _connection(config={"access_token": "leak"})


@pytest.mark.unit
def test_nested_forbidden_suffix_rejected() -> None:
    with pytest.raises(ValidationError):
        _connection(config={"nested": {"client_secret": "leak"}})


@pytest.mark.unit
def test_safe_nested_mapping_accepted() -> None:
    connection = _connection(
        config={
            "token_endpoint": "https://auth.example.test/oauth2/token",
            "nested": {"region": "eu-west-1"},
        }
    )
    assert connection.validated_secret_free_config["nested"]["region"] == "eu-west-1"


@pytest.mark.unit
def test_safe_list_traversal_accepted() -> None:
    connection = _connection(
        config={
            "endpoints": [
                "https://api.example.test/v1",
                {"host": "backup.example.test"},
            ]
        }
    )
    assert len(connection.validated_secret_free_config["endpoints"]) == 2


@pytest.mark.unit
def test_url_without_credentials_accepted() -> None:
    connection = _connection(config={"endpoint": "https://example.test/item?page=1"})
    assert connection.validated_secret_free_config["endpoint"].startswith("https://")


@pytest.mark.unit
def test_url_secret_query_parameter_accepted() -> None:
    connection = _connection(
        config={"endpoint": "https://example.com/?access_token=abc"}
    )
    assert "access_token=abc" in connection.validated_secret_free_config["endpoint"]


@pytest.mark.unit
def test_secret_like_scalar_value_accepted() -> None:
    connection = _connection(config={"note": "sk-live-example"})
    assert connection.validated_secret_free_config["note"] == "sk-live-example"


@pytest.mark.unit
def test_secret_free_config_error_excludes_literal() -> None:
    with pytest.raises(ValidationError) as exc_info:
        _connection(config={"endpoint": "https://user:super-secret@example.test/path"})
    assert "super-secret" not in str(exc_info.value)


@pytest.mark.unit
def test_benign_token_endpoint_accepted() -> None:
    connection = _connection()
    assert connection.validated_secret_free_config["token_endpoint"].startswith("https://")


@pytest.mark.unit
def test_safe_projection_omits_secrets() -> None:
    connection = _connection(credential_ref="cred-secret")
    safe = to_safe_tenant_connection(connection)
    dumped = safe.model_dump()
    assert "credential_ref" not in dumped
    assert "validated_secret_free_config" not in dumped
    assert "cred-secret" not in str(dumped)
    assert isinstance(safe, SafeTenantConnectionV1)


@pytest.mark.unit
def test_service_create_active_version_one() -> None:
    store = ConditionalInMemoryDocumentStore()
    repo = DocumentStoreTenantConnectionRepository(store)
    service = TenantConnectionService(tenant_id="tenant-1", repository=repo)
    created = service.create(_connection())
    assert created.configuration_version == 1
    assert created.administrative_status is TenantConnectionAdministrativeStatus.ACTIVE


@pytest.mark.unit
def test_service_wrong_tenant_rejected() -> None:
    service = TenantConnectionService(
        tenant_id="tenant-1",
        repository=DocumentStoreTenantConnectionRepository(
            ConditionalInMemoryDocumentStore()
        ),
    )
    with pytest.raises(TenantConnectionInvalidState):
        service.create(_connection(tenant_id="tenant-2"))


@pytest.mark.unit
def test_service_duplicate_create() -> None:
    store = ConditionalInMemoryDocumentStore()
    repo = DocumentStoreTenantConnectionRepository(store)
    service = TenantConnectionService(tenant_id="tenant-1", repository=repo)
    service.create(_connection())
    with pytest.raises(TenantConnectionAlreadyExists):
        service.create(_connection(safe_display_name="Other"))


@pytest.mark.unit
def test_service_version_mismatch() -> None:
    store = ConditionalInMemoryDocumentStore()
    repo = DocumentStoreTenantConnectionRepository(store)
    service = TenantConnectionService(tenant_id="tenant-1", repository=repo)
    service.create(_connection())
    with pytest.raises(TenantConnectionVersionConflict):
        service.update(
            _connection(configuration_version=2, updated_at=_utc_now(10)),
            expected_configuration_version=0,
        )


@pytest.mark.unit
def test_service_immutable_identity_change_rejected() -> None:
    store = ConditionalInMemoryDocumentStore()
    repo = DocumentStoreTenantConnectionRepository(store)
    service = TenantConnectionService(tenant_id="tenant-1", repository=repo)
    service.create(_connection())
    with pytest.raises(TenantConnectionInvalidState):
        service.update(
            _connection(
                provider_id="changed",
                configuration_version=2,
                updated_at=_utc_now(10),
            ),
            expected_configuration_version=1,
        )


@pytest.mark.unit
def test_service_status_transitions_and_revoked_terminal() -> None:
    store = ConditionalInMemoryDocumentStore()
    repo = DocumentStoreTenantConnectionRepository(store)
    service = TenantConnectionService(tenant_id="tenant-1", repository=repo)
    base_created = _utc_now()
    service.create(_connection(created_at=base_created, updated_at=base_created))

    disabled = _connection(
        configuration_version=2,
        administrative_status=TenantConnectionAdministrativeStatus.DISABLED,
        updated_at=_utc_now(10),
        created_at=base_created,
    )
    service.update(disabled, expected_configuration_version=1)
    assert service.get("conn-1").administrative_status is (
        TenantConnectionAdministrativeStatus.DISABLED
    )

    reenabled = _connection(
        configuration_version=3,
        administrative_status=TenantConnectionAdministrativeStatus.ACTIVE,
        updated_at=_utc_now(20),
        created_at=base_created,
    )
    service.update(reenabled, expected_configuration_version=2)
    assert service.get("conn-1").administrative_status is (
        TenantConnectionAdministrativeStatus.ACTIVE
    )

    service.create(
        _connection(
            connection_ref="conn-2",
            created_at=_utc_now(1),
            updated_at=_utc_now(1),
        )
    )
    active_revoked = _connection(
        connection_ref="conn-2",
        configuration_version=2,
        administrative_status=TenantConnectionAdministrativeStatus.REVOKED,
        updated_at=_utc_now(25),
        created_at=_utc_now(1),
    )
    service.update(active_revoked, expected_configuration_version=1)
    assert service.get("conn-2").administrative_status is (
        TenantConnectionAdministrativeStatus.REVOKED
    )

    revoked = _connection(
        configuration_version=4,
        administrative_status=TenantConnectionAdministrativeStatus.REVOKED,
        updated_at=_utc_now(30),
        created_at=base_created,
    )
    service.update(revoked, expected_configuration_version=3)
    assert service.get("conn-1").administrative_status is (
        TenantConnectionAdministrativeStatus.REVOKED
    )

    with pytest.raises(TenantConnectionInvalidState):
        service.update(
            _connection(
                configuration_version=5,
                updated_at=_utc_now(40),
                created_at=base_created,
            ),
            expected_configuration_version=4,
        )


@pytest.mark.unit
def test_service_disabled_to_revoked() -> None:
    store = ConditionalInMemoryDocumentStore()
    repo = DocumentStoreTenantConnectionRepository(store)
    base_created = _utc_now()
    repo.create(
        _connection(
            administrative_status=TenantConnectionAdministrativeStatus.DISABLED,
            created_at=base_created,
            updated_at=base_created,
        )
    )
    service = TenantConnectionService(tenant_id="tenant-1", repository=repo)
    revoked = _connection(
        configuration_version=2,
        administrative_status=TenantConnectionAdministrativeStatus.REVOKED,
        updated_at=_utc_now(10),
        created_at=base_created,
    )
    service.update(revoked, expected_configuration_version=1)
    assert service.get("conn-1").administrative_status is (
        TenantConnectionAdministrativeStatus.REVOKED
    )


@pytest.mark.unit
def test_service_deterministic_safe_list() -> None:
    store = ConditionalInMemoryDocumentStore()
    repo = DocumentStoreTenantConnectionRepository(store)
    service = TenantConnectionService(tenant_id="tenant-1", repository=repo)
    for ref in ("conn-c", "conn-a", "conn-b"):
        service.create(_connection(connection_ref=ref))
    listed = service.list_safe()
    assert [item.connection_ref for item in listed] == ["conn-a", "conn-b", "conn-c"]
    assert all("credential_ref" not in item.model_dump() for item in listed)


class _RecordingSecretsStore:
    def __init__(self, *, secret: str | None = "secret-value") -> None:
        self.secret = secret
        self.calls: list[str] = []

    def get_secret(self, path: str, *, version: str | None = None) -> str:
        self.calls.append(path)
        if self.secret is None:
            raise KeyError("missing")
        return self.secret

    def put_secret(self, path: str, value: str) -> None:
        return None

    def delete_secret(self, path: str) -> None:
        return None


class _CountingFactory:
    def __init__(self, *, integration: object | None = None, fail: bool = False) -> None:
        self.calls: list[dict[str, object]] = []
        self.integration = integration
        self.fail = fail

    def create_integration(
        self,
        *,
        tenant_id: str,
        connection_ref: str,
        provider_id: str,
        integration_kind: IntegrationCategory,
        credential_ref: str,
        credential: str,
        secret_free_config: Mapping[str, object],
    ) -> object:
        self.calls.append(
            {
                "tenant_id": tenant_id,
                "connection_ref": connection_ref,
                "provider_id": provider_id,
                "integration_kind": integration_kind,
                "credential_ref": credential_ref,
                "credential": credential,
                "secret_free_config": secret_free_config,
            }
        )
        if self.fail:
            raise RuntimeError("factory failed")
        if self.integration is not None:
            return self.integration
        return FakeConnectionIntegration(provider_id=provider_id, integration_kind=integration_kind.value)


@pytest.mark.unit
def test_restart_rehydration_proof() -> None:
    store = ConditionalInMemoryDocumentStore()
    repo_a = DocumentStoreTenantConnectionRepository(store)
    service_a = TenantConnectionService(tenant_id="tenant-1", repository=repo_a)
    service_a.create(_connection(credential_ref="cred-path"))

    repo_b = DocumentStoreTenantConnectionRepository(store)
    registry = KnowledgeConnectionRegistry()
    secrets = _RecordingSecretsStore()
    integration = FakeConnectionIntegration()
    factory = _CountingFactory(integration=integration)
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
    assert len(secrets.calls) == 1
    assert secrets.calls[0] == "cred-path"
    assert len(factory.calls) == 1
    assert factory.calls[0]["credential_ref"] == "cred-path"
    resolved = registry.resolve(
        tenant_id="tenant-1",
        connection_ref="conn-1",
        provider_id="ms365_graph",
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
    )
    assert resolved is integration
    dumped = results[0].model_dump()
    assert "secret-value" not in str(dumped)
    assert "credential_ref" not in dumped["connection"]
    document = store.get(
        "vendor_knowledge_connections:tenant-1",
        "connection:conn-1",
    )
    assert document is not None
    assert "secret-value" not in str(document.data)


@pytest.mark.unit
def test_rehydration_opaque_credential_preserved() -> None:
    opaque = "  opaque-secret\n"
    store = ConditionalInMemoryDocumentStore()
    repo = DocumentStoreTenantConnectionRepository(store)
    repo.create(_connection(credential_ref="cred-path"))
    secrets = _RecordingSecretsStore(secret=opaque)
    factory = _CountingFactory()
    rehydrator = TenantConnectionRehydrator(
        repository=repo,
        secrets_store=secrets,
        integration_factory=factory,
        connection_registry=KnowledgeConnectionRegistry(),
    )
    results = rehydrator.rehydrate_tenant(tenant_id="tenant-1")
    assert results[0].status is TenantConnectionRehydrationStatus.REGISTERED
    assert len(factory.calls) == 1
    assert factory.calls[0]["credential"] == opaque
    dumped = results[0].model_dump()
    assert opaque not in str(dumped)
    document = store.get(
        "vendor_knowledge_connections:tenant-1",
        "connection:conn-1",
    )
    assert document is not None
    assert opaque not in str(document.data)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("secret", "expected_code"),
    [
        (None, "tenant_connection_secret_unavailable"),
        ("   ", "tenant_connection_secret_unavailable"),
    ],
)
def test_rehydration_secret_unavailable(secret: str | None, expected_code: str) -> None:
    store = ConditionalInMemoryDocumentStore()
    repo = DocumentStoreTenantConnectionRepository(store)
    repo.create(_connection())
    secrets = _RecordingSecretsStore(secret=secret)
    factory = _CountingFactory()
    registry = KnowledgeConnectionRegistry()
    rehydrator = TenantConnectionRehydrator(
        repository=repo,
        secrets_store=secrets,
        integration_factory=factory,
        connection_registry=registry,
    )
    results = rehydrator.rehydrate_tenant(tenant_id="tenant-1")
    assert results[0].status is TenantConnectionRehydrationStatus.UNAVAILABLE
    assert results[0].error_code == expected_code
    assert factory.calls == []


@pytest.mark.unit
def test_rehydration_factory_failure_unavailable() -> None:
    store = ConditionalInMemoryDocumentStore()
    repo = DocumentStoreTenantConnectionRepository(store)
    repo.create(_connection())
    rehydrator = TenantConnectionRehydrator(
        repository=repo,
        secrets_store=_RecordingSecretsStore(),
        integration_factory=_CountingFactory(fail=True),
        connection_registry=KnowledgeConnectionRegistry(),
    )
    results = rehydrator.rehydrate_tenant(tenant_id="tenant-1")
    assert results[0].status is TenantConnectionRehydrationStatus.UNAVAILABLE
    assert results[0].error_code == "tenant_connection_runtime_unavailable"


@pytest.mark.unit
def test_rehydration_registry_identity_rejection() -> None:
    store = ConditionalInMemoryDocumentStore()
    repo = DocumentStoreTenantConnectionRepository(store)
    repo.create(_connection())
    bad_integration = FakeConnectionIntegration(provider_id="wrong-provider")
    rehydrator = TenantConnectionRehydrator(
        repository=repo,
        secrets_store=_RecordingSecretsStore(),
        integration_factory=_CountingFactory(integration=bad_integration),
        connection_registry=KnowledgeConnectionRegistry(),
    )
    results = rehydrator.rehydrate_tenant(tenant_id="tenant-1")
    assert results[0].status is TenantConnectionRehydrationStatus.UNAVAILABLE
    assert results[0].error_code == "tenant_connection_runtime_unavailable"


@pytest.mark.unit
def test_rehydration_skipped_disabled_and_revoked() -> None:
    store = ConditionalInMemoryDocumentStore()
    repo = DocumentStoreTenantConnectionRepository(store)
    base = _utc_now()
    repo.create(
        _connection(
            connection_ref="disabled",
            administrative_status=TenantConnectionAdministrativeStatus.DISABLED,
            created_at=base,
            updated_at=base,
        )
    )
    repo.create(
        _connection(
            connection_ref="revoked",
            administrative_status=TenantConnectionAdministrativeStatus.REVOKED,
            created_at=base,
            updated_at=base,
        )
    )
    secrets = _RecordingSecretsStore()
    factory = _CountingFactory()
    rehydrator = TenantConnectionRehydrator(
        repository=repo,
        secrets_store=secrets,
        integration_factory=factory,
        connection_registry=KnowledgeConnectionRegistry(),
    )
    results = rehydrator.rehydrate_tenant(tenant_id="tenant-1")
    by_ref = {item.connection.connection_ref: item for item in results}
    assert by_ref["disabled"].status is TenantConnectionRehydrationStatus.SKIPPED_DISABLED
    assert by_ref["revoked"].status is TenantConnectionRehydrationStatus.SKIPPED_REVOKED
    assert secrets.calls == []
    assert factory.calls == []


@pytest.mark.unit
def test_one_unavailable_does_not_block_active() -> None:
    store = ConditionalInMemoryDocumentStore()
    repo = DocumentStoreTenantConnectionRepository(store)
    repo.create(_connection(connection_ref="bad", credential_ref="missing"))
    repo.create(_connection(connection_ref="good", credential_ref="good-cred"))
    secrets = _RecordingSecretsStore()
    secrets.secret = "ok"

    class _SelectiveSecrets(_RecordingSecretsStore):
        def get_secret(self, path: str, *, version: str | None = None) -> str:
            self.calls.append(path)
            if path == "missing":
                raise KeyError("missing")
            return "ok"

    factory = _CountingFactory()
    registry = KnowledgeConnectionRegistry()
    rehydrator = TenantConnectionRehydrator(
        repository=repo,
        secrets_store=_SelectiveSecrets(),
        integration_factory=factory,
        connection_registry=registry,
    )
    results = rehydrator.rehydrate_tenant(tenant_id="tenant-1")
    by_ref = {item.connection.connection_ref: item for item in results}
    assert by_ref["bad"].status is TenantConnectionRehydrationStatus.UNAVAILABLE
    assert by_ref["good"].status is TenantConnectionRehydrationStatus.REGISTERED
    assert len(factory.calls) == 1


@pytest.mark.unit
def test_service_get_not_found() -> None:
    service = TenantConnectionService(
        tenant_id="tenant-1",
        repository=DocumentStoreTenantConnectionRepository(
            ConditionalInMemoryDocumentStore()
        ),
    )
    with pytest.raises(TenantConnectionNotFound):
        service.get("missing")
