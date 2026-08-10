from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.relational_store.databricks.integration import (
    DatabricksRelationalStoreIntegration,
)
from intergrax.integrations.providers.relational_store.databricks.tenant_connection_factory import (
    DatabricksTenantConnectionIntegrationFactory,
)
from intergrax.runtime.vendor_knowledge.connections import KnowledgeConnectionRegistry
from intergrax.runtime.vendor_knowledge.provider_composition import (
    build_default_vendor_knowledge_connection_factory_registry,
)
from intergrax.runtime.vendor_knowledge.tenant_connection_rehydration import (
    TenantConnectionRehydrator,
    TenantConnectionRehydrationStatus,
)
from intergrax.runtime.vendor_knowledge.tenant_connections import (
    TenantConnection,
    TenantConnectionAdministrativeStatus,
)


class _FakeCursor:
    def __init__(self) -> None:
        self.executed: list[str] = []

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def execute(self, statement: str, *_args: object) -> None:
        self.executed.append(statement)


class _FakeConnection:
    def __init__(self) -> None:
        self.cursor_instance = _FakeCursor()

    def cursor(self) -> _FakeCursor:
        return self.cursor_instance


class _Repository:
    def __init__(self, connection: TenantConnection) -> None:
        self.connection = connection

    def list(
        self,
        *,
        tenant_id: str,
        limit: int = 100,
        administrative_status: TenantConnectionAdministrativeStatus | None = None,
    ) -> tuple[TenantConnection, ...]:
        if tenant_id != self.connection.tenant_id or limit < 1:
            return ()
        if (
            administrative_status is not None
            and administrative_status is not self.connection.administrative_status
        ):
            return ()
        return (self.connection,)


class _SecretsStore:
    def __init__(self, token: str) -> None:
        self.token = token
        self.requested_refs: list[str] = []

    def get_secret(self, path: str, *, version: str | None = None) -> str:
        self.requested_refs.append(path)
        return self.token


def _tenant_connection() -> TenantConnection:
    timestamp = datetime(2026, 8, 10, 6, 0, tzinfo=timezone.utc)
    return TenantConnection(
        connection_ref="databricks-main",
        tenant_id="tenant-1",
        provider_id="databricks",
        integration_kind=IntegrationCategory.RELATIONAL_STORE,
        safe_display_name="Databricks SQL Warehouse",
        administrative_status=TenantConnectionAdministrativeStatus.ACTIVE,
        credential_ref="secret/databricks/main",
        validated_secret_free_config={
            "host": "adb.example.net",
            "http_path": "/sql/1.0/warehouses/main",
            "catalog": "main",
            "tenant_schema": "analytics",
        },
        configuration_version=1,
        created_at=timestamp,
        updated_at=timestamp,
    )


@pytest.mark.unit
def test_factory_rejects_wrong_canonical_identity() -> None:
    factory = DatabricksTenantConnectionIntegrationFactory()
    kwargs: dict[str, Any] = {
        "tenant_id": "tenant-1",
        "connection_ref": "connection-1",
        "provider_id": "databricks",
        "integration_kind": IntegrationCategory.RELATIONAL_STORE,
        "credential_ref": "secret-1",
        "credential": "dapi-token",
        "secret_free_config": {
            "host": "adb.example.net",
            "http_path": "/sql/1.0/warehouses/main",
        },
    }

    with pytest.raises(ValueError, match="provider_id"):
        factory.create_integration(**{**kwargs, "provider_id": "databricks_sql"})
    with pytest.raises(ValueError, match="integration_kind"):
        factory.create_integration(
            **{**kwargs, "integration_kind": IntegrationCategory.DOCUMENT_STORE}
        )


@pytest.mark.unit
def test_factory_rejects_secret_in_persisted_config_without_echoing_it() -> None:
    factory = DatabricksTenantConnectionIntegrationFactory()
    token = "dapi-super-secret"

    with pytest.raises(ValueError) as exc_info:
        factory.create_integration(
            tenant_id="tenant-1",
            connection_ref="connection-1",
            provider_id="databricks",
            integration_kind=IntegrationCategory.RELATIONAL_STORE,
            credential_ref="secret-1",
            credential=token,
            secret_free_config={
                "host": "adb.example.net",
                "http_path": "/sql/1.0/warehouses/main",
                "access_token": token,
            },
        )

    assert token not in str(exc_info.value)


@pytest.mark.unit
def test_default_registry_rehydrates_databricks_after_restart_without_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = "dapi-restart-secret"
    for name in (
        "INTERGRAX_DATABRICKS_HOST",
        "INTERGRAX_DATABRICKS_HTTP_PATH",
        "INTERGRAX_DATABRICKS_TOKEN",
        "INTERGRAX_DATABRICKS_CATALOG",
        "INTERGRAX_DATABRICKS_SCHEMA",
    ):
        monkeypatch.delenv(name, raising=False)

    connection = _tenant_connection()
    repository = _Repository(connection)
    secrets_store = _SecretsStore(token)
    runtime_connection = _FakeConnection()
    registry = KnowledgeConnectionRegistry()
    rehydrator = TenantConnectionRehydrator(
        repository=repository,
        secrets_store=secrets_store,
        integration_factory=build_default_vendor_knowledge_connection_factory_registry(
            databricks_connection_factory=lambda: runtime_connection,
        ),
        connection_registry=registry,
    )

    results = rehydrator.rehydrate_tenant(tenant_id="tenant-1")

    assert results[0].status is TenantConnectionRehydrationStatus.REGISTERED
    assert secrets_store.requested_refs == ["secret/databricks/main"]
    resolved = registry.resolve(
        tenant_id="tenant-1",
        connection_ref="databricks-main",
        provider_id="databricks",
        integration_kind=IntegrationCategory.RELATIONAL_STORE,
    )
    assert isinstance(resolved, DatabricksRelationalStoreIntegration)
    assert resolved.client is not None
    assert resolved.client.config.access_token == token
    assert runtime_connection.cursor_instance.executed == [
        "USE CATALOG main",
        "USE SCHEMA analytics",
    ]
    assert token not in str(connection.model_dump())
    assert "credential_ref" not in results[0].connection.model_dump()
