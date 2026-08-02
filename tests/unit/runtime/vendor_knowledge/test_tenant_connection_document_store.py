# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for DocumentStore tenant connection repository."""

from __future__ import annotations

from typing import Optional

import pytest

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.contracts.document_store import DocumentQueryResult, DocumentRecord
from intergrax.runtime.vendor_knowledge.tenant_connection_document_store import (
    DocumentStoreTenantConnectionRepository,
    _MAX_LIST_SCAN,
    connection_from_document,
    connection_partition_key,
    connection_row_key,
    connection_to_document,
)
from intergrax.runtime.vendor_knowledge.tenant_connections import (
    TenantConnection,
    TenantConnectionAdministrativeStatus,
    TenantConnectionAlreadyExists,
    TenantConnectionCorruptRecord,
    TenantConnectionNotFound,
    TenantConnectionVersionConflict,
)
from tests.unit.runtime.vendor_knowledge._fakes import InMemoryDocumentStore


class ConditionalInMemoryDocumentStore(InMemoryDocumentStore):
    """In-memory ConditionalDocumentStore for tenant connection repository tests."""

    def put_if_absent(self, document: DocumentRecord) -> bool:
        key = (document.partition_key, document.row_key)
        if key in self._rows:
            return False
        self._rows[key] = document
        return True

    def replace_if_match(
        self,
        *,
        expected: DocumentRecord,
        replacement: DocumentRecord,
    ) -> bool:
        key = (expected.partition_key, expected.row_key)
        current = self._rows.get(key)
        if current is None:
            return False
        if current.partition_key != expected.partition_key or current.row_key != expected.row_key:
            return False
        if dict(current.data) != dict(expected.data):
            return False
        self._rows[key] = replacement
        return True

    def delete_if_match(self, *, expected: DocumentRecord) -> bool:
        key = (expected.partition_key, expected.row_key)
        current = self._rows.get(key)
        if current is None:
            return False
        if dict(current.data) != dict(expected.data):
            return False
        del self._rows[key]
        return True


def _now(offset_seconds: int = 0) -> str:
    from datetime import datetime, timedelta, timezone

    return (
        datetime.now(timezone.utc) + timedelta(seconds=offset_seconds)
    ).isoformat()


def _connection(
    *,
    connection_ref: str = "conn-1",
    tenant_id: str = "tenant-1",
    provider_id: str = "example",
    configuration_version: int = 1,
    administrative_status: TenantConnectionAdministrativeStatus = (
        TenantConnectionAdministrativeStatus.ACTIVE
    ),
    safe_display_name: str = "Example connection",
    credential_ref: str = "cred-1",
    config: dict | None = None,
    created_offset: int = 0,
    updated_offset: int = 0,
) -> TenantConnection:
    created = _now(created_offset)
    updated = _now(updated_offset)
    return TenantConnection(
        connection_ref=connection_ref,
        tenant_id=tenant_id,
        provider_id=provider_id,
        integration_kind=IntegrationCategory.ISSUE_TRACKER,
        safe_display_name=safe_display_name,
        administrative_status=administrative_status,
        credential_ref=credential_ref,
        validated_secret_free_config=config or {"token_endpoint": "https://auth.example.test"},
        configuration_version=configuration_version,
        created_at=created,
        updated_at=updated,
        connected_principal_ref=None,
    )


@pytest.mark.unit
def test_conditional_document_store_required() -> None:
    with pytest.raises(TypeError, match="ConditionalDocumentStore"):
        DocumentStoreTenantConnectionRepository(InMemoryDocumentStore())


@pytest.mark.unit
def test_atomic_create_and_get() -> None:
    store = ConditionalInMemoryDocumentStore()
    repo = DocumentStoreTenantConnectionRepository(store)
    connection = _connection()
    repo.create(connection)
    loaded = repo.get(tenant_id="tenant-1", connection_ref="conn-1")
    assert loaded == connection


@pytest.mark.unit
def test_duplicate_create_rejected() -> None:
    store = ConditionalInMemoryDocumentStore()
    repo = DocumentStoreTenantConnectionRepository(store)
    repo.create(_connection())
    with pytest.raises(TenantConnectionAlreadyExists):
        repo.create(_connection(safe_display_name="Other"))


@pytest.mark.unit
def test_cross_tenant_lookup_returns_none() -> None:
    store = ConditionalInMemoryDocumentStore()
    repo = DocumentStoreTenantConnectionRepository(store)
    repo.create(_connection(tenant_id="tenant-a", connection_ref="shared"))
    assert repo.get(tenant_id="tenant-a", connection_ref="shared") is not None
    assert repo.get(tenant_id="tenant-b", connection_ref="shared") is None


@pytest.mark.unit
def test_deterministic_keys() -> None:
    assert connection_partition_key("tenant-1") == "vendor_knowledge_connections:tenant-1"
    assert connection_row_key("conn-1") == "connection:conn-1"
    document = connection_to_document(_connection())
    assert document.partition_key == "vendor_knowledge_connections:tenant-1"
    assert document.row_key == "connection:conn-1"


@pytest.mark.unit
def test_cas_update_success() -> None:
    store = ConditionalInMemoryDocumentStore()
    repo = DocumentStoreTenantConnectionRepository(store)
    repo.create(_connection())
    updated = _connection(
        configuration_version=2,
        safe_display_name="Updated",
        administrative_status=TenantConnectionAdministrativeStatus.DISABLED,
        updated_offset=10,
    )
    repo.update(updated, expected_configuration_version=1)
    loaded = repo.get(tenant_id="tenant-1", connection_ref="conn-1")
    assert loaded is not None
    assert loaded.safe_display_name == "Updated"
    assert loaded.administrative_status is TenantConnectionAdministrativeStatus.DISABLED
    assert loaded.configuration_version == 2


@pytest.mark.unit
def test_stale_version_conflict() -> None:
    store = ConditionalInMemoryDocumentStore()
    repo = DocumentStoreTenantConnectionRepository(store)
    repo.create(_connection())
    with pytest.raises(TenantConnectionVersionConflict):
        repo.update(
            _connection(configuration_version=2, updated_offset=5),
            expected_configuration_version=0,
        )


@pytest.mark.unit
def test_skipped_configuration_version_conflict() -> None:
    store = ConditionalInMemoryDocumentStore()
    repo = DocumentStoreTenantConnectionRepository(store)
    repo.create(_connection())
    with pytest.raises(TenantConnectionVersionConflict):
        repo.update(
            _connection(configuration_version=3, updated_offset=5),
            expected_configuration_version=1,
        )


@pytest.mark.unit
def test_concurrent_replacement_conflict() -> None:
    class _StaleCasStore(ConditionalInMemoryDocumentStore):
        def replace_if_match(
            self,
            *,
            expected: DocumentRecord,
            replacement: DocumentRecord,
        ) -> bool:
            return False

    store = _StaleCasStore()
    repo = DocumentStoreTenantConnectionRepository(store)
    repo.create(_connection())
    with pytest.raises(TenantConnectionVersionConflict):
        repo.update(
            _connection(configuration_version=2, updated_offset=5),
            expected_configuration_version=1,
        )


@pytest.mark.unit
def test_immutable_identity_rejected() -> None:
    store = ConditionalInMemoryDocumentStore()
    repo = DocumentStoreTenantConnectionRepository(store)
    repo.create(_connection())
    with pytest.raises(TenantConnectionVersionConflict):
        repo.update(
            _connection(provider_id="changed", configuration_version=2, updated_offset=5),
            expected_configuration_version=1,
        )


@pytest.mark.unit
def test_deterministic_list_and_status_filter() -> None:
    store = ConditionalInMemoryDocumentStore(reverse_query=True)
    repo = DocumentStoreTenantConnectionRepository(store)
    repo.create(_connection(connection_ref="conn-c"))
    repo.create(
        _connection(
            connection_ref="conn-a",
            administrative_status=TenantConnectionAdministrativeStatus.DISABLED,
        )
    )
    repo.create(_connection(connection_ref="conn-b"))
    listed = repo.list(tenant_id="tenant-1")
    assert [item.connection_ref for item in listed] == ["conn-a", "conn-b", "conn-c"]
    disabled = repo.list(
        tenant_id="tenant-1",
        administrative_status=TenantConnectionAdministrativeStatus.DISABLED,
    )
    assert len(disabled) == 1
    assert disabled[0].connection_ref == "conn-a"


@pytest.mark.unit
@pytest.mark.parametrize("limit", [0, 1001])
def test_list_limit_validation(limit: int) -> None:
    repo = DocumentStoreTenantConnectionRepository(ConditionalInMemoryDocumentStore())
    with pytest.raises(ValueError, match="limit"):
        repo.list(tenant_id="tenant-1", limit=limit)


@pytest.mark.unit
def test_partition_mismatch_rejected() -> None:
    connection = _connection()
    document = connection_to_document(connection)
    bad = DocumentRecord(
        partition_key="vendor_knowledge_connections:tenant-other",
        row_key=document.row_key,
        data=dict(document.data),
    )
    with pytest.raises(TenantConnectionCorruptRecord):
        connection_from_document(bad)


@pytest.mark.unit
def test_row_key_mismatch_rejected() -> None:
    connection = _connection()
    document = connection_to_document(connection)
    bad = DocumentRecord(
        partition_key=document.partition_key,
        row_key="connection:other",
        data=dict(document.data),
    )
    with pytest.raises(TenantConnectionCorruptRecord):
        connection_from_document(bad)


@pytest.mark.unit
def test_invalid_payload_rejected() -> None:
    store = ConditionalInMemoryDocumentStore()
    store.put(
        DocumentRecord(
            partition_key="vendor_knowledge_connections:tenant-1",
            row_key="connection:conn-1",
            data={"connection_ref": "conn-1"},
        )
    )
    repo = DocumentStoreTenantConnectionRepository(store)
    with pytest.raises(TenantConnectionCorruptRecord):
        repo.get(tenant_id="tenant-1", connection_ref="conn-1")


@pytest.mark.unit
def test_secret_bearing_nested_config_rejected() -> None:
    with pytest.raises(ValueError, match="forbidden"):
        _connection(config={"nested": {"api_key": "leak"}})


@pytest.mark.unit
def test_corrupt_durable_secret_config_normalized() -> None:
    store = ConditionalInMemoryDocumentStore()
    secret_value = "must-not-escape"
    store.put(
        DocumentRecord(
            partition_key="vendor_knowledge_connections:tenant-1",
            row_key="connection:conn-1",
            data={
                "connection_ref": "conn-1",
                "tenant_id": "tenant-1",
                "provider_id": "example",
                "integration_kind": "issue_tracker",
                "safe_display_name": "Example connection",
                "administrative_status": "active",
                "credential_ref": "cred-1",
                "validated_secret_free_config": {
                    "nested": {"api_key": secret_value},
                },
                "configuration_version": 1,
                "created_at": _now(),
                "updated_at": _now(),
            },
        )
    )
    repo = DocumentStoreTenantConnectionRepository(store)
    with pytest.raises(TenantConnectionCorruptRecord) as exc_info:
        repo.get(tenant_id="tenant-1", connection_ref="conn-1")
    assert secret_value not in str(exc_info.value)
    assert exc_info.type is TenantConnectionCorruptRecord


@pytest.mark.unit
def test_scan_limit_enforced() -> None:
    class _OverCapStore:
        def get(self, partition_key: str, row_key: str) -> Optional[DocumentRecord]:
            return None

        def put(self, document: DocumentRecord) -> None:
            return None

        def delete(self, partition_key: str, row_key: str) -> None:
            return None

        def query(
            self,
            partition_key: str,
            *,
            limit: int = 100,
            row_key_prefix: Optional[str] = None,
        ) -> DocumentQueryResult:
            return DocumentQueryResult(documents=(), total=_MAX_LIST_SCAN + 1)

        def close(self) -> None:
            return None

        def put_if_absent(self, document: DocumentRecord) -> bool:
            return True

        def replace_if_match(
            self,
            *,
            expected: DocumentRecord,
            replacement: DocumentRecord,
        ) -> bool:
            return True

        def delete_if_match(self, *, expected: DocumentRecord) -> bool:
            return True

    repo = DocumentStoreTenantConnectionRepository(_OverCapStore())
    with pytest.raises(TenantConnectionCorruptRecord, match="scan limit"):
        repo.list(tenant_id="tenant-1")


@pytest.mark.unit
def test_no_raw_secret_value_persisted() -> None:
    document = connection_to_document(_connection())
    assert "api_key" not in document.data
    assert "password" not in document.data
    assert document.data["credential_ref"] == "cred-1"
    assert "token_endpoint" in document.data["validated_secret_free_config"]


@pytest.mark.unit
def test_update_missing_not_found() -> None:
    repo = DocumentStoreTenantConnectionRepository(ConditionalInMemoryDocumentStore())
    with pytest.raises(TenantConnectionNotFound):
        repo.update(
            _connection(configuration_version=2, updated_offset=5),
            expected_configuration_version=1,
        )
