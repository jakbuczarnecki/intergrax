# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""DocumentStore-backed repository for durable tenant connections."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from intergrax.integrations.contracts.document_store import (
    ConditionalDocumentStore,
    DocumentRecord,
)
from intergrax.utils import attribute_access
from intergrax.runtime.vendor_knowledge.tenant_connections import (
    TenantConnection,
    TenantConnectionAdministrativeStatus,
    TenantConnectionAlreadyExists,
    TenantConnectionCorruptRecord,
    TenantConnectionNotFound,
    TenantConnectionVersionConflict,
    _assert_secret_free_config,
)

_PARTITION_PREFIX = "vendor_knowledge_connections"
_ROW_PREFIX = "connection"
_MAX_LIST_SCAN = 10_000
_DOCUMENT_QUERY_PAGE_LIMIT = 5_000


def connection_partition_key(tenant_id: str) -> str:
    cleaned = tenant_id.strip()
    if not cleaned:
        raise ValueError("tenant_id must be a non-empty string")
    return f"{_PARTITION_PREFIX}:{cleaned}"


def connection_row_key(connection_ref: str) -> str:
    cleaned = connection_ref.strip()
    if not cleaned:
        raise ValueError("connection_ref must be a non-empty string")
    return f"{_ROW_PREFIX}:{cleaned}"


def connection_to_document(connection: TenantConnection) -> DocumentRecord:
    data = connection.model_dump(mode="json")
    _assert_secret_free_config(
        data.get("validated_secret_free_config", {}),
        field_name="validated_secret_free_config",
    )
    return DocumentRecord(
        partition_key=connection_partition_key(connection.tenant_id),
        row_key=connection_row_key(connection.connection_ref),
        data=data,
        ttl_seconds=None,
    )


def connection_from_document(document: DocumentRecord) -> TenantConnection:
    data: dict[str, Any] = dict(document.data)
    tenant_id = data.get("tenant_id")
    connection_ref = data.get("connection_ref")
    if not isinstance(tenant_id, str) or not tenant_id.strip():
        raise TenantConnectionCorruptRecord("connection document tenant_id is invalid")
    if not isinstance(connection_ref, str) or not connection_ref.strip():
        raise TenantConnectionCorruptRecord("connection document connection_ref is invalid")

    expected_partition = connection_partition_key(tenant_id)
    if document.partition_key != expected_partition:
        raise TenantConnectionCorruptRecord(
            "connection document partition does not match tenant"
        )

    expected_row = connection_row_key(connection_ref)
    if document.row_key != expected_row:
        raise TenantConnectionCorruptRecord(
            "connection document row key does not match connection_ref"
        )

    try:
        config_raw = data.get("validated_secret_free_config")
        if isinstance(config_raw, dict):
            _assert_secret_free_config(
                config_raw,
                field_name="validated_secret_free_config",
            )
        return TenantConnection.model_validate(data)
    except TenantConnectionCorruptRecord:
        raise
    except (ValidationError, ValueError, TypeError) as exc:
        raise TenantConnectionCorruptRecord(
            "connection document payload is invalid"
        ) from exc


class DocumentStoreTenantConnectionRepository:
    """Production repository mapping tenant connections onto ConditionalDocumentStore."""

    def __init__(self, document_store: ConditionalDocumentStore) -> None:
        if not isinstance(document_store, ConditionalDocumentStore):
            raise TypeError("document_store must implement ConditionalDocumentStore")
        self._document_store = document_store

    def create(self, connection: TenantConnection) -> None:
        document = connection_to_document(connection)
        if not self._document_store.put_if_absent(document):
            raise TenantConnectionAlreadyExists("tenant connection already exists")

    def get(
        self,
        *,
        tenant_id: str,
        connection_ref: str,
    ) -> TenantConnection | None:
        cleaned_tenant = tenant_id.strip()
        cleaned_ref = connection_ref.strip()
        if not cleaned_tenant:
            raise ValueError("tenant_id must be a non-empty string")
        if not cleaned_ref:
            raise ValueError("connection_ref must be a non-empty string")

        document = self._document_store.get(
            connection_partition_key(cleaned_tenant),
            connection_row_key(cleaned_ref),
        )
        if document is None:
            return None
        connection = connection_from_document(document)
        if connection.tenant_id != cleaned_tenant:
            raise TenantConnectionCorruptRecord(
                "connection document tenant does not match lookup tenant"
            )
        return connection

    def update(
        self,
        connection: TenantConnection,
        *,
        expected_configuration_version: int,
    ) -> None:
        partition_key = connection_partition_key(connection.tenant_id)
        row_key = connection_row_key(connection.connection_ref)
        document = self._document_store.get(partition_key, row_key)
        if document is None:
            raise TenantConnectionNotFound("tenant connection was not found")

        current = connection_from_document(document)
        if current.configuration_version != expected_configuration_version:
            raise TenantConnectionVersionConflict(
                "tenant connection configuration version conflict"
            )
        if connection.configuration_version != expected_configuration_version + 1:
            raise TenantConnectionVersionConflict(
                "tenant connection configuration version conflict"
            )
        if (
            current.connection_ref != connection.connection_ref
            or current.tenant_id != connection.tenant_id
            or current.provider_id != connection.provider_id
            or current.integration_kind != connection.integration_kind
            or current.created_at != connection.created_at
        ):
            raise TenantConnectionVersionConflict(
                "tenant connection identity fields are immutable"
            )

        replacement = connection_to_document(connection)
        if not self._document_store.replace_if_match(
            expected=document,
            replacement=replacement,
        ):
            raise TenantConnectionVersionConflict(
                "tenant connection configuration version conflict"
            )

    def list(
        self,
        *,
        tenant_id: str,
        limit: int = 100,
        administrative_status: TenantConnectionAdministrativeStatus | None = None,
    ) -> tuple[TenantConnection, ...]:
        cleaned_tenant = tenant_id.strip()
        if not cleaned_tenant:
            raise ValueError("tenant_id must be a non-empty string")
        if limit < 1 or limit > 1000:
            raise ValueError("limit must be between 1 and 1000 inclusive")

        partition_key = connection_partition_key(cleaned_tenant)
        row_key_prefix = f"{_ROW_PREFIX}:"
        documents: list[DocumentRecord] = []
        cursor: str | None = None
        while len(documents) <= _MAX_LIST_SCAN:
            page_limit = min(
                _DOCUMENT_QUERY_PAGE_LIMIT,
                _MAX_LIST_SCAN + 1 - len(documents),
            )
            if cursor is None:
                page = self._document_store.query(
                    partition_key,
                    limit=page_limit,
                    row_key_prefix=row_key_prefix,
                )
            else:
                page = self._document_store.query(
                    partition_key,
                    limit=page_limit,
                    row_key_prefix=row_key_prefix,
                    cursor=cursor,
                )
            documents.extend(page.documents)
            if attribute_access.optional(page, "total", len(page.documents)) > _MAX_LIST_SCAN:
                raise TenantConnectionCorruptRecord("connection list exceeds scan limit")
            next_cursor = attribute_access.optional(page, "next_cursor", None)
            if next_cursor is None:
                break
            cursor = next_cursor
        if len(documents) > _MAX_LIST_SCAN:
            raise TenantConnectionCorruptRecord("connection list exceeds scan limit")

        connections: list[TenantConnection] = []
        for document in documents:
            connection = connection_from_document(document)
            if connection.tenant_id != cleaned_tenant:
                raise TenantConnectionCorruptRecord(
                    "connection document tenant does not match lookup tenant"
                )
            if administrative_status is not None:
                if connection.administrative_status is not administrative_status:
                    continue
            connections.append(connection)
        connections.sort(key=lambda item: item.connection_ref)
        return tuple(connections[:limit])
