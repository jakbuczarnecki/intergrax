# © Artur Czarnecki. All rights reserved.

"""Derived exact-ownership index for connected-source recovery records.

New COMPLETE_OWNERSHIP index receipts, enqueue intents and delivery accounting
rows are dual-written into this index. Historical incomplete / legacy records
are intentionally not indexed; full historical migration is out of scope.
Purge completion therefore applies only after an explicit migration gate has
confirmed that no relevant unindexed legacy recovery rows remain for the
target ownership scope.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from enum import StrEnum

from local_workspace_application.workspaces.connected_source_models import (
    ConnectedSourceOperationDeliveryAccounting,
    ConnectedSourceSyncEnqueueIntent,
)
from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.integrations.contracts.document_store import DocumentRecord

_INDEX_SCHEMA = "lkw.connected_source_recovery_ownership_index.v1"
_INDEX_ENTITY = "connected_source_recovery_ownership_index"
_SHA256_LENGTH = 64


def _normalized(value: str, field_name: str) -> str:
    if not isinstance(value, str) or value != value.strip() or not value:
        raise ValueError(f"{field_name}_must_be_normalized")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError(f"{field_name}_must_be_normalized")
    return value


def _utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name}_must_be_timezone_aware")
    if value.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name}_must_be_utc")
    return value.astimezone(UTC)


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def ownership_scope_digest(
    *,
    tenant_id: str,
    workspace_id: str,
    source_id: str,
    indexed_source_binding_id: str,
    knowledge_source_binding_ref: str,
) -> str:
    return hashlib.sha256(
        _canonical_json(
            {
                "tenant_id": tenant_id,
                "workspace_id": workspace_id,
                "source_id": source_id,
                "indexed_source_binding_id": indexed_source_binding_id,
                "knowledge_source_binding_ref": knowledge_source_binding_ref,
            }
        )
    ).hexdigest()


def canonical_record_fingerprint(payload: object) -> str:
    if isinstance(payload, BaseModel):
        data = payload.model_dump(mode="json")
    elif isinstance(payload, dict):
        data = payload
    else:
        raise TypeError("canonical_record_fingerprint_unsupported")
    return hashlib.sha256(_canonical_json(data)).hexdigest()


class RecoveryRecordKindV1(StrEnum):
    INDEX_RECEIPT = "index_receipt"
    ENQUEUE_INTENT = "enqueue_intent"
    DELIVERY_ACCOUNTING = "delivery_accounting"


class ConnectedSourceRecoveryOwnershipIndexEntryV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    tenant_id: str = Field(min_length=1, max_length=512)
    workspace_id: str = Field(min_length=1, max_length=512)
    source_id: str = Field(min_length=1, max_length=512)
    indexed_source_binding_id: str = Field(min_length=1, max_length=512)
    knowledge_source_binding_ref: str = Field(min_length=1, max_length=512)
    record_kind: RecoveryRecordKindV1
    operation_id: str = Field(min_length=1, max_length=512)
    delivery_id: str | None = Field(default=None, min_length=1, max_length=512)
    document_id: str | None = Field(default=None, min_length=1, max_length=512)
    canonical_partition_key: str = Field(min_length=1, max_length=1024)
    canonical_row_key: str = Field(min_length=1, max_length=1024)
    canonical_fingerprint: str = Field(
        min_length=_SHA256_LENGTH,
        max_length=_SHA256_LENGTH,
    )
    indexed_at: datetime

    _validate_ids = field_validator(
        "tenant_id",
        "workspace_id",
        "source_id",
        "indexed_source_binding_id",
        "knowledge_source_binding_ref",
        "operation_id",
        "canonical_partition_key",
        "canonical_row_key",
    )(
        lambda value, info: _normalized(value, info.field_name or "identifier")
    )
    _validate_optional_ids = field_validator("delivery_id", "document_id")(
        lambda value, info: None
        if value is None
        else _normalized(value, info.field_name or "identifier")
    )

    @field_validator("canonical_fingerprint")
    @classmethod
    def _validate_fingerprint(cls, value: str) -> str:
        if len(value) != _SHA256_LENGTH or any(
            char not in "0123456789abcdef" for char in value
        ):
            raise ValueError("canonical_fingerprint_must_be_sha256")
        return value

    _validate_indexed_at = field_validator("indexed_at")(
        lambda value, info: _utc(value, info.field_name or "indexed_at")
    )

    @property
    def ownership_scope(self) -> tuple[str, str, str, str, str]:
        return (
            self.tenant_id,
            self.workspace_id,
            self.source_id,
            self.indexed_source_binding_id,
            self.knowledge_source_binding_ref,
        )

    @property
    def row_key(self) -> str:
        scope = ownership_scope_digest(
            tenant_id=self.tenant_id,
            workspace_id=self.workspace_id,
            source_id=self.source_id,
            indexed_source_binding_id=self.indexed_source_binding_id,
            knowledge_source_binding_ref=self.knowledge_source_binding_ref,
        )
        identity = hashlib.sha256(
            _canonical_json(
                {
                    "record_kind": self.record_kind.value,
                    "operation_id": self.operation_id,
                    "delivery_id": self.delivery_id or "",
                    "document_id": self.document_id or "",
                    "canonical_partition_key": self.canonical_partition_key,
                    "canonical_row_key": self.canonical_row_key,
                }
            )
        ).hexdigest()
        return f"owner:{scope}:{self.record_kind.value}:{identity}"

    @property
    def scope_prefix(self) -> str:
        scope = ownership_scope_digest(
            tenant_id=self.tenant_id,
            workspace_id=self.workspace_id,
            source_id=self.source_id,
            indexed_source_binding_id=self.indexed_source_binding_id,
            knowledge_source_binding_ref=self.knowledge_source_binding_ref,
        )
        return f"owner:{scope}:{self.record_kind.value}:"

    def to_document(self) -> DocumentRecord:
        return DocumentRecord(
            partition_key=recovery_ownership_index_partition(self.tenant_id),
            row_key=self.row_key,
            data={
                "schema_version": _INDEX_SCHEMA,
                "entry": self.model_dump(mode="json"),
            },
        )


class ConnectedSourceRecoveryOwnershipPageV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    index_entries: tuple[ConnectedSourceRecoveryOwnershipIndexEntryV1, ...] = ()
    orphan_index_entries: tuple[ConnectedSourceRecoveryOwnershipIndexEntryV1, ...] = ()
    next_cursor: str | None = None


class ConnectedSourceRecoveryOwnershipIndexError(RuntimeError):
    def __init__(self, error_code: str) -> None:
        self.error_code = _normalized(error_code, "error_code")
        super().__init__(self.error_code)


def parse_recovery_ownership_index_entry(
    record: DocumentRecord,
) -> ConnectedSourceRecoveryOwnershipIndexEntryV1:
    if (
        record.data.get("schema_version") != _INDEX_SCHEMA
        or not isinstance(record.data.get("entry"), dict)
    ):
        raise ConnectedSourceRecoveryOwnershipIndexError(
            "recovery_ownership_index_corrupt"
        )
    try:
        entry = ConnectedSourceRecoveryOwnershipIndexEntryV1.model_validate(
            record.data["entry"],
            strict=False,
        )
    except (TypeError, ValueError):
        raise ConnectedSourceRecoveryOwnershipIndexError(
            "recovery_ownership_index_corrupt"
        ) from None
    if (
        record.partition_key != recovery_ownership_index_partition(entry.tenant_id)
        or record.row_key != entry.row_key
    ):
        raise ConnectedSourceRecoveryOwnershipIndexError(
            "recovery_ownership_index_identity_mismatch"
        )
    return entry


def recovery_ownership_index_partition(tenant_id: str) -> str:
    return (
        f"lkw.managed_workspace:"
        f"{_normalized(tenant_id, 'tenant_id')}:{_INDEX_ENTITY}"
    )


def recovery_ownership_scope_prefix(
    *,
    tenant_id: str,
    workspace_id: str,
    source_id: str,
    indexed_source_binding_id: str,
    knowledge_source_binding_ref: str,
    record_kind: RecoveryRecordKindV1,
) -> str:
    probe = ConnectedSourceRecoveryOwnershipIndexEntryV1(
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        source_id=source_id,
        indexed_source_binding_id=indexed_source_binding_id,
        knowledge_source_binding_ref=knowledge_source_binding_ref,
        record_kind=record_kind,
        operation_id="scope-probe",
        canonical_partition_key="scope-probe",
        canonical_row_key="scope-probe",
        canonical_fingerprint="0" * _SHA256_LENGTH,
        indexed_at=datetime.now(UTC),
    )
    return probe.scope_prefix


def index_entry_for_enqueue_intent(
    intent: ConnectedSourceSyncEnqueueIntent,
    *,
    canonical_partition_key: str,
    canonical_row_key: str,
    indexed_at: datetime | None = None,
) -> ConnectedSourceRecoveryOwnershipIndexEntryV1:
    if (
        intent.ownership_classification != "COMPLETE_OWNERSHIP"
        or intent.indexed_source_binding_id is None
        or intent.knowledge_source_binding_ref is None
    ):
        raise ConnectedSourceRecoveryOwnershipIndexError(
            "recovery_ownership_index_incomplete"
        )
    return ConnectedSourceRecoveryOwnershipIndexEntryV1(
        tenant_id=intent.tenant_id,
        workspace_id=intent.workspace_id,
        source_id=intent.source_id,
        indexed_source_binding_id=intent.indexed_source_binding_id,
        knowledge_source_binding_ref=intent.knowledge_source_binding_ref,
        record_kind=RecoveryRecordKindV1.ENQUEUE_INTENT,
        operation_id=intent.operation_id,
        delivery_id=None,
        document_id=None,
        canonical_partition_key=canonical_partition_key,
        canonical_row_key=canonical_row_key,
        canonical_fingerprint=canonical_record_fingerprint(intent),
        indexed_at=indexed_at or _utc(intent.updated_at, "indexed_at"),
    )


def index_entry_for_delivery_accounting(
    accounting: ConnectedSourceOperationDeliveryAccounting,
    *,
    canonical_partition_key: str,
    canonical_row_key: str,
    indexed_at: datetime | None = None,
) -> ConnectedSourceRecoveryOwnershipIndexEntryV1:
    if (
        accounting.ownership_classification != "COMPLETE_OWNERSHIP"
        or accounting.workspace_id is None
        or accounting.source_id is None
        or accounting.indexed_source_binding_id is None
        or accounting.knowledge_source_binding_ref is None
    ):
        raise ConnectedSourceRecoveryOwnershipIndexError(
            "recovery_ownership_index_incomplete"
        )
    workspace_id = accounting.workspace_id
    source_id = accounting.source_id
    binding_id = accounting.indexed_source_binding_id
    binding_ref = accounting.knowledge_source_binding_ref
    return ConnectedSourceRecoveryOwnershipIndexEntryV1(
        tenant_id=accounting.tenant_id,
        workspace_id=workspace_id,
        source_id=source_id,
        indexed_source_binding_id=binding_id,
        knowledge_source_binding_ref=binding_ref,
        record_kind=RecoveryRecordKindV1.DELIVERY_ACCOUNTING,
        operation_id=accounting.operation_id,
        delivery_id=accounting.delivery_id,
        document_id=None,
        canonical_partition_key=canonical_partition_key,
        canonical_row_key=canonical_row_key,
        canonical_fingerprint=canonical_record_fingerprint(accounting),
        indexed_at=indexed_at or _utc(accounting.accounted_at, "indexed_at"),
    )


def index_entry_for_index_receipt(
    receipt: BaseModel,
    *,
    canonical_partition_key: str,
    canonical_row_key: str,
    indexed_at: datetime | None = None,
) -> ConnectedSourceRecoveryOwnershipIndexEntryV1:
    """Build an index row for a connected-source `_WorkspaceDocumentIndexReceipt`."""
    from typing import Any, cast

    payload = cast(Any, receipt)
    ownership = payload.materialization_ownership
    if ownership is None:
        raise ConnectedSourceRecoveryOwnershipIndexError(
            "recovery_ownership_index_incomplete"
        )
    binding_id = ownership.indexed_source_binding_id
    binding_ref = ownership.knowledge_source_binding_ref
    if not isinstance(binding_id, str) or not isinstance(binding_ref, str):
        raise ConnectedSourceRecoveryOwnershipIndexError(
            "recovery_ownership_index_incomplete"
        )
    return ConnectedSourceRecoveryOwnershipIndexEntryV1(
        tenant_id=str(payload.tenant_id),
        workspace_id=str(payload.workspace_id),
        source_id=str(payload.source_id),
        indexed_source_binding_id=binding_id,
        knowledge_source_binding_ref=binding_ref,
        record_kind=RecoveryRecordKindV1.INDEX_RECEIPT,
        operation_id=str(payload.operation_id),
        delivery_id=ownership.delivery_id,
        document_id=str(payload.document_id),
        canonical_partition_key=canonical_partition_key,
        canonical_row_key=canonical_row_key,
        canonical_fingerprint=canonical_record_fingerprint(receipt),
        indexed_at=indexed_at or _utc(payload.created_at, "indexed_at"),
    )
