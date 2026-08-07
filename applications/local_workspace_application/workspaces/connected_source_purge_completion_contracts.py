# © Artur Czarnecki. All rights reserved.

"""Purge completion authorities: delivery-receipt ownership index and legacy migration gate.

Canonical delivery-receipt row keys are scoped only by workspace/source/delivery_id.
Purge cleanup and completion therefore dual-write a derived exact-ownership index
keyed by the five-part binding scope so enumeration never observes another binding's
receipt.

Recovery ownership indexes cover only new COMPLETE_OWNERSHIP writes. Index emptiness
alone cannot prove historical unindexed recovery rows are absent. Completion therefore
requires an explicit durable migration gate for the exact purge scope.

New connected-source bindings created under the ownership-complete schema generation
persist the gate as CLEARED (no pre-contract recovery rows can exist). Bindings that
predate that generation remain REQUIRED until a future deterministic migration clears
them. Missing gates are treated as REQUIRED (fail closed).
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from enum import StrEnum

from local_workspace_application.workspaces.connected_source_models import (
    ConnectedSourceDeliveryReceipt,
)
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.integrations.contracts.document_store import DocumentRecord

_RECEIPT_INDEX_SCHEMA = "lkw.connected_source_delivery_receipt_ownership_index.v1"
_RECEIPT_INDEX_ENTITY = "connected_source_delivery_receipt_ownership_index"
_MIGRATION_GATE_SCHEMA = "lkw.connected_source_recovery_migration_gate.v1"
_MIGRATION_GATE_ENTITY = "connected_source_recovery_migration_gate"
_OWNERSHIP_COMPLETE_SCHEMA_GENERATION = "ownership_complete_schema.v1"
_SHA256_LENGTH = 64
_NEW_BINDING_EVIDENCE_REVISION = f"{_OWNERSHIP_COMPLETE_SCHEMA_GENERATION}:new_binding"


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


def delivery_receipt_canonical_row_key(
    *,
    workspace_id: str,
    source_id: str,
    delivery_id: str,
) -> str:
    return f"{workspace_id}:{source_id}:{delivery_id}"


def delivery_receipt_canonical_partition(tenant_id: str) -> str:
    return (
        f"lkw.managed_workspace:"
        f"{_normalized(tenant_id, 'tenant_id')}:"
        f"connected_source_delivery_receipt"
    )


class ConnectedSourceDeliveryReceiptOwnershipIndexEntryV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    tenant_id: str = Field(min_length=1, max_length=512)
    workspace_id: str = Field(min_length=1, max_length=512)
    source_id: str = Field(min_length=1, max_length=512)
    indexed_source_binding_id: str = Field(min_length=1, max_length=512)
    knowledge_source_binding_ref: str = Field(min_length=1, max_length=512)
    delivery_id: str = Field(min_length=1, max_length=512)
    canonical_partition_key: str = Field(min_length=1, max_length=1024)
    canonical_row_key: str = Field(min_length=1, max_length=1024)
    indexed_at: datetime

    _validate_ids = field_validator(
        "tenant_id",
        "workspace_id",
        "source_id",
        "indexed_source_binding_id",
        "knowledge_source_binding_ref",
        "delivery_id",
        "canonical_partition_key",
        "canonical_row_key",
    )(
        lambda value, info: _normalized(value, info.field_name or "identifier")
    )
    _validate_indexed_at = field_validator("indexed_at")(
        lambda value, info: _utc(value, info.field_name or "indexed_at")
    )

    @classmethod
    def for_receipt(
        cls,
        receipt: ConnectedSourceDeliveryReceipt,
        *,
        indexed_at: datetime | None = None,
    ) -> ConnectedSourceDeliveryReceiptOwnershipIndexEntryV1:
        partition = delivery_receipt_canonical_partition(receipt.tenant_id)
        row_key = delivery_receipt_canonical_row_key(
            workspace_id=receipt.workspace_id,
            source_id=receipt.source_id,
            delivery_id=receipt.delivery_id,
        )
        return cls(
            tenant_id=receipt.tenant_id,
            workspace_id=receipt.workspace_id,
            source_id=receipt.source_id,
            indexed_source_binding_id=receipt.indexed_source_binding_id,
            knowledge_source_binding_ref=receipt.knowledge_source_binding_ref,
            delivery_id=receipt.delivery_id,
            canonical_partition_key=partition,
            canonical_row_key=row_key,
            indexed_at=indexed_at or receipt.created_at,
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
        delivery_digest = hashlib.sha256(self.delivery_id.encode("utf-8")).hexdigest()
        return f"owner:{scope}:{delivery_digest}"

    @property
    def scope_prefix(self) -> str:
        scope = ownership_scope_digest(
            tenant_id=self.tenant_id,
            workspace_id=self.workspace_id,
            source_id=self.source_id,
            indexed_source_binding_id=self.indexed_source_binding_id,
            knowledge_source_binding_ref=self.knowledge_source_binding_ref,
        )
        return f"owner:{scope}:"

    def to_document(self) -> DocumentRecord:
        return DocumentRecord(
            partition_key=delivery_receipt_ownership_index_partition(self.tenant_id),
            row_key=self.row_key,
            data={
                "schema_version": _RECEIPT_INDEX_SCHEMA,
                "entry": self.model_dump(mode="json"),
            },
        )


class ConnectedSourceDeliveryReceiptOwnershipPageV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    documents: tuple[DocumentRecord, ...] = ()
    orphan_index_entries: tuple[
        ConnectedSourceDeliveryReceiptOwnershipIndexEntryV1, ...
    ] = ()
    next_cursor: str | None = None


class ConnectedSourceDeliveryReceiptOwnershipIndexError(RuntimeError):
    def __init__(self, error_code: str) -> None:
        self.error_code = _normalized(error_code, "error_code")
        super().__init__(self.error_code)


def parse_delivery_receipt_ownership_index_entry(
    record: DocumentRecord,
) -> ConnectedSourceDeliveryReceiptOwnershipIndexEntryV1:
    if (
        record.data.get("schema_version") != _RECEIPT_INDEX_SCHEMA
        or not isinstance(record.data.get("entry"), dict)
    ):
        raise ConnectedSourceDeliveryReceiptOwnershipIndexError(
            "delivery_receipt_ownership_index_corrupt"
        )
    try:
        entry = ConnectedSourceDeliveryReceiptOwnershipIndexEntryV1.model_validate(
            record.data["entry"],
            strict=False,
        )
    except (TypeError, ValueError):
        raise ConnectedSourceDeliveryReceiptOwnershipIndexError(
            "delivery_receipt_ownership_index_corrupt"
        ) from None
    if (
        record.partition_key
        != delivery_receipt_ownership_index_partition(entry.tenant_id)
        or record.row_key != entry.row_key
    ):
        raise ConnectedSourceDeliveryReceiptOwnershipIndexError(
            "delivery_receipt_ownership_index_identity_mismatch"
        )
    return entry


def delivery_receipt_ownership_index_partition(tenant_id: str) -> str:
    return (
        f"lkw.managed_workspace:"
        f"{_normalized(tenant_id, 'tenant_id')}:{_RECEIPT_INDEX_ENTITY}"
    )


def delivery_receipt_ownership_scope_prefix(
    *,
    tenant_id: str,
    workspace_id: str,
    source_id: str,
    indexed_source_binding_id: str,
    knowledge_source_binding_ref: str,
) -> str:
    probe = ConnectedSourceDeliveryReceiptOwnershipIndexEntryV1(
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        source_id=source_id,
        indexed_source_binding_id=indexed_source_binding_id,
        knowledge_source_binding_ref=knowledge_source_binding_ref,
        delivery_id="scope-probe",
        canonical_partition_key="scope-probe",
        canonical_row_key="scope-probe",
        indexed_at=datetime.now(UTC),
    )
    return probe.scope_prefix


class ConnectedSourceRecoveryMigrationGateStatusV1(StrEnum):
    REQUIRED = "REQUIRED"
    CLEARED = "CLEARED"


class ConnectedSourceRecoveryMigrationGateV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    tenant_id: str = Field(min_length=1, max_length=512)
    workspace_id: str = Field(min_length=1, max_length=512)
    source_id: str = Field(min_length=1, max_length=512)
    indexed_source_binding_id: str = Field(min_length=1, max_length=512)
    knowledge_source_binding_ref: str = Field(min_length=1, max_length=512)
    status: ConnectedSourceRecoveryMigrationGateStatusV1
    schema_version: int = Field(default=1, ge=1, le=1)
    cleared_at: datetime | None = None
    evidence_revision: str | None = Field(default=None, min_length=1, max_length=256)

    _validate_ids = field_validator(
        "tenant_id",
        "workspace_id",
        "source_id",
        "indexed_source_binding_id",
        "knowledge_source_binding_ref",
    )(
        lambda value, info: _normalized(value, info.field_name or "identifier")
    )
    _validate_cleared_at = field_validator("cleared_at")(
        lambda value, info: None
        if value is None
        else _utc(value, info.field_name or "cleared_at")
    )
    _validate_evidence = field_validator("evidence_revision")(
        lambda value, info: None
        if value is None
        else _normalized(value, info.field_name or "evidence_revision")
    )

    @model_validator(mode="after")
    def _validate_status_fields(self) -> ConnectedSourceRecoveryMigrationGateV1:
        if self.status is ConnectedSourceRecoveryMigrationGateStatusV1.CLEARED:
            if self.cleared_at is None or self.evidence_revision is None:
                raise ValueError("migration_gate_cleared_requires_evidence")
        elif self.cleared_at is not None or self.evidence_revision is not None:
            raise ValueError("migration_gate_required_forbids_clearance_fields")
        return self

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
        return f"owner:{scope}:migration_gate"

    def to_document(self) -> DocumentRecord:
        return DocumentRecord(
            partition_key=recovery_migration_gate_partition(self.tenant_id),
            row_key=self.row_key,
            data={
                "schema_version": _MIGRATION_GATE_SCHEMA,
                "gate": self.model_dump(mode="json"),
            },
        )


class ConnectedSourceRecoveryMigrationGateError(RuntimeError):
    def __init__(self, error_code: str) -> None:
        self.error_code = _normalized(error_code, "error_code")
        super().__init__(self.error_code)


def recovery_migration_gate_partition(tenant_id: str) -> str:
    return (
        f"lkw.managed_workspace:"
        f"{_normalized(tenant_id, 'tenant_id')}:{_MIGRATION_GATE_ENTITY}"
    )


def parse_recovery_migration_gate(
    record: DocumentRecord,
) -> ConnectedSourceRecoveryMigrationGateV1:
    if (
        record.data.get("schema_version") != _MIGRATION_GATE_SCHEMA
        or not isinstance(record.data.get("gate"), dict)
    ):
        raise ConnectedSourceRecoveryMigrationGateError("migration_gate_corrupt")
    try:
        gate = ConnectedSourceRecoveryMigrationGateV1.model_validate(
            record.data["gate"],
            strict=False,
        )
    except (TypeError, ValueError):
        raise ConnectedSourceRecoveryMigrationGateError(
            "migration_gate_corrupt"
        ) from None
    if (
        record.partition_key != recovery_migration_gate_partition(gate.tenant_id)
        or record.row_key != gate.row_key
    ):
        raise ConnectedSourceRecoveryMigrationGateError(
            "migration_gate_identity_mismatch"
        )
    return gate


def migration_gate_for_new_ownership_complete_binding(
    *,
    tenant_id: str,
    workspace_id: str,
    source_id: str,
    indexed_source_binding_id: str,
    knowledge_source_binding_ref: str,
    cleared_at: datetime,
) -> ConnectedSourceRecoveryMigrationGateV1:
    """CLEARED gate for a binding created under ownership-complete schema generation."""
    return ConnectedSourceRecoveryMigrationGateV1(
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        source_id=source_id,
        indexed_source_binding_id=indexed_source_binding_id,
        knowledge_source_binding_ref=knowledge_source_binding_ref,
        status=ConnectedSourceRecoveryMigrationGateStatusV1.CLEARED,
        schema_version=1,
        cleared_at=cleared_at,
        evidence_revision=_NEW_BINDING_EVIDENCE_REVISION,
    )


def migration_gate_required(
    *,
    tenant_id: str,
    workspace_id: str,
    source_id: str,
    indexed_source_binding_id: str,
    knowledge_source_binding_ref: str,
) -> ConnectedSourceRecoveryMigrationGateV1:
    return ConnectedSourceRecoveryMigrationGateV1(
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        source_id=source_id,
        indexed_source_binding_id=indexed_source_binding_id,
        knowledge_source_binding_ref=knowledge_source_binding_ref,
        status=ConnectedSourceRecoveryMigrationGateStatusV1.REQUIRED,
        schema_version=1,
        cleared_at=None,
        evidence_revision=None,
    )
