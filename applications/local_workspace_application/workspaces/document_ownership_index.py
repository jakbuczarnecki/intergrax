# © Artur Czarnecki. All rights reserved.

"""Derived exact-ownership index for connected-source document references."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from enum import StrEnum

from local_workspace_application.workspaces.materialization_visibility import (
    KnowledgeMaterializationOwnershipModeV1,
)
from local_workspace_application.workspaces.models import WorkspaceDocumentReference
from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.integrations.contracts.document_store import DocumentRecord

_INDEX_SCHEMA = "lkw.workspace_document_ownership_index.v1"
_INDEX_ENTITY = "document_ownership_index"
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


def reference_fingerprint(reference: WorkspaceDocumentReference) -> str:
    return hashlib.sha256(
        _canonical_json(reference.model_dump(mode="json"))
    ).hexdigest()


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


class DocumentOwnershipIndexClassificationV1(StrEnum):
    COMPLETE_OWNERSHIP = "COMPLETE_OWNERSHIP"
    LEGACY_MIGRATION_REQUIRED = "LEGACY_MIGRATION_REQUIRED"
    LEGACY_NON_CONNECTED = "LEGACY_NON_CONNECTED"


class WorkspaceDocumentOwnershipIndexEntryV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    tenant_id: str = Field(min_length=1, max_length=512)
    workspace_id: str = Field(min_length=1, max_length=512)
    source_id: str = Field(min_length=1, max_length=512)
    indexed_source_binding_id: str = Field(min_length=1, max_length=512)
    knowledge_source_binding_ref: str = Field(min_length=1, max_length=512)
    document_id: str = Field(min_length=1, max_length=512)
    reference_fingerprint: str = Field(
        min_length=_SHA256_LENGTH,
        max_length=_SHA256_LENGTH,
    )
    indexed_at: datetime
    classification: DocumentOwnershipIndexClassificationV1 = (
        DocumentOwnershipIndexClassificationV1.COMPLETE_OWNERSHIP
    )

    _validate_ids = field_validator(
        "tenant_id",
        "workspace_id",
        "source_id",
        "indexed_source_binding_id",
        "knowledge_source_binding_ref",
        "document_id",
    )(
        lambda value, info: _normalized(value, info.field_name or "identifier")
    )

    @field_validator("reference_fingerprint")
    @classmethod
    def _validate_fingerprint(cls, value: str) -> str:
        if len(value) != _SHA256_LENGTH or any(
            char not in "0123456789abcdef" for char in value
        ):
            raise ValueError("reference_fingerprint_must_be_sha256")
        return value

    _validate_indexed_at = field_validator("indexed_at")(
        lambda value, info: _utc(value, info.field_name or "indexed_at")
    )

    @classmethod
    def for_reference(
        cls,
        reference: WorkspaceDocumentReference,
    ) -> WorkspaceDocumentOwnershipIndexEntryV1:
        ownership = reference.materialization_ownership
        if ownership is None:
            raise ValueError("document_reference_ownership_missing")
        if ownership.ownership_mode is not KnowledgeMaterializationOwnershipModeV1.CONNECTED_SOURCE:
            raise ValueError("document_reference_is_not_connected_source")
        assert ownership.indexed_source_binding_id is not None
        assert ownership.knowledge_source_binding_ref is not None
        return cls(
            tenant_id=reference.tenant_id,
            workspace_id=reference.workspace_id,
            source_id=reference.source_id,
            indexed_source_binding_id=ownership.indexed_source_binding_id,
            knowledge_source_binding_ref=ownership.knowledge_source_binding_ref,
            document_id=reference.document_id,
            reference_fingerprint=reference_fingerprint(reference),
            indexed_at=reference.indexed_at,
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
        document_digest = hashlib.sha256(self.document_id.encode("utf-8")).hexdigest()
        return f"owner:{scope}:{document_digest}"

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
            partition_key=f"lkw.managed_workspace:{self.tenant_id}:{_INDEX_ENTITY}",
            row_key=self.row_key,
            data={
                "schema_version": _INDEX_SCHEMA,
                "entry": self.model_dump(mode="json"),
            },
        )


class DocumentReferenceOwnershipPageV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    references: tuple[WorkspaceDocumentReference, ...] = ()
    orphan_index_entries: tuple[WorkspaceDocumentOwnershipIndexEntryV1, ...] = ()
    next_cursor: str | None = None


class DocumentOwnershipIndexError(RuntimeError):
    def __init__(self, error_code: str) -> None:
        self.error_code = _normalized(error_code, "error_code")
        super().__init__(self.error_code)


def parse_index_entry(
    record: DocumentRecord,
) -> WorkspaceDocumentOwnershipIndexEntryV1:
    if (
        record.data.get("schema_version") != _INDEX_SCHEMA
        or not isinstance(record.data.get("entry"), dict)
    ):
        raise DocumentOwnershipIndexError("document_ownership_index_corrupt")
    try:
        entry = WorkspaceDocumentOwnershipIndexEntryV1.model_validate(
            record.data["entry"],
            strict=False,
        )
    except (TypeError, ValueError):
        raise DocumentOwnershipIndexError("document_ownership_index_corrupt") from None
    if (
        record.partition_key
        != f"lkw.managed_workspace:{entry.tenant_id}:{_INDEX_ENTITY}"
        or record.row_key != entry.row_key
    ):
        raise DocumentOwnershipIndexError("document_ownership_index_identity_mismatch")
    return entry


def ownership_index_partition(tenant_id: str) -> str:
    return f"lkw.managed_workspace:{_normalized(tenant_id, 'tenant_id')}:{_INDEX_ENTITY}"
