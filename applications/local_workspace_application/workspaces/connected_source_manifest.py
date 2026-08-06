# © Artur Czarnecki. All rights reserved.

"""Bounded, single-record publication authority for connected-source pages."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.integrations.contracts.document_store import (
    ConditionalDocumentStore,
    DocumentRecord,
    DocumentStore,
)
from intergrax.runtime.vendor_knowledge.sync_publication_fence import (
    KnowledgeSyncPublicationFenceV1,
    KnowledgeSyncPublicationPermitV1,
)

if TYPE_CHECKING:
    from intergrax.runtime.vendor_knowledge.sync_models import KnowledgeSyncBatch


_MANIFEST_SCHEMA = "lkw.connected_source_materialization_manifest.v1"
_MANIFEST_ENTITY = "connected_source_materialization_manifest"
MAX_MANIFEST_ENTRY_COUNT = 1000
MAX_MANIFEST_SERIALIZED_BYTES = 1_048_576
_IDENTIFIER_RE = re.compile(r"^[^\x00-\x1f\x7f]{1,512}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _identifier(value: str, field_name: str) -> str:
    if not isinstance(value, str) or value != value.strip():
        raise ValueError(f"{field_name}_must_be_normalized")
    if _IDENTIFIER_RE.fullmatch(value) is None:
        raise ValueError(f"{field_name}_must_be_normalized")
    return value


def _sha256(value: str, field_name: str) -> str:
    cleaned = _identifier(value, field_name)
    if _SHA256_RE.fullmatch(cleaned) is None:
        raise ValueError(f"{field_name}_must_be_sha256")
    return cleaned


def _utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name}_must_be_timezone_aware")
    if value.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name}_must_be_utc")
    return value.astimezone(UTC)


def _canonical_json(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


class ConnectedSourceMaterializationManifestEntryV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    remote_id: str = Field(..., min_length=1, max_length=512)
    document_id: str = Field(..., min_length=1, max_length=512)
    materialization_generation: str = Field(..., min_length=1, max_length=512)
    content_hash: str = Field(..., min_length=1, max_length=512)

    _validate_ids = field_validator(
        "remote_id",
        "document_id",
        "materialization_generation",
        "content_hash",
    )(
        lambda value, info: _identifier(value, info.field_name or "identifier")
    )


class ConnectedSourceMaterializationManifestV1(BaseModel):
    """The complete query-visible authority for one bounded delivery page."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    tenant_id: str = Field(..., min_length=1, max_length=512)
    workspace_id: str = Field(..., min_length=1, max_length=512)
    source_id: str = Field(..., min_length=1, max_length=512)
    indexed_source_binding_id: str = Field(..., min_length=1, max_length=512)
    knowledge_source_binding_ref: str = Field(..., min_length=1, max_length=512)
    delivery_id: str = Field(..., min_length=1, max_length=512)
    materialization_sequence: int = Field(..., gt=0)
    binding_configuration_version: int = Field(..., ge=1)
    publication_fence_revision: int = Field(..., ge=1)
    publication_fence_token_fingerprint: str
    document_entries: tuple[ConnectedSourceMaterializationManifestEntryV1, ...]
    payload_fingerprint: str
    committed_at: datetime

    _validate_ids = field_validator(
        "tenant_id",
        "workspace_id",
        "source_id",
        "indexed_source_binding_id",
        "knowledge_source_binding_ref",
        "delivery_id",
    )(
        lambda value, info: _identifier(value, info.field_name or "identifier")
    )
    _validate_hashes = field_validator(
        "publication_fence_token_fingerprint",
        "payload_fingerprint",
    )(
        lambda value, info: _sha256(value, info.field_name or "fingerprint")
    )
    _validate_committed_at = field_validator("committed_at")(
        lambda value: _utc(value, "committed_at")
    )

    @field_validator("document_entries")
    @classmethod
    def _immutable_entries(
        cls,
        value: tuple[ConnectedSourceMaterializationManifestEntryV1, ...]
        | list[ConnectedSourceMaterializationManifestEntryV1],
    ) -> tuple[ConnectedSourceMaterializationManifestEntryV1, ...]:
        entries = tuple(value)
        if len(entries) > MAX_MANIFEST_ENTRY_COUNT:
            raise ValueError("connected_source_manifest_entry_count_exceeded")
        return entries

    @model_validator(mode="after")
    def _entry_invariants(
        self,
    ) -> ConnectedSourceMaterializationManifestV1:
        ordered = tuple(
            sorted(self.document_entries, key=lambda item: (item.remote_id, item.document_id))
        )
        if self.document_entries != ordered:
            raise ValueError("connected_source_manifest_entries_not_deterministic")
        remote_ids = [entry.remote_id for entry in self.document_entries]
        if len(remote_ids) != len(set(remote_ids)):
            raise ValueError("connected_source_manifest_remote_id_duplicate")
        document_ids = [entry.document_id for entry in self.document_entries]
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("connected_source_manifest_document_id_duplicate")
        if len(self.serialized_bytes()) > MAX_MANIFEST_SERIALIZED_BYTES:
            raise ValueError("connected_source_manifest_serialized_size_exceeded")
        return self

    def serialized_bytes(self) -> bytes:
        return _canonical_json(self.model_dump(mode="json"))


def materialization_manifest_payload_fingerprint(batch: KnowledgeSyncBatch) -> str:
    """Hash the bounded delivery payload without including lifecycle secrets."""
    payload = {
        "tenant_id": batch.tenant_id,
        "binding_id": batch.binding_id,
        "binding_configuration_version": batch.binding_configuration_version,
        "source": batch.source.model_dump(mode="json"),
        "mode": batch.mode.value,
        "delivery_id": batch.delivery_id,
        "envelopes": [
            envelope.model_dump(mode="json") for envelope in batch.envelopes
        ],
        "has_more": batch.has_more,
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def publication_fence_token_fingerprint(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class ManifestCommitStatus(StrEnum):
    COMMITTED = "committed"
    REPLAYED = "replayed"
    STALE = "stale"


class ConnectedSourceMaterializationManifestConflict(RuntimeError):
    """The current manifest cannot be reconciled with the candidate."""


class ConnectedSourceMaterializationManifestRepository:
    """CAS repository whose current row is the page visibility linearization point."""

    def __init__(self, document_store: DocumentStore) -> None:
        if not isinstance(document_store, ConditionalDocumentStore):
            raise TypeError("manifest repository requires ConditionalDocumentStore")
        self._store = document_store

    def get_current(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        source_id: str,
        indexed_source_binding_id: str,
    ) -> ConnectedSourceMaterializationManifestV1 | None:
        record = self._store.get(
            self._partition(tenant_id),
            self._row_key(
                workspace_id=workspace_id,
                source_id=source_id,
                indexed_source_binding_id=indexed_source_binding_id,
            ),
        )
        if record is None:
            return None
        return self._parse(
            record,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            source_id=source_id,
            indexed_source_binding_id=indexed_source_binding_id,
        )[1]

    def commit(
        self,
        manifest: ConnectedSourceMaterializationManifestV1,
        *,
        expected_fence: KnowledgeSyncPublicationFenceV1,
        publication_permit: KnowledgeSyncPublicationPermitV1,
        validate_publication: Callable[
            [KnowledgeSyncPublicationFenceV1, KnowledgeSyncPublicationPermitV1],
            None,
        ],
    ) -> ManifestCommitStatus:
        partition = self._partition(manifest.tenant_id)
        row_key = self._row_key(
            workspace_id=manifest.workspace_id,
            source_id=manifest.source_id,
            indexed_source_binding_id=manifest.indexed_source_binding_id,
        )
        while True:
            current_record = self._store.get(partition, row_key)
            current = (
                None
                if current_record is None
                else self._parse(
                    current_record,
                    tenant_id=manifest.tenant_id,
                    workspace_id=manifest.workspace_id,
                    source_id=manifest.source_id,
                    indexed_source_binding_id=manifest.indexed_source_binding_id,
                )[1]
            )
            if current is not None:
                if current.materialization_sequence > manifest.materialization_sequence:
                    return ManifestCommitStatus.STALE
                if current.materialization_sequence == manifest.materialization_sequence:
                    if current == manifest:
                        validate_publication(expected_fence, publication_permit)
                        return ManifestCommitStatus.REPLAYED
                    raise ConnectedSourceMaterializationManifestConflict(
                        "connected_source_manifest_sequence_conflict"
                    )
            validate_publication(expected_fence, publication_permit)
            candidate = self._record(manifest, partition=partition, row_key=row_key)
            if current_record is None:
                if self._store.put_if_absent(candidate):
                    return ManifestCommitStatus.COMMITTED
                continue
            if self._store.replace_if_match(
                expected=current_record,
                replacement=candidate,
            ):
                return ManifestCommitStatus.COMMITTED

    @staticmethod
    def _partition(tenant_id: str) -> str:
        return f"lkw.managed_workspace:{tenant_id}:{_MANIFEST_ENTITY}"

    @staticmethod
    def _row_key(
        *,
        workspace_id: str,
        source_id: str,
        indexed_source_binding_id: str,
    ) -> str:
        return f"{workspace_id}:{source_id}:{indexed_source_binding_id}:current"

    @classmethod
    def _record(
        cls,
        manifest: ConnectedSourceMaterializationManifestV1,
        *,
        partition: str,
        row_key: str,
    ) -> DocumentRecord:
        return DocumentRecord(
            partition_key=partition,
            row_key=row_key,
            data={
                "schema_version": _MANIFEST_SCHEMA,
                "manifest": manifest.model_dump(mode="json"),
            },
        )

    @classmethod
    def _parse(
        cls,
        record: DocumentRecord,
        *,
        tenant_id: str,
        workspace_id: str,
        source_id: str,
        indexed_source_binding_id: str,
    ) -> tuple[DocumentRecord, ConnectedSourceMaterializationManifestV1]:
        expected_partition = cls._partition(tenant_id)
        expected_row = cls._row_key(
            workspace_id=workspace_id,
            source_id=source_id,
            indexed_source_binding_id=indexed_source_binding_id,
        )
        if (
            record.partition_key != expected_partition
            or record.row_key != expected_row
            or record.data.get("schema_version") != _MANIFEST_SCHEMA
            or not isinstance(record.data.get("manifest"), dict)
        ):
            raise ConnectedSourceMaterializationManifestConflict(
                "connected_source_manifest_corrupt"
            )
        try:
            manifest = ConnectedSourceMaterializationManifestV1.model_validate(
                record.data["manifest"]
            )
        except (TypeError, ValueError):
            raise ConnectedSourceMaterializationManifestConflict(
                "connected_source_manifest_corrupt"
            ) from None
        if (
            manifest.tenant_id != tenant_id
            or manifest.workspace_id != workspace_id
            or manifest.source_id != source_id
            or manifest.indexed_source_binding_id != indexed_source_binding_id
        ):
            raise ConnectedSourceMaterializationManifestConflict(
                "connected_source_manifest_identity_conflict"
            )
        return record, manifest
