# © Artur Czarnecki. All rights reserved.

"""Bounded, single-record publication authority for connected-source pages."""

from __future__ import annotations

import hashlib
import json
import re
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
    DocumentStoreKnowledgeSyncPublicationFenceRepository,
    KnowledgeSyncCommittedPublicationV1,
    KnowledgeSyncPublicationCommitStatus,
    KnowledgeSyncPublicationFencePort,
    KnowledgeSyncPublicationFenceV1,
    KnowledgeSyncPublicationPermitV1,
)

if TYPE_CHECKING:
    from intergrax.runtime.vendor_knowledge.sync_models import KnowledgeSyncBatch


_IMMUTABLE_MANIFEST_SCHEMA = "lkw.connected_source_materialization_manifest.immutable.v1"
_PUBLICATION_COMMIT_SCHEMA = "lkw.connected_source_publication_commit.v1"
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

    @property
    def manifest_fingerprint(self) -> str:
        return hashlib.sha256(self.serialized_bytes()).hexdigest()

    @property
    def manifest_id(self) -> str:
        identity = {
            "tenant_id": self.tenant_id,
            "workspace_id": self.workspace_id,
            "source_id": self.source_id,
            "indexed_source_binding_id": self.indexed_source_binding_id,
            "knowledge_source_binding_ref": self.knowledge_source_binding_ref,
            "delivery_id": self.delivery_id,
            "materialization_sequence": self.materialization_sequence,
            "manifest_fingerprint": self.manifest_fingerprint,
        }
        return hashlib.sha256(_canonical_json(identity)).hexdigest()


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
    """Immutable manifest store coordinated by the publication fence authority."""

    def __init__(
        self,
        document_store: DocumentStore,
        *,
        publication_authority: KnowledgeSyncPublicationFencePort | None = None,
    ) -> None:
        if not isinstance(document_store, ConditionalDocumentStore):
            raise TypeError("manifest repository requires ConditionalDocumentStore")
        self._store = document_store
        self._publication_authority = publication_authority or (
            DocumentStoreKnowledgeSyncPublicationFenceRepository(document_store)
        )

    def get_current(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        source_id: str,
        indexed_source_binding_id: str,
    ) -> ConnectedSourceMaterializationManifestV1 | None:
        manifests = self.list_committed(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            source_id=source_id,
            indexed_source_binding_id=indexed_source_binding_id,
        )
        return max(
            manifests,
            key=lambda manifest: manifest.materialization_sequence,
            default=None,
        )

    def get_committed_for_delivery(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        source_id: str,
        indexed_source_binding_id: str,
        delivery_id: str,
    ) -> ConnectedSourceMaterializationManifestV1 | None:
        descriptor = self._get_commit_descriptor(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            source_id=source_id,
            indexed_source_binding_id=indexed_source_binding_id,
            delivery_id=delivery_id,
        )
        if descriptor is None:
            return None
        return self._load_immutable(descriptor)

    def get_prepared_for_delivery(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        source_id: str,
        indexed_source_binding_id: str,
        delivery_id: str,
    ) -> ConnectedSourceMaterializationManifestV1 | None:
        prefix = self._scope_prefix(
            workspace_id=workspace_id,
            source_id=source_id,
            indexed_source_binding_id=indexed_source_binding_id,
        )
        result = self._store.query(
            self._immutable_partition(tenant_id),
            limit=5000,
            row_key_prefix=prefix,
        )
        candidates: list[ConnectedSourceMaterializationManifestV1] = []
        for record in result.documents:
            if not str(record.row_key).startswith(prefix + "manifest:"):
                continue
            try:
                manifest = self._parse_immutable(
                    record,
                    tenant_id=tenant_id,
                    workspace_id=workspace_id,
                    source_id=source_id,
                    indexed_source_binding_id=indexed_source_binding_id,
                )
            except ConnectedSourceMaterializationManifestConflict:
                continue
            if manifest.delivery_id == delivery_id:
                candidates.append(manifest)
        if not candidates:
            return None
        return max(candidates, key=lambda manifest: manifest.materialization_sequence)

    def list_committed(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        source_id: str,
        indexed_source_binding_id: str,
    ) -> tuple[ConnectedSourceMaterializationManifestV1, ...]:
        prefix = self._scope_prefix(
            workspace_id=workspace_id,
            source_id=source_id,
            indexed_source_binding_id=indexed_source_binding_id,
        )
        descriptors: dict[str, KnowledgeSyncCommittedPublicationV1] = {}
        result = self._store.query(
            self._commit_partition(tenant_id),
            limit=5000,
            row_key_prefix=prefix,
        )
        for record in result.documents:
            if not str(record.row_key).startswith(prefix + "commit:"):
                continue
            descriptor = self._parse_commit_descriptor(record)
            if (
                descriptor.workspace_id == workspace_id
                and descriptor.source_id == source_id
                and descriptor.indexed_source_binding_id == indexed_source_binding_id
            ):
                descriptors[descriptor.delivery_id] = descriptor
        current = self._publication_authority.list_committed_publications(
            tenant_id=tenant_id
        )
        for descriptor in current:
            if (
                descriptor.workspace_id == workspace_id
                and descriptor.source_id == source_id
                and descriptor.indexed_source_binding_id == indexed_source_binding_id
            ):
                descriptors.setdefault(descriptor.delivery_id, descriptor)
        manifests = [self._load_immutable(descriptor) for descriptor in descriptors.values()]
        return tuple(sorted(manifests, key=lambda item: item.materialization_sequence))

    def commit(
        self,
        manifest: ConnectedSourceMaterializationManifestV1,
        *,
        expected_fence: KnowledgeSyncPublicationFenceV1,
        publication_permit: KnowledgeSyncPublicationPermitV1,
        publication_authority: KnowledgeSyncPublicationFencePort | None = None,
        validate_publication: object | None = None,
    ) -> ManifestCommitStatus:
        # Kept as a source-compatible no-op; visibility is authorized only by the
        # same-record publication CAS below.
        _ = validate_publication
        authority = publication_authority or self._publication_authority
        prepared = self.get_prepared_for_delivery(
            tenant_id=manifest.tenant_id,
            workspace_id=manifest.workspace_id,
            source_id=manifest.source_id,
            indexed_source_binding_id=manifest.indexed_source_binding_id,
            delivery_id=manifest.delivery_id,
        )
        if prepared is not None and prepared != manifest:
            raise ConnectedSourceMaterializationManifestConflict(
                "connected_source_manifest_delivery_conflict"
            )
        self._put_immutable(manifest)
        existing = self._get_commit_descriptor(
            tenant_id=manifest.tenant_id,
            workspace_id=manifest.workspace_id,
            source_id=manifest.source_id,
            indexed_source_binding_id=manifest.indexed_source_binding_id,
            delivery_id=manifest.delivery_id,
        )
        candidate_descriptor = self._descriptor(manifest)
        if existing is not None and existing != candidate_descriptor:
            raise ConnectedSourceMaterializationManifestConflict(
                "connected_source_manifest_delivery_conflict"
            )
        for committed in self._list_commit_descriptors(
            tenant_id=manifest.tenant_id,
            workspace_id=manifest.workspace_id,
            source_id=manifest.source_id,
            indexed_source_binding_id=manifest.indexed_source_binding_id,
        ):
            if (
                committed.materialization_sequence
                == manifest.materialization_sequence
                and committed != candidate_descriptor
            ):
                raise ConnectedSourceMaterializationManifestConflict(
                    "connected_source_manifest_sequence_conflict"
                )
        result = authority.commit_publication_under_permit(
            expected_fence=expected_fence,
            publication_permit=publication_permit,
            publication_descriptor=candidate_descriptor,
        )
        if result.status is KnowledgeSyncPublicationCommitStatus.STALE:
            self._put_commit_descriptor(result.descriptor)
            return ManifestCommitStatus.STALE
        self._put_commit_descriptor(result.descriptor)
        if result.status is KnowledgeSyncPublicationCommitStatus.REPLAYED:
            return ManifestCommitStatus.REPLAYED
        return ManifestCommitStatus.COMMITTED

    @staticmethod
    def _immutable_partition(tenant_id: str) -> str:
        return f"lkw.managed_workspace:{tenant_id}:{_MANIFEST_ENTITY}:immutable"

    @staticmethod
    def _commit_partition(tenant_id: str) -> str:
        return f"lkw.managed_workspace:{tenant_id}:{_MANIFEST_ENTITY}:commit"

    @staticmethod
    def _scope_prefix(
        *,
        workspace_id: str,
        source_id: str,
        indexed_source_binding_id: str,
    ) -> str:
        return f"{workspace_id}:{source_id}:{indexed_source_binding_id}:"

    @classmethod
    def _immutable_row_key(cls, manifest: ConnectedSourceMaterializationManifestV1) -> str:
        return (
            cls._scope_prefix(
                workspace_id=manifest.workspace_id,
                source_id=manifest.source_id,
                indexed_source_binding_id=manifest.indexed_source_binding_id,
            )
            + f"manifest:{manifest.delivery_id}:{manifest.materialization_sequence}:{manifest.manifest_id}"
        )

    @classmethod
    def _commit_row_key(cls, descriptor: KnowledgeSyncCommittedPublicationV1) -> str:
        return (
            cls._scope_prefix(
                workspace_id=descriptor.workspace_id,
                source_id=descriptor.source_id,
                indexed_source_binding_id=descriptor.indexed_source_binding_id,
            )
            + f"commit:{descriptor.delivery_id}"
        )

    @classmethod
    def _immutable_record(
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
                "schema_version": _IMMUTABLE_MANIFEST_SCHEMA,
                "manifest_id": manifest.manifest_id,
                "manifest_fingerprint": manifest.manifest_fingerprint,
                "manifest": manifest.model_dump(mode="json"),
            },
        )

    @classmethod
    def _commit_record(
        cls,
        descriptor: KnowledgeSyncCommittedPublicationV1,
    ) -> DocumentRecord:
        return DocumentRecord(
            partition_key=cls._commit_partition(descriptor.tenant_id),
            row_key=cls._commit_row_key(descriptor),
            data={
                "schema_version": _PUBLICATION_COMMIT_SCHEMA,
                "descriptor": descriptor.model_dump(mode="json"),
            },
        )

    def _put_immutable(self, manifest: ConnectedSourceMaterializationManifestV1) -> None:
        record = self._immutable_record(
            manifest,
            partition=self._immutable_partition(manifest.tenant_id),
            row_key=self._immutable_row_key(manifest),
        )
        existing = self._store.get(record.partition_key, record.row_key)
        if existing is not None:
            parsed = self._parse_immutable(
                existing,
                tenant_id=manifest.tenant_id,
                workspace_id=manifest.workspace_id,
                source_id=manifest.source_id,
                indexed_source_binding_id=manifest.indexed_source_binding_id,
            )
            if parsed != manifest:
                raise ConnectedSourceMaterializationManifestConflict(
                    "connected_source_manifest_immutable_conflict"
                )
            return
        if not self._store.put_if_absent(record):
            retry = self._store.get(record.partition_key, record.row_key)
            if retry is None:
                raise ConnectedSourceMaterializationManifestConflict(
                    "connected_source_manifest_immutable_write_conflict"
                )
            parsed = self._parse_immutable(
                retry,
                tenant_id=manifest.tenant_id,
                workspace_id=manifest.workspace_id,
                source_id=manifest.source_id,
                indexed_source_binding_id=manifest.indexed_source_binding_id,
            )
            if parsed != manifest:
                raise ConnectedSourceMaterializationManifestConflict(
                    "connected_source_manifest_immutable_conflict"
                )

    def _parse_immutable(
        self,
        record: DocumentRecord,
        *,
        tenant_id: str,
        workspace_id: str,
        source_id: str,
        indexed_source_binding_id: str,
    ) -> ConnectedSourceMaterializationManifestV1:
        expected_partition = self._immutable_partition(tenant_id)
        expected_prefix = self._scope_prefix(
            workspace_id=workspace_id,
            source_id=source_id,
            indexed_source_binding_id=indexed_source_binding_id,
        )
        if (
            record.partition_key != expected_partition
            or not str(record.row_key).startswith(expected_prefix + "manifest:")
            or record.data.get("schema_version") != _IMMUTABLE_MANIFEST_SCHEMA
            or not isinstance(record.data.get("manifest"), dict)
            or not isinstance(record.data.get("manifest_id"), str)
            or not isinstance(record.data.get("manifest_fingerprint"), str)
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
            or record.row_key != self._immutable_row_key(manifest)
            or record.data.get("manifest_id") != manifest.manifest_id
            or record.data.get("manifest_fingerprint") != manifest.manifest_fingerprint
        ):
            raise ConnectedSourceMaterializationManifestConflict(
                "connected_source_manifest_identity_conflict"
            )
        return manifest

    @staticmethod
    def _descriptor(
        manifest: ConnectedSourceMaterializationManifestV1,
    ) -> KnowledgeSyncCommittedPublicationV1:
        return KnowledgeSyncCommittedPublicationV1(
            tenant_id=manifest.tenant_id,
            binding_id=manifest.knowledge_source_binding_ref,
            workspace_id=manifest.workspace_id,
            source_id=manifest.source_id,
            indexed_source_binding_id=manifest.indexed_source_binding_id,
            delivery_id=manifest.delivery_id,
            materialization_sequence=manifest.materialization_sequence,
            manifest_id=manifest.manifest_id,
            manifest_fingerprint=manifest.manifest_fingerprint,
            committed_at=manifest.committed_at,
        )

    def _get_commit_descriptor(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        source_id: str,
        indexed_source_binding_id: str,
        delivery_id: str,
    ) -> KnowledgeSyncCommittedPublicationV1 | None:
        record = self._store.get(
            self._commit_partition(tenant_id),
            self._commit_row_key_values(
                workspace_id=workspace_id,
                source_id=source_id,
                indexed_source_binding_id=indexed_source_binding_id,
                delivery_id=delivery_id,
            ),
        )
        if record is not None:
            descriptor = self._parse_commit_descriptor(record)
            if (
                descriptor.workspace_id != workspace_id
                or descriptor.source_id != source_id
                or descriptor.indexed_source_binding_id != indexed_source_binding_id
                or descriptor.delivery_id != delivery_id
            ):
                raise ConnectedSourceMaterializationManifestConflict(
                    "connected_source_manifest_commit_identity_conflict"
                )
            return descriptor
        for descriptor in self._publication_authority.list_committed_publications(
            tenant_id=tenant_id
        ):
            if (
                descriptor.workspace_id == workspace_id
                and descriptor.source_id == source_id
                and descriptor.indexed_source_binding_id == indexed_source_binding_id
                and descriptor.delivery_id == delivery_id
            ):
                return descriptor
        return None

    def _list_commit_descriptors(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        source_id: str,
        indexed_source_binding_id: str,
    ) -> tuple[KnowledgeSyncCommittedPublicationV1, ...]:
        prefix = self._scope_prefix(
            workspace_id=workspace_id,
            source_id=source_id,
            indexed_source_binding_id=indexed_source_binding_id,
        )
        result = self._store.query(
            self._commit_partition(tenant_id),
            limit=5000,
            row_key_prefix=prefix,
        )
        return tuple(
            self._parse_commit_descriptor(record)
            for record in result.documents
            if str(record.row_key).startswith(prefix + "commit:")
        )

    def _put_commit_descriptor(
        self,
        descriptor: KnowledgeSyncCommittedPublicationV1,
    ) -> None:
        record = self._commit_record(descriptor)
        existing = self._store.get(record.partition_key, record.row_key)
        if existing is not None:
            if self._parse_commit_descriptor(existing) != descriptor:
                raise ConnectedSourceMaterializationManifestConflict(
                    "connected_source_manifest_delivery_conflict"
                )
            return
        if not self._store.put_if_absent(record):
            retry = self._store.get(record.partition_key, record.row_key)
            if retry is None or self._parse_commit_descriptor(retry) != descriptor:
                raise ConnectedSourceMaterializationManifestConflict(
                    "connected_source_manifest_commit_record_conflict"
                )

    @staticmethod
    def _parse_commit_descriptor(
        record: DocumentRecord,
    ) -> KnowledgeSyncCommittedPublicationV1:
        if (
            record.data.get("schema_version") != _PUBLICATION_COMMIT_SCHEMA
            or not isinstance(record.data.get("descriptor"), dict)
        ):
            raise ConnectedSourceMaterializationManifestConflict(
                "connected_source_manifest_commit_record_corrupt"
            )
        try:
            return KnowledgeSyncCommittedPublicationV1.model_validate(
                record.data["descriptor"]
            )
        except (TypeError, ValueError):
            raise ConnectedSourceMaterializationManifestConflict(
                "connected_source_manifest_commit_record_corrupt"
            ) from None

    def _load_immutable(
        self,
        descriptor: KnowledgeSyncCommittedPublicationV1,
    ) -> ConnectedSourceMaterializationManifestV1:
        record = self._store.get(
            self._immutable_partition(descriptor.tenant_id),
            self._immutable_row_key_values(
                workspace_id=descriptor.workspace_id,
                source_id=descriptor.source_id,
                indexed_source_binding_id=descriptor.indexed_source_binding_id,
                delivery_id=descriptor.delivery_id,
                materialization_sequence=descriptor.materialization_sequence,
                manifest_id=descriptor.manifest_id,
            ),
        )
        if record is None:
            raise ConnectedSourceMaterializationManifestConflict(
                "connected_source_manifest_immutable_missing"
            )
        manifest = self._parse_immutable(
            record,
            tenant_id=descriptor.tenant_id,
            workspace_id=descriptor.workspace_id,
            source_id=descriptor.source_id,
            indexed_source_binding_id=descriptor.indexed_source_binding_id,
        )
        if (
            manifest.knowledge_source_binding_ref != descriptor.binding_id
            or manifest.delivery_id != descriptor.delivery_id
            or manifest.materialization_sequence != descriptor.materialization_sequence
            or manifest.manifest_id != descriptor.manifest_id
            or manifest.manifest_fingerprint != descriptor.manifest_fingerprint
        ):
            raise ConnectedSourceMaterializationManifestConflict(
                "connected_source_manifest_descriptor_mismatch"
            )
        return manifest

    @staticmethod
    def _commit_row_key_values(
        *,
        workspace_id: str,
        source_id: str,
        indexed_source_binding_id: str,
        delivery_id: str,
    ) -> str:
        return (
            f"{workspace_id}:{source_id}:{indexed_source_binding_id}:"
            f"commit:{delivery_id}"
        )

    @staticmethod
    def _immutable_row_key_values(
        *,
        workspace_id: str,
        source_id: str,
        indexed_source_binding_id: str,
        delivery_id: str,
        materialization_sequence: int,
        manifest_id: str,
    ) -> str:
        return (
            f"{workspace_id}:{source_id}:{indexed_source_binding_id}:"
            f"manifest:{delivery_id}:{materialization_sequence}:{manifest_id}"
        )
