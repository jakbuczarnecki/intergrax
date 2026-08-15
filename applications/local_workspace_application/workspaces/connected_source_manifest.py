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
from intergrax.runtime.vendor_knowledge.models import KnowledgeItemRevision
from intergrax.runtime.vendor_knowledge.sync_publication_fence import (
    DocumentStoreKnowledgeSyncPublicationFenceRepository,
    KnowledgeSyncCommittedPublicationV1,
    KnowledgeSyncCommittedPublicationV2,
    KnowledgeSyncPublicationCommitStatus,
    KnowledgeSyncPublicationFencePort,
    KnowledgeSyncPublicationFenceV1,
    KnowledgeSyncPublicationPermitV1,
)

if TYPE_CHECKING:
    from intergrax.runtime.vendor_knowledge.sync_models import KnowledgeSyncBatch


_IMMUTABLE_MANIFEST_SCHEMA = "lkw.connected_source_materialization_manifest.immutable.v1"
_DELIVERY_INDEX_SCHEMA = "lkw.connected_source_delivery_index.immutable.v1"
_REMOTE_CANDIDATE_SCHEMA = "lkw.connected_source_remote_candidate.immutable.v1"
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
    source_revision: KnowledgeItemRevision | None = None

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
    revoked_remote_ids: tuple[str, ...] = ()
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

    @field_validator("revoked_remote_ids")
    @classmethod
    def _immutable_revoked_remote_ids(
        cls,
        value: tuple[str, ...] | list[str],
    ) -> tuple[str, ...]:
        remote_ids = tuple(value)
        if len(remote_ids) > MAX_MANIFEST_ENTRY_COUNT:
            raise ValueError("connected_source_manifest_revocation_count_exceeded")
        normalized = tuple(sorted(_identifier(item, "revoked_remote_id") for item in remote_ids))
        if len(normalized) != len(set(normalized)):
            raise ValueError("connected_source_manifest_revocation_duplicate")
        if remote_ids != normalized:
            raise ValueError("connected_source_manifest_revocations_not_deterministic")
        return normalized

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
        if set(remote_ids).intersection(self.revoked_remote_ids):
            raise ValueError("connected_source_manifest_active_revocation_conflict")
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


class ConnectedSourceRemoteCandidateV1(BaseModel):
    """One immutable remote/version candidate, prepared before publication CAS."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    tenant_id: str = Field(..., min_length=1, max_length=512)
    workspace_id: str = Field(..., min_length=1, max_length=512)
    source_id: str = Field(..., min_length=1, max_length=512)
    indexed_source_binding_id: str = Field(..., min_length=1, max_length=512)
    remote_id: str = Field(..., min_length=1, max_length=512)
    delivery_id: str = Field(..., min_length=1, max_length=512)
    materialization_sequence: int = Field(..., gt=0)
    manifest_id: str = Field(..., min_length=1, max_length=512)
    manifest_fingerprint: str = Field(
        ...,
        min_length=64,
        max_length=64,
    )
    entry: ConnectedSourceMaterializationManifestEntryV1

    _validate_ids = field_validator(
        "tenant_id",
        "workspace_id",
        "source_id",
        "indexed_source_binding_id",
        "remote_id",
        "delivery_id",
        "manifest_id",
    )(
        lambda value, info: _identifier(value, info.field_name or "identifier")
    )
    _validate_hash = field_validator("manifest_fingerprint")(
        lambda value, info: _sha256(value, info.field_name or "fingerprint")
    )


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
        knowledge_source_binding_ref: str,
    ) -> ConnectedSourceMaterializationManifestV1 | None:
        manifests = self.list_committed(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            source_id=source_id,
            indexed_source_binding_id=indexed_source_binding_id,
            knowledge_source_binding_ref=knowledge_source_binding_ref,
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

    def get_committed_for_remote(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        source_id: str,
        indexed_source_binding_id: str,
        knowledge_source_binding_ref: str,
        remote_id: str,
    ) -> ConnectedSourceMaterializationManifestV1 | None:
        descriptors = self._list_commit_descriptors(
            tenant_id=tenant_id,
            binding_id=knowledge_source_binding_ref,
            workspace_id=workspace_id,
            source_id=source_id,
            indexed_source_binding_id=indexed_source_binding_id,
        )
        candidates: list[
            tuple[int, KnowledgeSyncCommittedPublicationV2]
        ] = []
        for descriptor in descriptors:
            manifest = self._load_immutable(descriptor)
            if remote_id in manifest.revoked_remote_ids:
                return None
            candidate = self._get_remote_candidate(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                source_id=source_id,
                indexed_source_binding_id=indexed_source_binding_id,
                remote_id=remote_id,
                materialization_sequence=descriptor.materialization_sequence,
            )
            if candidate is None:
                continue
            if (
                candidate.delivery_id != descriptor.delivery_id
                or candidate.manifest_id != descriptor.manifest_id
                or candidate.manifest_fingerprint != descriptor.manifest_fingerprint
            ):
                raise ConnectedSourceMaterializationManifestConflict(
                    "connected_source_remote_candidate_descriptor_mismatch"
                )
            candidates.append((descriptor.materialization_sequence, descriptor))
        if not candidates:
            return None
        return self._load_immutable(max(candidates, key=lambda item: item[0])[1])

    def get_prepared_for_delivery(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        source_id: str,
        indexed_source_binding_id: str,
        delivery_id: str,
    ) -> ConnectedSourceMaterializationManifestV1 | None:
        index = self._store.get(
            self._immutable_partition(tenant_id),
            self._delivery_index_row_key(
                workspace_id=workspace_id,
                source_id=source_id,
                indexed_source_binding_id=indexed_source_binding_id,
                delivery_id=delivery_id,
            ),
        )
        if index is None:
            return None
        manifest = self._parse_delivery_index(
            index,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            source_id=source_id,
            indexed_source_binding_id=indexed_source_binding_id,
            delivery_id=delivery_id,
        )
        return manifest

    def list_committed(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        source_id: str,
        indexed_source_binding_id: str,
        knowledge_source_binding_ref: str,
    ) -> tuple[ConnectedSourceMaterializationManifestV1, ...]:
        descriptors = self._list_commit_descriptors(
            tenant_id=tenant_id,
            binding_id=knowledge_source_binding_ref,
            workspace_id=workspace_id,
            source_id=source_id,
            indexed_source_binding_id=indexed_source_binding_id,
        )
        manifests = [self._load_immutable(descriptor) for descriptor in descriptors]
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
        self._put_delivery_index(manifest)
        self._put_remote_candidates(manifest)
        existing = self._get_commit_descriptor(
            tenant_id=manifest.tenant_id,
            workspace_id=manifest.workspace_id,
            source_id=manifest.source_id,
            indexed_source_binding_id=manifest.indexed_source_binding_id,
            delivery_id=manifest.delivery_id,
        )
        candidate_descriptor = self._descriptor(manifest)
        if existing is not None and not self._descriptor_matches(
            existing,
            candidate_descriptor,
        ):
            raise ConnectedSourceMaterializationManifestConflict(
                "connected_source_manifest_delivery_conflict"
            )
        for committed in self._list_commit_descriptors(
            tenant_id=manifest.tenant_id,
            binding_id=manifest.knowledge_source_binding_ref,
            workspace_id=manifest.workspace_id,
            source_id=manifest.source_id,
            indexed_source_binding_id=manifest.indexed_source_binding_id,
        ):
            if (
                committed.materialization_sequence
                == manifest.materialization_sequence
                and not self._descriptor_matches(committed, candidate_descriptor)
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
            return ManifestCommitStatus.STALE
        if result.status is KnowledgeSyncPublicationCommitStatus.REPLAYED:
            return ManifestCommitStatus.REPLAYED
        return ManifestCommitStatus.COMMITTED

    @staticmethod
    def _immutable_partition(tenant_id: str) -> str:
        return f"lkw.managed_workspace:{tenant_id}:{_MANIFEST_ENTITY}:immutable"

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
    def _delivery_index_row_key(
        cls,
        *,
        workspace_id: str,
        source_id: str,
        indexed_source_binding_id: str,
        delivery_id: str,
    ) -> str:
        return (
            cls._scope_prefix(
                workspace_id=workspace_id,
                source_id=source_id,
                indexed_source_binding_id=indexed_source_binding_id,
            )
            + f"delivery:{delivery_id}"
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
    def _delivery_index_record(
        cls,
        manifest: ConnectedSourceMaterializationManifestV1,
    ) -> DocumentRecord:
        return DocumentRecord(
            partition_key=cls._immutable_partition(manifest.tenant_id),
            row_key=cls._delivery_index_row_key(
                workspace_id=manifest.workspace_id,
                source_id=manifest.source_id,
                indexed_source_binding_id=manifest.indexed_source_binding_id,
                delivery_id=manifest.delivery_id,
            ),
            data={
                "schema_version": _DELIVERY_INDEX_SCHEMA,
                "tenant_id": manifest.tenant_id,
                "workspace_id": manifest.workspace_id,
                "source_id": manifest.source_id,
                "indexed_source_binding_id": manifest.indexed_source_binding_id,
                "knowledge_source_binding_ref": manifest.knowledge_source_binding_ref,
                "delivery_id": manifest.delivery_id,
                "materialization_sequence": manifest.materialization_sequence,
                "manifest_id": manifest.manifest_id,
                "manifest_fingerprint": manifest.manifest_fingerprint,
            },
        )

    @classmethod
    def _remote_candidate_row_key(
        cls,
        *,
        workspace_id: str,
        source_id: str,
        indexed_source_binding_id: str,
        remote_id: str,
        materialization_sequence: int,
    ) -> str:
        remote_key = hashlib.sha256(remote_id.encode("utf-8")).hexdigest()
        return (
            cls._scope_prefix(
                workspace_id=workspace_id,
                source_id=source_id,
                indexed_source_binding_id=indexed_source_binding_id,
            )
            + f"remote:{remote_key}:sequence:{materialization_sequence}"
        )

    @classmethod
    def _remote_candidate_record(
        cls,
        manifest: ConnectedSourceMaterializationManifestV1,
        entry: ConnectedSourceMaterializationManifestEntryV1,
    ) -> DocumentRecord:
        candidate = ConnectedSourceRemoteCandidateV1(
            tenant_id=manifest.tenant_id,
            workspace_id=manifest.workspace_id,
            source_id=manifest.source_id,
            indexed_source_binding_id=manifest.indexed_source_binding_id,
            remote_id=entry.remote_id,
            delivery_id=manifest.delivery_id,
            materialization_sequence=manifest.materialization_sequence,
            manifest_id=manifest.manifest_id,
            manifest_fingerprint=manifest.manifest_fingerprint,
            entry=entry,
        )
        return DocumentRecord(
            partition_key=cls._immutable_partition(manifest.tenant_id),
            row_key=cls._remote_candidate_row_key(
                workspace_id=manifest.workspace_id,
                source_id=manifest.source_id,
                indexed_source_binding_id=manifest.indexed_source_binding_id,
                remote_id=entry.remote_id,
                materialization_sequence=manifest.materialization_sequence,
            ),
            data={
                "schema_version": _REMOTE_CANDIDATE_SCHEMA,
                "candidate": candidate.model_dump(mode="json"),
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

    def _put_delivery_index(
        self,
        manifest: ConnectedSourceMaterializationManifestV1,
    ) -> None:
        record = self._delivery_index_record(manifest)
        existing = self._store.get(record.partition_key, record.row_key)
        if existing is not None:
            if dict(existing.data) != dict(record.data):
                raise ConnectedSourceMaterializationManifestConflict(
                    "connected_source_delivery_index_conflict"
                )
            return
        if not self._store.put_if_absent(record):
            retry = self._store.get(record.partition_key, record.row_key)
            if retry is None or dict(retry.data) != dict(record.data):
                raise ConnectedSourceMaterializationManifestConflict(
                    "connected_source_delivery_index_conflict"
                )

    def _put_remote_candidates(
        self,
        manifest: ConnectedSourceMaterializationManifestV1,
    ) -> None:
        for entry in manifest.document_entries:
            record = self._remote_candidate_record(manifest, entry)
            existing = self._store.get(record.partition_key, record.row_key)
            if existing is not None:
                if dict(existing.data) != dict(record.data):
                    raise ConnectedSourceMaterializationManifestConflict(
                        "connected_source_remote_candidate_conflict"
                    )
                continue
            if not self._store.put_if_absent(record):
                retry = self._store.get(record.partition_key, record.row_key)
                if retry is None or dict(retry.data) != dict(record.data):
                    raise ConnectedSourceMaterializationManifestConflict(
                        "connected_source_remote_candidate_conflict"
                    )

    def _parse_delivery_index(
        self,
        record: DocumentRecord,
        *,
        tenant_id: str,
        workspace_id: str,
        source_id: str,
        indexed_source_binding_id: str,
        delivery_id: str,
    ) -> ConnectedSourceMaterializationManifestV1:
        data = dict(record.data)
        if (
            record.partition_key != self._immutable_partition(tenant_id)
            or record.row_key
            != self._delivery_index_row_key(
                workspace_id=workspace_id,
                source_id=source_id,
                indexed_source_binding_id=indexed_source_binding_id,
                delivery_id=delivery_id,
            )
            or data.get("schema_version") != _DELIVERY_INDEX_SCHEMA
            or data.get("tenant_id") != tenant_id
            or data.get("workspace_id") != workspace_id
            or data.get("source_id") != source_id
            or data.get("indexed_source_binding_id") != indexed_source_binding_id
            or data.get("delivery_id") != delivery_id
            or not isinstance(data.get("manifest_id"), str)
            or not isinstance(data.get("manifest_fingerprint"), str)
            or not isinstance(data.get("materialization_sequence"), int)
        ):
            raise ConnectedSourceMaterializationManifestConflict(
                "connected_source_delivery_index_corrupt"
            )
        manifest_record = self._store.get(
            self._immutable_partition(tenant_id),
            self._immutable_row_key_values(
                workspace_id=workspace_id,
                source_id=source_id,
                indexed_source_binding_id=indexed_source_binding_id,
                delivery_id=delivery_id,
                materialization_sequence=data["materialization_sequence"],
                manifest_id=data["manifest_id"],
            ),
        )
        if manifest_record is None:
            raise ConnectedSourceMaterializationManifestConflict(
                "connected_source_manifest_immutable_missing"
            )
        manifest = self._parse_immutable(
            manifest_record,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            source_id=source_id,
            indexed_source_binding_id=indexed_source_binding_id,
        )
        if (
            manifest.delivery_id != delivery_id
            or manifest.manifest_id != data["manifest_id"]
            or manifest.manifest_fingerprint != data["manifest_fingerprint"]
            or manifest.materialization_sequence != data["materialization_sequence"]
        ):
            raise ConnectedSourceMaterializationManifestConflict(
                "connected_source_delivery_index_mismatch"
            )
        return manifest

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
                record.data["manifest"],
                strict=False,
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
    ) -> KnowledgeSyncCommittedPublicationV2 | None:
        index = self._store.get(
            self._immutable_partition(tenant_id),
            self._delivery_index_row_key(
                workspace_id=workspace_id,
                source_id=source_id,
                indexed_source_binding_id=indexed_source_binding_id,
                delivery_id=delivery_id,
            ),
        )
        if index is None:
            return None
        manifest = self._parse_delivery_index(
            index,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            source_id=source_id,
            indexed_source_binding_id=indexed_source_binding_id,
            delivery_id=delivery_id,
        )
        return self._publication_authority.read_committed_publication_for_delivery(
            tenant_id=tenant_id,
            binding_id=manifest.knowledge_source_binding_ref,
            delivery_id=delivery_id,
        )

    def _list_commit_descriptors(
        self,
        *,
        tenant_id: str,
        binding_id: str,
        workspace_id: str,
        source_id: str,
        indexed_source_binding_id: str,
    ) -> tuple[KnowledgeSyncCommittedPublicationV2, ...]:
        return tuple(
            descriptor
            for descriptor in self._publication_authority.list_committed_publications(
                tenant_id=tenant_id,
                binding_id=binding_id,
            )
            if (
                descriptor.workspace_id == workspace_id
                and descriptor.source_id == source_id
                and descriptor.indexed_source_binding_id == indexed_source_binding_id
            )
        )

    def _get_remote_candidate(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        source_id: str,
        indexed_source_binding_id: str,
        remote_id: str,
        materialization_sequence: int,
    ) -> ConnectedSourceRemoteCandidateV1 | None:
        record = self._store.get(
            self._immutable_partition(tenant_id),
            self._remote_candidate_row_key(
                workspace_id=workspace_id,
                source_id=source_id,
                indexed_source_binding_id=indexed_source_binding_id,
                remote_id=remote_id,
                materialization_sequence=materialization_sequence,
            ),
        )
        if record is None:
            return None
        if (
            record.data.get("schema_version") != _REMOTE_CANDIDATE_SCHEMA
            or not isinstance(record.data.get("candidate"), dict)
        ):
            raise ConnectedSourceMaterializationManifestConflict(
                "connected_source_remote_candidate_corrupt"
            )
        try:
            candidate = ConnectedSourceRemoteCandidateV1.model_validate(
                record.data["candidate"],
                strict=False,
            )
        except (TypeError, ValueError):
            raise ConnectedSourceMaterializationManifestConflict(
                "connected_source_remote_candidate_corrupt"
            ) from None
        if (
            record.partition_key != self._immutable_partition(tenant_id)
            or record.row_key
            != self._remote_candidate_row_key(
                workspace_id=workspace_id,
                source_id=source_id,
                indexed_source_binding_id=indexed_source_binding_id,
                remote_id=remote_id,
                materialization_sequence=materialization_sequence,
            )
            or candidate.tenant_id != tenant_id
            or candidate.workspace_id != workspace_id
            or candidate.source_id != source_id
            or candidate.indexed_source_binding_id != indexed_source_binding_id
            or candidate.remote_id != remote_id
            or candidate.materialization_sequence != materialization_sequence
        ):
            raise ConnectedSourceMaterializationManifestConflict(
                "connected_source_remote_candidate_identity_conflict"
            )
        return candidate

    @staticmethod
    def _descriptor_matches(
        left: KnowledgeSyncCommittedPublicationV1,
        right: KnowledgeSyncCommittedPublicationV1,
    ) -> bool:
        return (
            left.tenant_id == right.tenant_id
            and left.binding_id == right.binding_id
            and left.workspace_id == right.workspace_id
            and left.source_id == right.source_id
            and left.indexed_source_binding_id == right.indexed_source_binding_id
            and left.delivery_id == right.delivery_id
            and left.materialization_sequence == right.materialization_sequence
            and left.manifest_id == right.manifest_id
            and left.manifest_fingerprint == right.manifest_fingerprint
        )

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
