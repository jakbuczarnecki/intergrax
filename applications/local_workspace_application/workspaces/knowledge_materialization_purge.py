# © Artur Czarnecki. All rights reserved.

"""Restart-safe, exact-scope deletion of connected-source materialization."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any, Protocol, cast

from local_workspace_application.workspaces.connected_source_manifest import (
    ConnectedSourceMaterializationManifestConflict,
    ConnectedSourceMaterializationManifestRepository,
)
from local_workspace_application.workspaces.connected_source_models import (
    ConnectedSourceDeliverySequenceAssignment,
)
from local_workspace_application.workspaces.materialization_visibility import (
    KnowledgeMaterializationOwnershipV1,
)
from local_workspace_application.workspaces.models import WorkspaceDocumentReference
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository
from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

from intergrax.integrations.contracts.document_store import (
    ConditionalDocumentStore,
    DocumentRecord,
)
from intergrax.rag.vectorstore.contracts.native_vectorstore import VectorStoreScope
from intergrax.runtime.vendor_knowledge.sync_publication_fence import (
    KnowledgeSyncPublicationCommitNodeV1,
    KnowledgeSyncPublicationFenceConflict,
    KnowledgeSyncPublicationFencePort,
    KnowledgeSyncPublicationInProgress,
)

_PURGE_SCHEMA = "lkw.knowledge_materialization_purge.v1"
_PURGE_ENTITY = "knowledge_materialization_purge"
_IDENTIFIER_MAX_LENGTH = 512
_PAGE_DEFAULT = 200
_PAGE_MAX = 500
_VECTOR_PAGE = 200
_SHA256_LENGTH = 64


def _workspace_partition(tenant_id: str, entity: str) -> str:
    return f"lkw.managed_workspace:{tenant_id}:{entity}"


def _identifier(value: str, field_name: str) -> str:
    if not isinstance(value, str) or value != value.strip() or not value:
        raise ValueError(f"{field_name}_must_be_normalized")
    if len(value) > _IDENTIFIER_MAX_LENGTH or any(
        ord(char) < 32 or ord(char) == 127 for char in value
    ):
        raise ValueError(f"{field_name}_must_be_normalized")
    return value


def _utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name}_must_be_timezone_aware")
    if value.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name}_must_be_utc")
    return value.astimezone(UTC)


class KnowledgeMaterializationPurgeStatusV1(StrEnum):
    PREPARING = "preparing"
    INVALIDATED = "invalidated"
    DELETING = "deleting"
    COMPLETED = "completed"
    FAILED = "failed"


class KnowledgeMaterializationPurgePhaseV1(StrEnum):
    MANIFESTS = "manifests"
    PUBLICATION_CHAIN = "publication_chain"
    SEQUENCE = "sequence"


class KnowledgeMaterializationPurgeRequestV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    tenant_id: str = Field(min_length=1, max_length=_IDENTIFIER_MAX_LENGTH)
    workspace_id: str = Field(min_length=1, max_length=_IDENTIFIER_MAX_LENGTH)
    source_id: str = Field(min_length=1, max_length=_IDENTIFIER_MAX_LENGTH)
    indexed_source_binding_id: str = Field(
        min_length=1, max_length=_IDENTIFIER_MAX_LENGTH
    )
    knowledge_source_binding_ref: str = Field(
        min_length=1, max_length=_IDENTIFIER_MAX_LENGTH
    )
    requested_lifecycle_revision: StrictInt = Field(ge=1)
    operation_id: str = Field(min_length=1, max_length=_IDENTIFIER_MAX_LENGTH)

    _validate_ids = field_validator(
        "tenant_id",
        "workspace_id",
        "source_id",
        "indexed_source_binding_id",
        "knowledge_source_binding_ref",
        "operation_id",
    )(
        lambda value, info: _identifier(value, info.field_name or "identifier")
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


def knowledge_materialization_purge_id(
    request: KnowledgeMaterializationPurgeRequestV1,
) -> str:
    """Derive one stable purge identity from the exact ownership scope."""
    payload = json.dumps(
        {
            "schema": _PURGE_SCHEMA,
            "tenant_id": request.tenant_id,
            "workspace_id": request.workspace_id,
            "source_id": request.source_id,
            "indexed_source_binding_id": request.indexed_source_binding_id,
            "knowledge_source_binding_ref": request.knowledge_source_binding_ref,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class KnowledgeMaterializationPurgeCursorV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    phase: KnowledgeMaterializationPurgePhaseV1
    publication_commit_id: str | None = Field(
        default=None, min_length=_SHA256_LENGTH, max_length=_SHA256_LENGTH
    )

    @field_validator("publication_commit_id")
    @classmethod
    def _validate_commit_id(cls, value: str | None) -> str | None:
        if value is not None and any(char not in "0123456789abcdef" for char in value):
            raise ValueError("publication_commit_id_must_be_sha256")
        return value


class KnowledgeMaterializationPurgeCountersV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    documents_deleted: StrictInt = Field(default=0, ge=0)
    document_refs_deleted: StrictInt = Field(default=0, ge=0)
    chunks_deleted: StrictInt = Field(default=0, ge=0)
    embeddings_deleted: StrictInt = Field(default=0, ge=0)
    manifests_deleted: StrictInt = Field(default=0, ge=0)
    remote_candidates_deleted: StrictInt = Field(default=0, ge=0)
    receipts_deleted: StrictInt = Field(default=0, ge=0)
    sequence_assignments_deleted: StrictInt = Field(default=0, ge=0)
    active_pointers_deleted: StrictInt = Field(default=0, ge=0)
    publication_nodes_deleted: StrictInt = Field(default=0, ge=0)
    already_absent: StrictInt = Field(default=0, ge=0)


class KnowledgeMaterializationPurgeStateV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    purge_id: str = Field(min_length=_SHA256_LENGTH, max_length=_SHA256_LENGTH)
    request: KnowledgeMaterializationPurgeRequestV1
    status: KnowledgeMaterializationPurgeStatusV1
    cursor: KnowledgeMaterializationPurgeCursorV1 | None
    counters: KnowledgeMaterializationPurgeCountersV1
    started_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None
    last_error_code: str | None = Field(default=None, max_length=128)

    _validate_purge_id = field_validator("purge_id")(
        lambda value, info: value
        if len(value) == _SHA256_LENGTH
        and all(char in "0123456789abcdef" for char in value)
        else (_ for _ in ()).throw(
            ValueError(f"{info.field_name or 'purge_id'}_must_be_sha256")
        )
    )
    _validate_datetimes = field_validator("started_at", "updated_at", "completed_at")(
        lambda value, info: None
        if value is None
        else _utc(value, info.field_name or "datetime")
    )


class KnowledgeMaterializationPurgeResultV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    state: KnowledgeMaterializationPurgeStateV1


class KnowledgeMaterializationDeletionResultV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    chunks_deleted: StrictInt = Field(default=0, ge=0)
    embeddings_deleted: StrictInt = Field(default=0, ge=0)
    already_absent: StrictInt = Field(default=0, ge=0)


class KnowledgeMaterializationDeletionPort(Protocol):
    def delete_document_materialization(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        document_id: str,
        expected_ownership: KnowledgeMaterializationOwnershipV1,
    ) -> KnowledgeMaterializationDeletionResultV1:
        ...


class VectorStoreKnowledgeMaterializationDeletion:
    """Exact document-id vector deletion through the existing vector-store port."""

    def __init__(self, vectorstore_manager: Any, *, page_size: int = _VECTOR_PAGE) -> None:
        self._manager = vectorstore_manager
        self._page_size = max(1, min(page_size, _PAGE_MAX))

    def delete_document_materialization(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        document_id: str,
        expected_ownership: KnowledgeMaterializationOwnershipV1,
    ) -> KnowledgeMaterializationDeletionResultV1:
        if (
            expected_ownership.tenant_id != tenant_id
            or expected_ownership.workspace_id != workspace_id
        ):
            raise KnowledgeMaterializationPurgeError("ownership_mismatch")
        search = getattr(self._manager, "search_by_metadata", None)
        delete = getattr(self._manager, "delete", None)
        if not callable(search) or not callable(delete):
            raise KnowledgeMaterializationPurgeError("BLOCKED_ARCHITECTURE")
        deleted = 0
        while True:
            raw_matches = search(
                conditions={
                    "tenant_id": tenant_id,
                    "workspace_id": workspace_id,
                    "source_id": expected_ownership.source_id,
                    "document_id": document_id,
                },
                limit=self._page_size,
            )
            matches = cast(list[dict[str, Any]], raw_matches)
            if not matches:
                break
            ids: list[str] = []
            for match in matches:
                metadata = match.get("metadata")
                vector_id = str(match.get("id") or "").strip()
                if not isinstance(metadata, dict) or not vector_id:
                    raise KnowledgeMaterializationPurgeError(
                        "BLOCKED_CORRUPT_STATE"
                    )
                if any(
                    metadata.get(key) != value
                    for key, value in (
                        ("tenant_id", tenant_id),
                        ("workspace_id", workspace_id),
                        ("source_id", expected_ownership.source_id),
                        ("document_id", document_id),
                    )
                ):
                    raise KnowledgeMaterializationPurgeError(
                        "vector_ownership_mismatch"
                    )
                ids.append(vector_id)
            if not ids:
                break
            try:
                delete(
                    ids,
                    scope=VectorStoreScope(
                        tenant_id=tenant_id,
                        workspace_id=workspace_id,
                    ),
                )
            except TypeError as exc:
                raise KnowledgeMaterializationPurgeError(
                    "BLOCKED_ARCHITECTURE"
                ) from exc
            deleted += len(ids)
            if len(matches) < self._page_size:
                break
        return KnowledgeMaterializationDeletionResultV1(
            chunks_deleted=deleted,
            embeddings_deleted=deleted,
            already_absent=1 if deleted == 0 else 0,
        )


class KnowledgeMaterializationPurgeError(RuntimeError):
    def __init__(self, error_code: str) -> None:
        self.error_code = _identifier(error_code, "error_code")
        super().__init__(self.error_code)


class KnowledgeMaterializationPurgeService:
    """Durable orchestration boundary used by the future detach lifecycle."""

    def __init__(
        self,
        *,
        repository: ManagedWorkspaceRepository,
        publication_authority: KnowledgeSyncPublicationFencePort,
        deletion_port: KnowledgeMaterializationDeletionPort,
        clock: Callable[[], datetime] | None = None,
        page_size: int = _PAGE_DEFAULT,
    ) -> None:
        if not isinstance(repository.document_store, ConditionalDocumentStore):
            raise TypeError("purge requires ConditionalDocumentStore")
        if page_size < 1 or page_size > _PAGE_MAX:
            raise ValueError("page_size must be in range 1..500")
        self._repository = repository
        self._store = repository.document_store
        self._publication_authority = publication_authority
        self._deletion_port = deletion_port
        self._clock = clock or (lambda: datetime.now(UTC))
        self._page_size = page_size

    def start_or_resume(
        self,
        request: KnowledgeMaterializationPurgeRequestV1,
    ) -> KnowledgeMaterializationPurgeResultV1:
        purge_id = knowledge_materialization_purge_id(request)
        record = self._read_state(request.tenant_id, purge_id)
        if record is None:
            state = self._new_state(request, purge_id)
            if not self._store.put_if_absent(self._state_record(state)):
                record = self._read_state(request.tenant_id, purge_id)
                if record is None:
                    raise KnowledgeMaterializationPurgeError("purge_state_create_conflict")
            else:
                record = self._state_record(state)
        assert record is not None
        state = self._parse_state(record, request, purge_id)
        if state.status is KnowledgeMaterializationPurgeStatusV1.COMPLETED:
            return KnowledgeMaterializationPurgeResultV1(state=state)
        if (
            state.status is KnowledgeMaterializationPurgeStatusV1.FAILED
            and state.last_error_code
            and (
                state.last_error_code.startswith("BLOCKED_")
                or "mismatch" in state.last_error_code
                or "corrupt" in state.last_error_code
            )
        ):
            return KnowledgeMaterializationPurgeResultV1(state=state)
        if state.status is KnowledgeMaterializationPurgeStatusV1.FAILED:
            state = state.model_copy(
                update={
                    "status": KnowledgeMaterializationPurgeStatusV1.PREPARING,
                    "updated_at": self._now(),
                    "last_error_code": None,
                }
            )
        try:
            next_state = self._step(state)
        except KnowledgeMaterializationPurgeError as exc:
            failed = state.model_copy(
                update={
                    "status": KnowledgeMaterializationPurgeStatusV1.FAILED,
                    "updated_at": self._now(),
                    "last_error_code": exc.error_code,
                }
            )
            self._replace_state(record, failed)
            return KnowledgeMaterializationPurgeResultV1(state=failed)
        except KnowledgeSyncPublicationInProgress:
            failed = state.model_copy(
                update={
                    "status": KnowledgeMaterializationPurgeStatusV1.FAILED,
                    "updated_at": self._now(),
                    "last_error_code": "publication_in_progress",
                }
            )
            self._replace_state(record, failed)
            return KnowledgeMaterializationPurgeResultV1(state=failed)
        except KnowledgeSyncPublicationFenceConflict:
            failed = state.model_copy(
                update={
                    "status": KnowledgeMaterializationPurgeStatusV1.FAILED,
                    "updated_at": self._now(),
                    "last_error_code": "publication_fence_conflict",
                }
            )
            self._replace_state(record, failed)
            return KnowledgeMaterializationPurgeResultV1(state=failed)
        except ConnectedSourceMaterializationManifestConflict:
            failed = state.model_copy(
                update={
                    "status": KnowledgeMaterializationPurgeStatusV1.FAILED,
                    "updated_at": self._now(),
                    "last_error_code": "BLOCKED_CORRUPT_STATE",
                }
            )
            self._replace_state(record, failed)
            return KnowledgeMaterializationPurgeResultV1(state=failed)
        except (TypeError, ValueError):
            failed = state.model_copy(
                update={
                    "status": KnowledgeMaterializationPurgeStatusV1.FAILED,
                    "updated_at": self._now(),
                    "last_error_code": "BLOCKED_CORRUPT_STATE",
                }
            )
            self._replace_state(record, failed)
            return KnowledgeMaterializationPurgeResultV1(state=failed)
        saved = self._replace_state(record, next_state)
        return KnowledgeMaterializationPurgeResultV1(state=saved)

    def _step(
        self, state: KnowledgeMaterializationPurgeStateV1
    ) -> KnowledgeMaterializationPurgeStateV1:
        if state.status is KnowledgeMaterializationPurgeStatusV1.PREPARING:
            fence = self._publication_authority.invalidate_for_purge(
                tenant_id=state.request.tenant_id,
                binding_id=state.request.knowledge_source_binding_ref,
                requested_lifecycle_revision=state.request.requested_lifecycle_revision,
                purge_id=state.purge_id,
            )
            head = self._publication_authority.read_publication_head(
                tenant_id=state.request.tenant_id,
                binding_id=state.request.knowledge_source_binding_ref,
            )
            if fence.enabled or not fence.detached:
                raise KnowledgeMaterializationPurgeError("purge_invalidation_not_committed")
            return state.model_copy(
                update={
                    "status": KnowledgeMaterializationPurgeStatusV1.INVALIDATED,
                    "updated_at": self._now(),
                    "cursor": KnowledgeMaterializationPurgeCursorV1(
                        phase=KnowledgeMaterializationPurgePhaseV1.MANIFESTS,
                        publication_commit_id=(
                            head.publication_commit_id if head is not None else None
                        ),
                    ),
                    "last_error_code": None,
                }
            )
        if state.cursor is None:
            raise KnowledgeMaterializationPurgeError("BLOCKED_CORRUPT_STATE")
        if state.cursor.phase is KnowledgeMaterializationPurgePhaseV1.MANIFESTS:
            return self._delete_manifest_page(state)
        if state.cursor.phase is KnowledgeMaterializationPurgePhaseV1.PUBLICATION_CHAIN:
            return self._delete_publication_node(state)
        return self._delete_sequence_page(state)

    def _delete_manifest_page(
        self, state: KnowledgeMaterializationPurgeStateV1
    ) -> KnowledgeMaterializationPurgeStateV1:
        request = state.request
        cursor = state.cursor
        if cursor is None:
            raise KnowledgeMaterializationPurgeError("BLOCKED_CORRUPT_STATE")
        manifest_repository = ConnectedSourceMaterializationManifestRepository(self._store)
        partition = manifest_repository._immutable_partition(request.tenant_id)
        prefix = manifest_repository._scope_prefix(
            workspace_id=request.workspace_id,
            source_id=request.source_id,
            indexed_source_binding_id=request.indexed_source_binding_id,
        ) + "manifest:"
        page = self._store.query(partition, limit=self._page_size, row_key_prefix=prefix)
        if not page.documents:
            return state.model_copy(
                update={
                    "status": KnowledgeMaterializationPurgeStatusV1.DELETING,
                    "updated_at": self._now(),
                    "cursor": cursor.model_copy(
                        update={
                            "phase": KnowledgeMaterializationPurgePhaseV1.PUBLICATION_CHAIN
                        }
                    ),
                }
            )
        counters = state.counters
        for record in page.documents:
            try:
                manifest = manifest_repository._parse_immutable(
                    record,
                    tenant_id=request.tenant_id,
                    workspace_id=request.workspace_id,
                    source_id=request.source_id,
                    indexed_source_binding_id=request.indexed_source_binding_id,
                )
            except (ConnectedSourceMaterializationManifestConflict, TypeError, ValueError):
                raise KnowledgeMaterializationPurgeError(
                    "BLOCKED_CORRUPT_STATE"
                ) from None
            if (
                manifest.knowledge_source_binding_ref
                != request.knowledge_source_binding_ref
            ):
                raise KnowledgeMaterializationPurgeError("ownership_mismatch")
            counters = self._delete_manifest_materialization(
                counters, manifest, manifest_repository
            )
        return state.model_copy(
            update={"status": KnowledgeMaterializationPurgeStatusV1.DELETING,
                    "updated_at": self._now(), "counters": counters}
        )

    def _delete_manifest_materialization(
        self,
        counters: KnowledgeMaterializationPurgeCountersV1,
        manifest: Any,
        manifest_repository: ConnectedSourceMaterializationManifestRepository,
    ) -> KnowledgeMaterializationPurgeCountersV1:
        counters_update: dict[str, int] = {"manifests_deleted": 1}
        for entry in manifest.document_entries:
            ownership = KnowledgeMaterializationOwnershipV1.connected(
                tenant_id=manifest.tenant_id,
                workspace_id=manifest.workspace_id,
                source_id=manifest.source_id,
                indexed_source_binding_id=manifest.indexed_source_binding_id,
                knowledge_source_binding_ref=manifest.knowledge_source_binding_ref,
                delivery_id=manifest.delivery_id,
                remote_id=entry.remote_id,
                materialization_generation=entry.materialization_generation,
                materialization_sequence=manifest.materialization_sequence,
            )
            ref = self._repository.get_document_ref(
                tenant_id=manifest.tenant_id,
                workspace_id=manifest.workspace_id,
                document_id=entry.document_id,
            )
            if ref is not None and ref.materialization_ownership != ownership:
                raise KnowledgeMaterializationPurgeError("ownership_mismatch")
            result = self._deletion_port.delete_document_materialization(
                tenant_id=manifest.tenant_id,
                workspace_id=manifest.workspace_id,
                document_id=entry.document_id,
                expected_ownership=ownership,
            )
            counters_update["chunks_deleted"] = counters_update.get(
                "chunks_deleted", 0
            ) + result.chunks_deleted
            counters_update["embeddings_deleted"] = counters_update.get(
                "embeddings_deleted", 0
            ) + result.embeddings_deleted
            counters_update["already_absent"] = counters_update.get(
                "already_absent", 0
            ) + result.already_absent
            if ref is None:
                counters_update["already_absent"] = counters_update.get(
                    "already_absent", 0
                ) + 1
            else:
                self._delete_document_ref(ref)
                counters_update["documents_deleted"] = counters_update.get(
                    "documents_deleted", 0
                ) + 1
                counters_update["document_refs_deleted"] = counters_update.get(
                    "document_refs_deleted", 0
                ) + 1
            pointer = self._repository.get_active_materialization_pointer(
                tenant_id=ownership.tenant_id,
                workspace_id=ownership.workspace_id,
                source_id=ownership.source_id,
                indexed_source_binding_id=ownership.indexed_source_binding_id or "",
                remote_id=ownership.remote_id or "",
            )
            if pointer is not None:
                if (
                    pointer.tenant_id != ownership.tenant_id
                    or pointer.workspace_id != ownership.workspace_id
                    or pointer.source_id != ownership.source_id
                    or pointer.indexed_source_binding_id
                    != ownership.indexed_source_binding_id
                    or pointer.delivery_id != ownership.delivery_id
                    or pointer.document_id != entry.document_id
                    or pointer.materialization_generation
                    != ownership.materialization_generation
                ):
                    raise KnowledgeMaterializationPurgeError("ownership_mismatch")
                self._store.delete(
                    _workspace_partition(
                        ownership.tenant_id, "materialization_active_pointer"
                    ),
                    self._repository._active_materialization_pointer_key(
                        workspace_id=ownership.workspace_id,
                        source_id=ownership.source_id,
                        indexed_source_binding_id=ownership.indexed_source_binding_id or "",
                        remote_id=ownership.remote_id or "",
                    ),
                )
                counters_update["active_pointers_deleted"] = counters_update.get(
                    "active_pointers_deleted", 0
                ) + 1
            candidate = manifest_repository._get_remote_candidate(
                tenant_id=manifest.tenant_id,
                workspace_id=manifest.workspace_id,
                source_id=manifest.source_id,
                indexed_source_binding_id=manifest.indexed_source_binding_id,
                remote_id=entry.remote_id,
                materialization_sequence=manifest.materialization_sequence,
            )
            if candidate is not None:
                if (
                    candidate.delivery_id != manifest.delivery_id
                    or candidate.manifest_id != manifest.manifest_id
                    or candidate.manifest_fingerprint != manifest.manifest_fingerprint
                ):
                    raise KnowledgeMaterializationPurgeError("BLOCKED_CORRUPT_STATE")
                self._store.delete(
                    manifest_repository._immutable_partition(manifest.tenant_id),
                    manifest_repository._remote_candidate_row_key(
                        workspace_id=manifest.workspace_id,
                        source_id=manifest.source_id,
                        indexed_source_binding_id=manifest.indexed_source_binding_id,
                        remote_id=entry.remote_id,
                        materialization_sequence=manifest.materialization_sequence,
                    ),
                )
                counters_update["remote_candidates_deleted"] = counters_update.get(
                    "remote_candidates_deleted", 0
                ) + 1
            else:
                counters_update["already_absent"] = counters_update.get(
                    "already_absent", 0
                ) + 1
        receipt = self._repository.get_connected_source_delivery_receipt(
            tenant_id=manifest.tenant_id,
            workspace_id=manifest.workspace_id,
            source_id=manifest.source_id,
            delivery_id=manifest.delivery_id,
        )
        if receipt is not None:
            if (
                receipt.tenant_id != manifest.tenant_id
                or receipt.workspace_id != manifest.workspace_id
                or receipt.source_id != manifest.source_id
                or receipt.indexed_source_binding_id
                != manifest.indexed_source_binding_id
                or receipt.knowledge_source_binding_ref
                != manifest.knowledge_source_binding_ref
                or receipt.delivery_id != manifest.delivery_id
                or receipt.materialization_sequence
                != manifest.materialization_sequence
            ):
                raise KnowledgeMaterializationPurgeError("ownership_mismatch")
            self._repository.delete_connected_source_delivery_receipt(
                tenant_id=manifest.tenant_id,
                workspace_id=manifest.workspace_id,
                source_id=manifest.source_id,
                delivery_id=manifest.delivery_id,
            )
            counters_update["receipts_deleted"] = 1
        else:
            counters_update["already_absent"] = counters_update.get(
                "already_absent", 0
            ) + 1
        assignment = self._repository.get_connected_source_delivery_sequence_assignment(
            tenant_id=manifest.tenant_id,
            workspace_id=manifest.workspace_id,
            source_id=manifest.source_id,
            indexed_source_binding_id=manifest.indexed_source_binding_id,
            delivery_id=manifest.delivery_id,
        )
        if assignment is not None:
            if assignment.materialization_sequence != manifest.materialization_sequence:
                raise KnowledgeMaterializationPurgeError("ownership_mismatch")
            self._delete_sequence_assignment(assignment)
            counters_update["sequence_assignments_deleted"] = 1
        index_record = self._store.get(
            manifest_repository._immutable_partition(manifest.tenant_id),
            manifest_repository._delivery_index_row_key(
                workspace_id=manifest.workspace_id,
                source_id=manifest.source_id,
                indexed_source_binding_id=manifest.indexed_source_binding_id,
                delivery_id=manifest.delivery_id,
            ),
        )
        if index_record is not None:
            expected_index = manifest_repository._delivery_index_record(manifest)
            if dict(index_record.data) != dict(expected_index.data):
                raise KnowledgeMaterializationPurgeError("BLOCKED_CORRUPT_STATE")
            self._store.delete(index_record.partition_key, index_record.row_key)
        self._store.delete(
            manifest_repository._immutable_partition(manifest.tenant_id),
            manifest_repository._immutable_row_key(manifest),
        )
        return counters.model_copy(update=counters_update)

    def _delete_publication_node(
        self, state: KnowledgeMaterializationPurgeStateV1
    ) -> KnowledgeMaterializationPurgeStateV1:
        cursor = state.cursor
        assert cursor is not None
        current_id = cursor.publication_commit_id
        fence = self._publication_authority.read_fence(
            tenant_id=state.request.tenant_id,
            binding_id=state.request.knowledge_source_binding_ref,
        )
        if fence is None or fence.enabled or not fence.detached:
            raise KnowledgeMaterializationPurgeError("BLOCKED_CORRUPT_STATE")
        if current_id is None:
            return state.model_copy(
                update={
                    "cursor": cursor.model_copy(
                        update={"phase": KnowledgeMaterializationPurgePhaseV1.SEQUENCE}
                    ),
                    "updated_at": self._now(),
                }
            )
        node = self._publication_authority.read_publication_commit_node(
            tenant_id=state.request.tenant_id,
            binding_id=state.request.knowledge_source_binding_ref,
            commit_id=current_id,
        )
        self._validate_publication_node(node, state.request)
        previous_id = node.previous_commit_id
        previous_descriptor = None
        if previous_id is not None:
            previous_descriptor = self._publication_authority.read_publication_commit_node(
                tenant_id=state.request.tenant_id,
                binding_id=state.request.knowledge_source_binding_ref,
                commit_id=previous_id,
            ).descriptor
        current_head = self._publication_authority.read_publication_head(
            tenant_id=state.request.tenant_id,
            binding_id=state.request.knowledge_source_binding_ref,
        )
        if current_head == node.descriptor:
            self._publication_authority.advance_purge_publication_head(
                expected_fence=fence,
                expected_head=node.descriptor,
                replacement_head=previous_descriptor,
            )
        elif current_head != previous_descriptor:
            raise KnowledgeMaterializationPurgeError("BLOCKED_CORRUPT_STATE")
        self._publication_authority.delete_publication_commit_node(node=node)
        return state.model_copy(
            update={
                "updated_at": self._now(),
                "counters": state.counters.model_copy(
                    update={
                        "publication_nodes_deleted": state.counters.publication_nodes_deleted
                        + 1
                    }
                ),
                "cursor": cursor.model_copy(
                    update={"publication_commit_id": previous_id}
                ),
            }
        )

    def _delete_sequence_page(
        self, state: KnowledgeMaterializationPurgeStateV1
    ) -> KnowledgeMaterializationPurgeStateV1:
        request = state.request
        partition = _workspace_partition(
            request.tenant_id, "connected_source_delivery_sequence_assignment"
        )
        prefix = self._repository._delivery_sequence_head_key(
            workspace_id=request.workspace_id,
            source_id=request.source_id,
            indexed_source_binding_id=request.indexed_source_binding_id,
        ) + ":"
        page = self._store.query(partition, limit=self._page_size, row_key_prefix=prefix)
        counters = state.counters
        if page.documents:
            deleted = 0
            for record in page.documents:
                try:
                    assignment = ConnectedSourceDeliverySequenceAssignment.model_validate(
                        dict(record.data), strict=False
                    )
                except (TypeError, ValueError):
                    raise KnowledgeMaterializationPurgeError(
                        "BLOCKED_CORRUPT_STATE"
                    ) from None
                if (
                    assignment.tenant_id != request.tenant_id
                    or assignment.workspace_id != request.workspace_id
                    or assignment.source_id != request.source_id
                    or assignment.indexed_source_binding_id
                    != request.indexed_source_binding_id
                ):
                    raise KnowledgeMaterializationPurgeError("ownership_mismatch")
                self._store.delete(record.partition_key, record.row_key)
                deleted += 1
            return state.model_copy(
                update={
                    "updated_at": self._now(),
                    "counters": counters.model_copy(
                        update={
                            "sequence_assignments_deleted": counters.sequence_assignments_deleted
                            + deleted
                        }
                    ),
                }
            )
        head = self._repository.get_connected_source_delivery_sequence_head(
            tenant_id=request.tenant_id,
            workspace_id=request.workspace_id,
            source_id=request.source_id,
            indexed_source_binding_id=request.indexed_source_binding_id,
        )
        if head is not None:
            if (
                head.tenant_id != request.tenant_id
                or head.workspace_id != request.workspace_id
                or head.source_id != request.source_id
                or head.indexed_source_binding_id
                != request.indexed_source_binding_id
            ):
                raise KnowledgeMaterializationPurgeError("ownership_mismatch")
            self._store.delete(partition.rsplit(":", 1)[0] + ":connected_source_delivery_sequence_head",
                               self._repository._delivery_sequence_head_key(
                                   workspace_id=request.workspace_id,
                                   source_id=request.source_id,
                                   indexed_source_binding_id=request.indexed_source_binding_id,
                               ))
        completed_at = self._now()
        return state.model_copy(
            update={
                "status": KnowledgeMaterializationPurgeStatusV1.COMPLETED,
                "updated_at": completed_at,
                "completed_at": completed_at,
                "cursor": None,
                "last_error_code": None,
            }
        )

    def _delete_document_ref(self, ref: WorkspaceDocumentReference) -> None:
        partition = _workspace_partition(ref.tenant_id, "document")
        primary_key = f"{ref.workspace_id}:{ref.document_id}"
        current = self._store.get(partition, primary_key)
        if current is None:
            return
        if dict(current.data) != ref.model_dump(mode="json"):
            raise KnowledgeMaterializationPurgeError("BLOCKED_CORRUPT_STATE")
        self._store.delete(partition, primary_key)
        path_key = f"path:{ref.workspace_id}:{ref.source_id}:{ref.source_path}"
        path_record = self._store.get(partition, path_key)
        if path_record is not None:
            if path_record.data.get("document_id") != ref.document_id:
                raise KnowledgeMaterializationPurgeError("ownership_mismatch")
            self._store.delete(partition, path_key)

    def _delete_sequence_assignment(
        self, assignment: ConnectedSourceDeliverySequenceAssignment
    ) -> None:
        self._store.delete(
            _workspace_partition(
                assignment.tenant_id,
                "connected_source_delivery_sequence_assignment",
            ),
            self._repository._delivery_sequence_assignment_key(
                workspace_id=assignment.workspace_id,
                source_id=assignment.source_id,
                indexed_source_binding_id=assignment.indexed_source_binding_id,
                delivery_id=assignment.delivery_id,
            ),
        )

    @staticmethod
    def _validate_publication_node(
        node: KnowledgeSyncPublicationCommitNodeV1,
        request: KnowledgeMaterializationPurgeRequestV1,
    ) -> None:
        descriptor = node.descriptor
        if (
            descriptor.tenant_id != request.tenant_id
            or descriptor.binding_id != request.knowledge_source_binding_ref
            or descriptor.workspace_id != request.workspace_id
            or descriptor.source_id != request.source_id
            or descriptor.indexed_source_binding_id
            != request.indexed_source_binding_id
        ):
            raise KnowledgeMaterializationPurgeError("ownership_mismatch")

    def _new_state(
        self,
        request: KnowledgeMaterializationPurgeRequestV1,
        purge_id: str,
    ) -> KnowledgeMaterializationPurgeStateV1:
        now = self._now()
        return KnowledgeMaterializationPurgeStateV1(
            purge_id=purge_id,
            request=request,
            status=KnowledgeMaterializationPurgeStatusV1.PREPARING,
            cursor=KnowledgeMaterializationPurgeCursorV1(
                phase=KnowledgeMaterializationPurgePhaseV1.MANIFESTS
            ),
            counters=KnowledgeMaterializationPurgeCountersV1(),
            started_at=now,
            updated_at=now,
        )

    def _read_state(self, tenant_id: str, purge_id: str) -> DocumentRecord | None:
        return self._store.get(
            self._state_partition(tenant_id),
            self._state_row_key(purge_id),
        )

    def _parse_state(
        self,
        record: DocumentRecord,
        request: KnowledgeMaterializationPurgeRequestV1,
        purge_id: str,
    ) -> KnowledgeMaterializationPurgeStateV1:
        if (
            record.partition_key != self._state_partition(request.tenant_id)
            or record.row_key != self._state_row_key(purge_id)
            or record.data.get("schema_version") != _PURGE_SCHEMA
            or not isinstance(record.data.get("state"), dict)
        ):
            raise KnowledgeMaterializationPurgeError("BLOCKED_CORRUPT_STATE")
        try:
            state = KnowledgeMaterializationPurgeStateV1.model_validate(
                record.data["state"], strict=False
            )
        except (TypeError, ValueError):
            raise KnowledgeMaterializationPurgeError("BLOCKED_CORRUPT_STATE") from None
        if (
            state.purge_id != purge_id
            or state.request.ownership_scope != request.ownership_scope
        ):
            raise KnowledgeMaterializationPurgeError("purge_scope_conflict")
        return state

    def _replace_state(
        self,
        expected: DocumentRecord,
        state: KnowledgeMaterializationPurgeStateV1,
    ) -> KnowledgeMaterializationPurgeStateV1:
        replacement = self._state_record(state)
        if self._store.replace_if_match(expected=expected, replacement=replacement):
            return state
        current = self._store.get(expected.partition_key, expected.row_key)
        if current is None:
            raise KnowledgeMaterializationPurgeError("purge_state_missing")
        return self._parse_state(current, state.request, state.purge_id)

    def _state_record(self, state: KnowledgeMaterializationPurgeStateV1) -> DocumentRecord:
        return DocumentRecord(
            partition_key=self._state_partition(state.request.tenant_id),
            row_key=self._state_row_key(state.purge_id),
            data={"schema_version": _PURGE_SCHEMA, "state": state.model_dump(mode="json")},
        )

    def _now(self) -> datetime:
        return _utc(self._clock(), "clock")

    @staticmethod
    def _state_partition(tenant_id: str) -> str:
        return f"lkw.managed_workspace:{tenant_id}:{_PURGE_ENTITY}"

    @staticmethod
    def _state_row_key(purge_id: str) -> str:
        return f"purge:{purge_id}"
