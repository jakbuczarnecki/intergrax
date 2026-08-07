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
    ConnectedSourceDeliveryReceipt,
    ConnectedSourceDeliverySequenceAssignment,
    ConnectedSourceOperationDeliveryAccounting,
    ConnectedSourceSyncEnqueueIntent,
)
from local_workspace_application.workspaces.connected_source_purge_completion_contracts import (
    ConnectedSourceDeliveryReceiptOwnershipIndexEntryV1,
    ConnectedSourceDeliveryReceiptOwnershipIndexError,
    ConnectedSourceDeliveryReceiptOwnershipPageV1,
    ConnectedSourceRecoveryMigrationGateError,
    ConnectedSourceRecoveryMigrationGateStatusV1,
)
from local_workspace_application.workspaces.connected_source_recovery_ownership_index import (
    ConnectedSourceRecoveryOwnershipIndexEntryV1,
    ConnectedSourceRecoveryOwnershipIndexError,
    RecoveryRecordKindV1,
    canonical_record_fingerprint,
    index_entry_for_delivery_accounting,
    index_entry_for_enqueue_intent,
    index_entry_for_index_receipt,
)
from local_workspace_application.workspaces.document_indexing import (
    _WorkspaceDocumentIndexReceipt,
)
from local_workspace_application.workspaces.document_ownership_index import (
    DocumentOwnershipIndexError,
    WorkspaceDocumentOwnershipIndexEntryV1,
)
from local_workspace_application.workspaces.materialization_visibility import (
    KnowledgeMaterializationActivePointerV1,
    KnowledgeMaterializationOwnershipModeV1,
    KnowledgeMaterializationOwnershipV1,
)
from local_workspace_application.workspaces.models import WorkspaceDocumentReference
from local_workspace_application.workspaces.repository import (
    ManagedWorkspaceRepository,
    WorkspaceKnowledgeConfigurationRepositoryError,
)
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
    DOCUMENT_REFERENCES = "document_references"
    RECOVERY_RECORDS = "recovery_records"
    MANIFESTS = "manifests"
    DELIVERY_RECORDS = "delivery_records"
    PUBLICATION_CHAIN = "publication_chain"
    COMPLETION_PROOF = "completion_proof"
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
    purge_id: str | None = Field(
        default=None,
        min_length=_SHA256_LENGTH,
        max_length=_SHA256_LENGTH,
    )
    document_store_cursor: str | None = Field(default=None, max_length=4096)
    publication_commit_id: str | None = Field(
        default=None, min_length=_SHA256_LENGTH, max_length=_SHA256_LENGTH
    )
    current_delivery_id: str | None = Field(default=None, max_length=512)
    record_kind: str | None = Field(default=None, max_length=64)
    current_manifest_row_key: str | None = Field(default=None, max_length=1024)
    current_manifest_id: str | None = Field(
        default=None, min_length=_SHA256_LENGTH, max_length=_SHA256_LENGTH
    )
    current_manifest_fingerprint: str | None = Field(
        default=None, min_length=_SHA256_LENGTH, max_length=_SHA256_LENGTH
    )
    manifest_entry_offset: StrictInt | None = Field(default=None, ge=0)

    @field_validator("purge_id", "publication_commit_id")
    @classmethod
    def _validate_digest(cls, value: str | None) -> str | None:
        if value is not None and any(char not in "0123456789abcdef" for char in value):
            raise ValueError("purge_cursor_digest_must_be_sha256")
        return value

    @field_validator("current_manifest_id", "current_manifest_fingerprint")
    @classmethod
    def _validate_manifest_digest(cls, value: str | None) -> str | None:
        if value is not None and (
            len(value) != _SHA256_LENGTH
            or any(char not in "0123456789abcdef" for char in value)
        ):
            raise ValueError("purge_cursor_manifest_digest_must_be_sha256")
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
    ownership_index_entries_deleted: StrictInt = Field(default=0, ge=0)
    orphan_index_entries_deleted: StrictInt = Field(default=0, ge=0)
    index_receipts_deleted: StrictInt = Field(default=0, ge=0)
    enqueue_intents_deleted: StrictInt = Field(default=0, ge=0)
    delivery_accounting_deleted: StrictInt = Field(default=0, ge=0)
    delivery_indexes_deleted: StrictInt = Field(default=0, ge=0)
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
        except (
            ConnectedSourceDeliveryReceiptOwnershipIndexError,
            ConnectedSourceRecoveryMigrationGateError,
        ):
            failed = state.model_copy(
                update={
                    "status": KnowledgeMaterializationPurgeStatusV1.FAILED,
                    "updated_at": self._now(),
                    "last_error_code": "BLOCKED_CORRUPT_STATE",
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
        except WorkspaceKnowledgeConfigurationRepositoryError as exc:
            failed = state.model_copy(
                update={
                    "status": KnowledgeMaterializationPurgeStatusV1.FAILED,
                    "updated_at": self._now(),
                    "last_error_code": (
                        "BLOCKED_LEGACY_MIGRATION"
                        if "migration_required" in exc.error_code
                        else "BLOCKED_CORRUPT_STATE"
                    ),
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
                        phase=KnowledgeMaterializationPurgePhaseV1.DOCUMENT_REFERENCES,
                        purge_id=state.purge_id,
                        publication_commit_id=(
                            head.publication_commit_id if head is not None else None
                        ),
                    ),
                    "last_error_code": None,
                }
            )
        if state.cursor is None:
            raise KnowledgeMaterializationPurgeError("BLOCKED_CORRUPT_STATE")
        if (
            state.cursor.purge_id is not None
            and state.cursor.purge_id != state.purge_id
        ):
            raise KnowledgeMaterializationPurgeError("purge_scope_conflict")
        if state.cursor.phase is KnowledgeMaterializationPurgePhaseV1.DOCUMENT_REFERENCES:
            return self._delete_document_reference_page(state)
        if state.cursor.phase is KnowledgeMaterializationPurgePhaseV1.RECOVERY_RECORDS:
            return self._delete_recovery_page(state)
        if state.cursor.phase is KnowledgeMaterializationPurgePhaseV1.MANIFESTS:
            return self._delete_manifest_page(state)
        if state.cursor.phase is KnowledgeMaterializationPurgePhaseV1.DELIVERY_RECORDS:
            return self._delete_delivery_record_page(state)
        if state.cursor.phase is KnowledgeMaterializationPurgePhaseV1.PUBLICATION_CHAIN:
            return self._delete_publication_node(state)
        if state.cursor.phase is KnowledgeMaterializationPurgePhaseV1.COMPLETION_PROOF:
            return self._completion_proof(state)
        if state.cursor.phase is KnowledgeMaterializationPurgePhaseV1.SEQUENCE:
            return self._delete_sequence_page(state)
        raise KnowledgeMaterializationPurgeError("BLOCKED_CORRUPT_STATE")

    def _delete_document_reference_page(
        self, state: KnowledgeMaterializationPurgeStateV1
    ) -> KnowledgeMaterializationPurgeStateV1:
        request = state.request
        cursor = state.cursor
        assert cursor is not None
        try:
            page = self._repository.list_document_refs_by_materialization_owner(
                tenant_id=request.tenant_id,
                workspace_id=request.workspace_id,
                source_id=request.source_id,
                indexed_source_binding_id=request.indexed_source_binding_id,
                knowledge_source_binding_ref=request.knowledge_source_binding_ref,
                limit=self._page_size,
                cursor=cursor.document_store_cursor,
            )
        except DocumentOwnershipIndexError as exc:
            raise KnowledgeMaterializationPurgeError(
                "BLOCKED_CORRUPT_STATE"
            ) from exc
        counters = state.counters
        for entry in page.orphan_index_entries:
            if self._repository.delete_document_ownership_index_entry(entry):
                counters = counters.model_copy(
                    update={
                        "ownership_index_entries_deleted": (
                            counters.ownership_index_entries_deleted + 1
                        ),
                        "orphan_index_entries_deleted": (
                            counters.orphan_index_entries_deleted + 1
                        ),
                        "already_absent": counters.already_absent + 1,
                    }
                )
        for ref in page.references:
            ownership = ref.materialization_ownership
            if (
                ownership is None
                or ownership.ownership_mode
                is not KnowledgeMaterializationOwnershipModeV1.CONNECTED_SOURCE
                or ownership.indexed_source_binding_id
                != request.indexed_source_binding_id
                or ownership.knowledge_source_binding_ref
                != request.knowledge_source_binding_ref
                or ownership.tenant_id != request.tenant_id
                or ownership.workspace_id != request.workspace_id
                or ownership.source_id != request.source_id
            ):
                raise KnowledgeMaterializationPurgeError("ownership_mismatch")
            result = self._deletion_port.delete_document_materialization(
                tenant_id=request.tenant_id,
                workspace_id=request.workspace_id,
                document_id=ref.document_id,
                expected_ownership=ownership,
            )
            self._delete_document_ref(ref)
            try:
                entry = WorkspaceDocumentOwnershipIndexEntryV1.for_reference(ref)
            except (TypeError, ValueError) as exc:
                raise KnowledgeMaterializationPurgeError(
                    "BLOCKED_CORRUPT_STATE"
                ) from exc
            index_deleted = self._repository.delete_document_ownership_index_entry(
                entry
            )
            counters = counters.model_copy(
                update={
                    "documents_deleted": counters.documents_deleted + 1,
                    "document_refs_deleted": counters.document_refs_deleted + 1,
                    "chunks_deleted": counters.chunks_deleted + result.chunks_deleted,
                    "embeddings_deleted": (
                        counters.embeddings_deleted + result.embeddings_deleted
                    ),
                    "already_absent": counters.already_absent + result.already_absent,
                    "ownership_index_entries_deleted": (
                        counters.ownership_index_entries_deleted
                        + (1 if index_deleted else 0)
                    ),
                }
            )
        if not page.references and not page.orphan_index_entries:
            next_cursor = cursor.model_copy(
                update={
                    "phase": KnowledgeMaterializationPurgePhaseV1.RECOVERY_RECORDS,
                    "document_store_cursor": None,
                    "record_kind": "index_receipts",
                }
            )
        else:
            next_cursor = cursor.model_copy(update={"document_store_cursor": page.next_cursor})
        return state.model_copy(
            update={
                "status": KnowledgeMaterializationPurgeStatusV1.DELETING,
                "updated_at": self._now(),
                "cursor": next_cursor,
                "counters": counters,
            }
        )

    def _delete_recovery_page(
        self, state: KnowledgeMaterializationPurgeStateV1
    ) -> KnowledgeMaterializationPurgeStateV1:
        request = state.request
        cursor = state.cursor
        assert cursor is not None
        kind = cursor.record_kind or "index_receipts"
        record_kind = {
            "index_receipts": RecoveryRecordKindV1.INDEX_RECEIPT,
            "accounting": RecoveryRecordKindV1.DELIVERY_ACCOUNTING,
            "enqueue_intents": RecoveryRecordKindV1.ENQUEUE_INTENT,
        }.get(kind)
        if record_kind is None:
            raise KnowledgeMaterializationPurgeError("BLOCKED_CORRUPT_STATE")
        try:
            page = self._repository.list_connected_source_recovery_records_by_owner(
                tenant_id=request.tenant_id,
                workspace_id=request.workspace_id,
                source_id=request.source_id,
                indexed_source_binding_id=request.indexed_source_binding_id,
                knowledge_source_binding_ref=request.knowledge_source_binding_ref,
                record_kind=record_kind,
                limit=self._page_size,
                cursor=cursor.document_store_cursor,
            )
        except ConnectedSourceRecoveryOwnershipIndexError as exc:
            raise KnowledgeMaterializationPurgeError(
                "BLOCKED_CORRUPT_STATE"
            ) from exc
        counters = state.counters
        for entry in page.orphan_index_entries:
            if self._repository.delete_connected_source_recovery_ownership_index_entry(
                entry
            ):
                counters = counters.model_copy(
                    update={
                        "orphan_index_entries_deleted": (
                            counters.orphan_index_entries_deleted + 1
                        ),
                        "already_absent": counters.already_absent + 1,
                    }
                )
        for entry in page.index_entries:
            counters = self._delete_recovery_index_entry(entry, request, counters)
        if page.index_entries or page.orphan_index_entries:
            next_cursor = cursor.model_copy(
                update={"document_store_cursor": page.next_cursor}
            )
        else:
            next_kind = {
                "index_receipts": "accounting",
                "accounting": "enqueue_intents",
                "enqueue_intents": None,
            }[kind]
            if next_kind is None:
                next_cursor = cursor.model_copy(
                    update={
                        "phase": KnowledgeMaterializationPurgePhaseV1.MANIFESTS,
                        "document_store_cursor": None,
                        "record_kind": None,
                        "current_manifest_row_key": None,
                        "current_manifest_id": None,
                        "current_manifest_fingerprint": None,
                        "current_delivery_id": None,
                        "manifest_entry_offset": None,
                    }
                )
            else:
                next_cursor = cursor.model_copy(
                    update={
                        "document_store_cursor": None,
                        "record_kind": next_kind,
                    }
                )
        return state.model_copy(
            update={
                "status": KnowledgeMaterializationPurgeStatusV1.DELETING,
                "updated_at": self._now(),
                "cursor": next_cursor,
                "counters": counters,
            }
        )

    def _delete_recovery_index_entry(
        self,
        entry: ConnectedSourceRecoveryOwnershipIndexEntryV1,
        request: KnowledgeMaterializationPurgeRequestV1,
        counters: KnowledgeMaterializationPurgeCountersV1,
    ) -> KnowledgeMaterializationPurgeCountersV1:
        if entry.ownership_scope != request.ownership_scope:
            raise KnowledgeMaterializationPurgeError("ownership_mismatch")
        canonical = self._store.get(
            entry.canonical_partition_key,
            entry.canonical_row_key,
        )
        if canonical is None:
            if self._repository.delete_connected_source_recovery_ownership_index_entry(
                entry
            ):
                return counters.model_copy(
                    update={
                        "orphan_index_entries_deleted": (
                            counters.orphan_index_entries_deleted + 1
                        ),
                        "already_absent": counters.already_absent + 1,
                    }
                )
            return counters
        if entry.record_kind is RecoveryRecordKindV1.INDEX_RECEIPT:
            return self._delete_indexed_index_receipt(entry, canonical, request, counters)
        if entry.record_kind is RecoveryRecordKindV1.DELIVERY_ACCOUNTING:
            return self._delete_indexed_accounting(entry, canonical, request, counters)
        if entry.record_kind is RecoveryRecordKindV1.ENQUEUE_INTENT:
            return self._delete_indexed_enqueue(entry, canonical, request, counters)
        raise KnowledgeMaterializationPurgeError("BLOCKED_CORRUPT_STATE")

    def _delete_indexed_index_receipt(
        self,
        entry: ConnectedSourceRecoveryOwnershipIndexEntryV1,
        record: DocumentRecord,
        request: KnowledgeMaterializationPurgeRequestV1,
        counters: KnowledgeMaterializationPurgeCountersV1,
    ) -> KnowledgeMaterializationPurgeCountersV1:
        try:
            receipt = _WorkspaceDocumentIndexReceipt.model_validate(
                dict(record.data), strict=False
            )
        except (TypeError, ValueError) as exc:
            raise KnowledgeMaterializationPurgeError(
                "BLOCKED_CORRUPT_STATE"
            ) from exc
        ownership = receipt.materialization_ownership
        if ownership is None:
            raise KnowledgeMaterializationPurgeError("BLOCKED_LEGACY_MIGRATION")
        if ownership.ownership_mode is not KnowledgeMaterializationOwnershipModeV1.CONNECTED_SOURCE:
            raise KnowledgeMaterializationPurgeError("ownership_mismatch")
        if ownership.identity_scope != receipt.materialization_scope:
            raise KnowledgeMaterializationPurgeError("ownership_mismatch")
        scope = (
            ownership.tenant_id,
            ownership.workspace_id,
            ownership.source_id,
            ownership.indexed_source_binding_id,
            ownership.knowledge_source_binding_ref,
        )
        if scope != request.ownership_scope:
            raise KnowledgeMaterializationPurgeError("ownership_mismatch")
        if canonical_record_fingerprint(receipt) != entry.canonical_fingerprint:
            raise KnowledgeMaterializationPurgeError("BLOCKED_CORRUPT_STATE")
        expected = index_entry_for_index_receipt(
            receipt,
            canonical_partition_key=entry.canonical_partition_key,
            canonical_row_key=entry.canonical_row_key,
            indexed_at=entry.indexed_at,
        )
        if (
            expected.row_key != entry.row_key
            or expected.ownership_scope != entry.ownership_scope
        ):
            raise KnowledgeMaterializationPurgeError("BLOCKED_CORRUPT_STATE")
        ref = self._repository.get_document_ref(
            tenant_id=request.tenant_id,
            workspace_id=request.workspace_id,
            document_id=receipt.document_id,
        )
        if ref is not None:
            if ref.materialization_ownership != ownership:
                raise KnowledgeMaterializationPurgeError("ownership_mismatch")
            result = self._deletion_port.delete_document_materialization(
                tenant_id=request.tenant_id,
                workspace_id=request.workspace_id,
                document_id=receipt.document_id,
                expected_ownership=ownership,
            )
            self._delete_document_ref(ref)
            doc_entry = WorkspaceDocumentOwnershipIndexEntryV1.for_reference(ref)
            index_deleted = self._repository.delete_document_ownership_index_entry(
                doc_entry
            )
            counters = counters.model_copy(
                update={
                    "documents_deleted": counters.documents_deleted + 1,
                    "document_refs_deleted": counters.document_refs_deleted + 1,
                    "chunks_deleted": counters.chunks_deleted + result.chunks_deleted,
                    "embeddings_deleted": (
                        counters.embeddings_deleted + result.embeddings_deleted
                    ),
                    "ownership_index_entries_deleted": (
                        counters.ownership_index_entries_deleted
                        + (1 if index_deleted else 0)
                    ),
                }
            )
        self._store.delete(record.partition_key, record.row_key)
        self._repository.delete_connected_source_recovery_ownership_index_entry(entry)
        return counters.model_copy(
            update={"index_receipts_deleted": counters.index_receipts_deleted + 1}
        )

    def _delete_indexed_accounting(
        self,
        entry: ConnectedSourceRecoveryOwnershipIndexEntryV1,
        record: DocumentRecord,
        request: KnowledgeMaterializationPurgeRequestV1,
        counters: KnowledgeMaterializationPurgeCountersV1,
    ) -> KnowledgeMaterializationPurgeCountersV1:
        try:
            item = ConnectedSourceOperationDeliveryAccounting.model_validate(
                dict(record.data), strict=False
            )
        except (TypeError, ValueError) as exc:
            raise KnowledgeMaterializationPurgeError(
                "BLOCKED_CORRUPT_STATE"
            ) from exc
        if item.ownership_classification != "COMPLETE_OWNERSHIP":
            raise KnowledgeMaterializationPurgeError("BLOCKED_LEGACY_MIGRATION")
        scope = (
            item.tenant_id,
            item.workspace_id,
            item.source_id,
            item.indexed_source_binding_id,
            item.knowledge_source_binding_ref,
        )
        if scope != request.ownership_scope:
            raise KnowledgeMaterializationPurgeError("ownership_mismatch")
        if canonical_record_fingerprint(item) != entry.canonical_fingerprint:
            raise KnowledgeMaterializationPurgeError("BLOCKED_CORRUPT_STATE")
        expected = index_entry_for_delivery_accounting(
            item,
            canonical_partition_key=entry.canonical_partition_key,
            canonical_row_key=entry.canonical_row_key,
            indexed_at=entry.indexed_at,
        )
        if expected.row_key != entry.row_key:
            raise KnowledgeMaterializationPurgeError("BLOCKED_CORRUPT_STATE")
        self._repository.delete_connected_source_delivery_accounting(
            tenant_id=item.tenant_id,
            operation_id=item.operation_id,
            delivery_id=item.delivery_id,
        )
        self._repository.delete_connected_source_recovery_ownership_index_entry(entry)
        return counters.model_copy(
            update={
                "delivery_accounting_deleted": counters.delivery_accounting_deleted + 1
            }
        )

    def _delete_indexed_enqueue(
        self,
        entry: ConnectedSourceRecoveryOwnershipIndexEntryV1,
        record: DocumentRecord,
        request: KnowledgeMaterializationPurgeRequestV1,
        counters: KnowledgeMaterializationPurgeCountersV1,
    ) -> KnowledgeMaterializationPurgeCountersV1:
        try:
            item = ConnectedSourceSyncEnqueueIntent.model_validate(
                dict(record.data), strict=False
            )
        except (TypeError, ValueError) as exc:
            raise KnowledgeMaterializationPurgeError(
                "BLOCKED_CORRUPT_STATE"
            ) from exc
        if item.ownership_classification != "COMPLETE_OWNERSHIP":
            raise KnowledgeMaterializationPurgeError("BLOCKED_LEGACY_MIGRATION")
        scope = (
            item.tenant_id,
            item.workspace_id,
            item.source_id,
            item.indexed_source_binding_id,
            item.knowledge_source_binding_ref,
        )
        if scope != request.ownership_scope:
            raise KnowledgeMaterializationPurgeError("ownership_mismatch")
        if canonical_record_fingerprint(item) != entry.canonical_fingerprint:
            raise KnowledgeMaterializationPurgeError("BLOCKED_CORRUPT_STATE")
        expected = index_entry_for_enqueue_intent(
            item,
            canonical_partition_key=entry.canonical_partition_key,
            canonical_row_key=entry.canonical_row_key,
            indexed_at=entry.indexed_at,
        )
        if expected.row_key != entry.row_key:
            raise KnowledgeMaterializationPurgeError("BLOCKED_CORRUPT_STATE")
        self._repository.delete_connected_source_sync_enqueue_intent(
            tenant_id=item.tenant_id,
            operation_id=item.operation_id,
        )
        self._repository.delete_connected_source_recovery_ownership_index_entry(entry)
        return counters.model_copy(
            update={"enqueue_intents_deleted": counters.enqueue_intents_deleted + 1}
        )

    def _delete_manifest_page(
        self, state: KnowledgeMaterializationPurgeStateV1
    ) -> KnowledgeMaterializationPurgeStateV1:
        cursor = state.cursor
        if cursor is None:
            raise KnowledgeMaterializationPurgeError("BLOCKED_CORRUPT_STATE")
        manifest_repository = ConnectedSourceMaterializationManifestRepository(
            self._store,
            publication_authority=self._publication_authority,
        )
        if cursor.current_manifest_id is None:
            return self._select_next_manifest(state, manifest_repository)
        return self._delete_manifest_entry_page(state, manifest_repository)

    def _select_next_manifest(
        self,
        state: KnowledgeMaterializationPurgeStateV1,
        manifest_repository: ConnectedSourceMaterializationManifestRepository,
    ) -> KnowledgeMaterializationPurgeStateV1:
        request = state.request
        cursor = state.cursor
        assert cursor is not None
        if (
            cursor.current_manifest_row_key is not None
            or cursor.current_manifest_fingerprint is not None
            or cursor.current_delivery_id is not None
            or cursor.manifest_entry_offset is not None
        ):
            raise KnowledgeMaterializationPurgeError("BLOCKED_CORRUPT_STATE")
        partition = manifest_repository._immutable_partition(request.tenant_id)
        prefix = manifest_repository._scope_prefix(
            workspace_id=request.workspace_id,
            source_id=request.source_id,
            indexed_source_binding_id=request.indexed_source_binding_id,
        ) + "manifest:"
        page = self._store.query(
            partition,
            limit=1,
            row_key_prefix=prefix,
            cursor=cursor.document_store_cursor,
        )
        if not page.documents:
            return state.model_copy(
                update={
                    "status": KnowledgeMaterializationPurgeStatusV1.DELETING,
                    "updated_at": self._now(),
                    "cursor": cursor.model_copy(
                        update={
                            "phase": KnowledgeMaterializationPurgePhaseV1.DELIVERY_RECORDS,
                            "document_store_cursor": None,
                            "record_kind": "delivery_indexes",
                            "current_manifest_row_key": None,
                            "current_manifest_id": None,
                            "current_manifest_fingerprint": None,
                            "current_delivery_id": None,
                            "manifest_entry_offset": None,
                        }
                    ),
                }
            )
        record = page.documents[0]
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
        selected = cursor.model_copy(
            update={
                "document_store_cursor": page.next_cursor,
                "current_manifest_row_key": record.row_key,
                "current_manifest_id": manifest.manifest_id,
                "current_manifest_fingerprint": manifest.manifest_fingerprint,
                "current_delivery_id": manifest.delivery_id,
                "manifest_entry_offset": 0,
            }
        )
        return self._delete_manifest_entry_page(
            state.model_copy(
                update={
                    "status": KnowledgeMaterializationPurgeStatusV1.DELETING,
                    "updated_at": self._now(),
                    "cursor": selected,
                }
            ),
            manifest_repository,
        )

    def _load_current_manifest(
        self,
        state: KnowledgeMaterializationPurgeStateV1,
        manifest_repository: ConnectedSourceMaterializationManifestRepository,
    ) -> Any:
        request = state.request
        cursor = state.cursor
        assert cursor is not None
        if (
            cursor.current_manifest_row_key is None
            or cursor.current_manifest_id is None
            or cursor.current_manifest_fingerprint is None
            or cursor.current_delivery_id is None
            or cursor.manifest_entry_offset is None
        ):
            raise KnowledgeMaterializationPurgeError("BLOCKED_CORRUPT_STATE")
        record = self._store.get(
            manifest_repository._immutable_partition(request.tenant_id),
            cursor.current_manifest_row_key,
        )
        if record is None:
            raise KnowledgeMaterializationPurgeError("BLOCKED_CORRUPT_STATE")
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
            or manifest.manifest_id != cursor.current_manifest_id
            or manifest.manifest_fingerprint != cursor.current_manifest_fingerprint
            or manifest.delivery_id != cursor.current_delivery_id
            or record.row_key != cursor.current_manifest_row_key
        ):
            raise KnowledgeMaterializationPurgeError("BLOCKED_CORRUPT_STATE")
        offset = cursor.manifest_entry_offset
        if offset < 0 or offset > len(manifest.document_entries):
            raise KnowledgeMaterializationPurgeError("BLOCKED_CORRUPT_STATE")
        return manifest

    def _delete_manifest_entry_page(
        self,
        state: KnowledgeMaterializationPurgeStateV1,
        manifest_repository: ConnectedSourceMaterializationManifestRepository,
    ) -> KnowledgeMaterializationPurgeStateV1:
        cursor = state.cursor
        assert cursor is not None
        manifest = self._load_current_manifest(state, manifest_repository)
        offset = cursor.manifest_entry_offset
        assert offset is not None
        end = min(offset + self._page_size, len(manifest.document_entries))
        counters = state.counters
        for entry in manifest.document_entries[offset:end]:
            counters = self._delete_one_manifest_entry(
                counters, manifest, entry, manifest_repository
            )
        if end < len(manifest.document_entries):
            next_cursor = cursor.model_copy(update={"manifest_entry_offset": end})
            return state.model_copy(
                update={
                    "status": KnowledgeMaterializationPurgeStatusV1.DELETING,
                    "updated_at": self._now(),
                    "cursor": next_cursor,
                    "counters": counters,
                }
            )
        counters = self._finalize_manifest_cleanup(
            counters, manifest, manifest_repository
        )
        next_cursor = cursor.model_copy(
            update={
                "current_manifest_row_key": None,
                "current_manifest_id": None,
                "current_manifest_fingerprint": None,
                "current_delivery_id": None,
                "manifest_entry_offset": None,
            }
        )
        return state.model_copy(
            update={
                "status": KnowledgeMaterializationPurgeStatusV1.DELETING,
                "updated_at": self._now(),
                "cursor": next_cursor,
                "counters": counters,
            }
        )

    def _delete_one_manifest_entry(
        self,
        counters: KnowledgeMaterializationPurgeCountersV1,
        manifest: Any,
        entry: Any,
        manifest_repository: ConnectedSourceMaterializationManifestRepository,
    ) -> KnowledgeMaterializationPurgeCountersV1:
        counters_update: dict[str, int] = {}
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
        counters_update["chunks_deleted"] = result.chunks_deleted
        counters_update["embeddings_deleted"] = result.embeddings_deleted
        counters_update["already_absent"] = result.already_absent
        if ref is None:
            counters_update["already_absent"] = (
                counters_update.get("already_absent", 0) + 1
            )
        else:
            self._delete_document_ref(ref)
            try:
                index_entry = WorkspaceDocumentOwnershipIndexEntryV1.for_reference(ref)
            except (TypeError, ValueError) as exc:
                raise KnowledgeMaterializationPurgeError(
                    "BLOCKED_CORRUPT_STATE"
                ) from exc
            if self._repository.delete_document_ownership_index_entry(index_entry):
                counters_update["ownership_index_entries_deleted"] = 1
            counters_update["documents_deleted"] = 1
            counters_update["document_refs_deleted"] = 1
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
                    indexed_source_binding_id=ownership.indexed_source_binding_id
                    or "",
                    remote_id=ownership.remote_id or "",
                ),
            )
            counters_update["active_pointers_deleted"] = 1
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
            counters_update["remote_candidates_deleted"] = 1
        else:
            counters_update["already_absent"] = (
                counters_update.get("already_absent", 0) + 1
            )
        return counters.model_copy(
            update={
                key: getattr(counters, key) + value
                for key, value in counters_update.items()
            }
        )

    def _finalize_manifest_cleanup(
        self,
        counters: KnowledgeMaterializationPurgeCountersV1,
        manifest: Any,
        manifest_repository: ConnectedSourceMaterializationManifestRepository,
    ) -> KnowledgeMaterializationPurgeCountersV1:
        counters_update: dict[str, int] = {"manifests_deleted": 1}
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
            counters_update["already_absent"] = 1
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
            counters_update["delivery_indexes_deleted"] = (
                counters_update.get("delivery_indexes_deleted", 0) + 1
            )
        self._store.delete(
            manifest_repository._immutable_partition(manifest.tenant_id),
            manifest_repository._immutable_row_key(manifest),
        )
        return counters.model_copy(
            update={
                key: getattr(counters, key) + value
                for key, value in counters_update.items()
            }
        )

    def _delete_delivery_record_page(
        self, state: KnowledgeMaterializationPurgeStateV1
    ) -> KnowledgeMaterializationPurgeStateV1:
        request = state.request
        cursor = state.cursor
        assert cursor is not None
        kind = cursor.record_kind or "delivery_indexes"
        manifest_repository = ConnectedSourceMaterializationManifestRepository(
            self._store
        )
        partition = manifest_repository._immutable_partition(request.tenant_id)
        scope_prefix = manifest_repository._scope_prefix(
            workspace_id=request.workspace_id,
            source_id=request.source_id,
            indexed_source_binding_id=request.indexed_source_binding_id,
        )
        if kind == "delivery_indexes":
            page = self._store.query(
                partition,
                limit=self._page_size,
                row_key_prefix=scope_prefix + "delivery:",
                cursor=cursor.document_store_cursor,
            )
        elif kind == "remote_candidates":
            page = self._store.query(
                partition,
                limit=self._page_size,
                row_key_prefix=scope_prefix + "remote:",
                cursor=cursor.document_store_cursor,
            )
        elif kind == "active_pointers":
            page = self._store.query(
                _workspace_partition(
                    request.tenant_id, "materialization_active_pointer"
                ),
                limit=self._page_size,
                row_key_prefix=(
                    f"{request.workspace_id}:{request.source_id}:"
                    f"{request.indexed_source_binding_id}:"
                ),
                cursor=cursor.document_store_cursor,
            )
        elif kind == "receipts":
            try:
                receipt_page = (
                    self._repository.list_connected_source_delivery_receipts_by_owner(
                        tenant_id=request.tenant_id,
                        workspace_id=request.workspace_id,
                        source_id=request.source_id,
                        indexed_source_binding_id=request.indexed_source_binding_id,
                        knowledge_source_binding_ref=request.knowledge_source_binding_ref,
                        limit=self._page_size,
                        cursor=cursor.document_store_cursor,
                    )
                )
            except ConnectedSourceDeliveryReceiptOwnershipIndexError as exc:
                raise KnowledgeMaterializationPurgeError(
                    "BLOCKED_CORRUPT_STATE"
                ) from exc
            page = receipt_page
        elif kind == "assignments":
            page = self._store.query(
                _workspace_partition(
                    request.tenant_id,
                    "connected_source_delivery_sequence_assignment",
                ),
                limit=self._page_size,
                row_key_prefix=(
                    f"{request.workspace_id}:{request.source_id}:"
                    f"{request.indexed_source_binding_id}:"
                ),
                cursor=cursor.document_store_cursor,
            )
        elif kind == "sequence_head":
            return self._delete_sequence_head(state)
        else:
            raise KnowledgeMaterializationPurgeError("BLOCKED_CORRUPT_STATE")

        counters = state.counters
        for record in page.documents:
            if kind == "delivery_indexes":
                counters = self._process_delivery_index(record, request, counters)
            elif kind == "remote_candidates":
                counters = self._process_remote_candidate(
                    record, request, counters, manifest_repository
                )
            elif kind == "active_pointers":
                counters = self._process_active_pointer(record, request, counters)
            elif kind == "receipts":
                counters = self._process_delivery_receipt(record, request, counters)
            else:
                counters = self._process_sequence_assignment(record, request, counters)
        if kind == "receipts":
            assert isinstance(page, ConnectedSourceDeliveryReceiptOwnershipPageV1)
            for orphan in page.orphan_index_entries:
                if self._repository.delete_connected_source_delivery_receipt_ownership_index_entry(
                    orphan
                ):
                    counters = counters.model_copy(
                        update={
                            "orphan_index_entries_deleted": (
                                counters.orphan_index_entries_deleted + 1
                            ),
                            "already_absent": counters.already_absent + 1,
                        }
                    )
        if page.next_cursor is not None:
            next_cursor = cursor.model_copy(
                update={"document_store_cursor": page.next_cursor}
            )
        else:
            next_kind = {
                "delivery_indexes": "remote_candidates",
                "remote_candidates": "active_pointers",
                "active_pointers": "receipts",
                "receipts": "assignments",
                "assignments": "sequence_head",
            }[kind]
            next_cursor = cursor.model_copy(
                update={
                    "document_store_cursor": None,
                    "record_kind": next_kind,
                }
            )
        return state.model_copy(
            update={
                "status": KnowledgeMaterializationPurgeStatusV1.DELETING,
                "updated_at": self._now(),
                "cursor": next_cursor,
                "counters": counters,
            }
        )

    def _process_delivery_index(
        self,
        record: DocumentRecord,
        request: KnowledgeMaterializationPurgeRequestV1,
        counters: KnowledgeMaterializationPurgeCountersV1,
    ) -> KnowledgeMaterializationPurgeCountersV1:
        manifest_repository = ConnectedSourceMaterializationManifestRepository(
            self._store,
            publication_authority=self._publication_authority,
        )
        data = dict(record.data)
        if (
            record.partition_key
            != f"lkw.managed_workspace:{request.tenant_id}:"
            "connected_source_materialization_manifest:immutable"
            or data.get("schema_version")
            != "lkw.connected_source_delivery_index.immutable.v1"
            or data.get("tenant_id") != request.tenant_id
            or data.get("workspace_id") != request.workspace_id
            or data.get("source_id") != request.source_id
            or data.get("indexed_source_binding_id")
            != request.indexed_source_binding_id
            or data.get("knowledge_source_binding_ref")
            != request.knowledge_source_binding_ref
            or not isinstance(data.get("delivery_id"), str)
            or not isinstance(data.get("manifest_id"), str)
            or not isinstance(data.get("manifest_fingerprint"), str)
            or not isinstance(data.get("materialization_sequence"), int)
            or record.row_key
            != manifest_repository._delivery_index_row_key(
                workspace_id=request.workspace_id,
                source_id=request.source_id,
                indexed_source_binding_id=request.indexed_source_binding_id,
                delivery_id=data["delivery_id"],
            )
        ):
            raise KnowledgeMaterializationPurgeError("BLOCKED_CORRUPT_STATE")
        self._store.delete(record.partition_key, record.row_key)
        return counters.model_copy(
            update={"delivery_indexes_deleted": counters.delivery_indexes_deleted + 1}
        )

    def _process_remote_candidate(
        self,
        record: DocumentRecord,
        request: KnowledgeMaterializationPurgeRequestV1,
        counters: KnowledgeMaterializationPurgeCountersV1,
        manifest_repository: ConnectedSourceMaterializationManifestRepository,
    ) -> KnowledgeMaterializationPurgeCountersV1:
        try:
            candidate_data = record.data.get("candidate")
            if not isinstance(candidate_data, dict):
                raise TypeError("candidate_missing")
            remote_id = candidate_data.get("remote_id")
            sequence = candidate_data.get("materialization_sequence")
            if not isinstance(remote_id, str) or not isinstance(sequence, int):
                raise TypeError("candidate_identity_missing")
            candidate = manifest_repository._get_remote_candidate(
                tenant_id=request.tenant_id,
                workspace_id=request.workspace_id,
                source_id=request.source_id,
                indexed_source_binding_id=request.indexed_source_binding_id,
                remote_id=remote_id,
                materialization_sequence=sequence,
            )
        except (TypeError, ValueError, KeyError, ConnectedSourceMaterializationManifestConflict) as exc:
            raise KnowledgeMaterializationPurgeError(
                "BLOCKED_CORRUPT_STATE"
            ) from exc
        if candidate is None or candidate.tenant_id != request.tenant_id:
            raise KnowledgeMaterializationPurgeError("BLOCKED_CORRUPT_STATE")
        if candidate.indexed_source_binding_id != request.indexed_source_binding_id:
            raise KnowledgeMaterializationPurgeError("ownership_mismatch")
        if (
            record.row_key
            != manifest_repository._remote_candidate_row_key(
                workspace_id=request.workspace_id,
                source_id=request.source_id,
                indexed_source_binding_id=request.indexed_source_binding_id,
                remote_id=candidate.remote_id,
                materialization_sequence=candidate.materialization_sequence,
            )
        ):
            raise KnowledgeMaterializationPurgeError("BLOCKED_CORRUPT_STATE")
        self._store.delete(record.partition_key, record.row_key)
        return counters.model_copy(
            update={"remote_candidates_deleted": counters.remote_candidates_deleted + 1}
        )

    def _process_active_pointer(
        self,
        record: DocumentRecord,
        request: KnowledgeMaterializationPurgeRequestV1,
        counters: KnowledgeMaterializationPurgeCountersV1,
    ) -> KnowledgeMaterializationPurgeCountersV1:
        try:
            pointer = KnowledgeMaterializationActivePointerV1.model_validate(
                dict(record.data), strict=False
            )
        except (TypeError, ValueError) as exc:
            raise KnowledgeMaterializationPurgeError(
                "BLOCKED_CORRUPT_STATE"
            ) from exc
        if (
            pointer.tenant_id != request.tenant_id
            or pointer.workspace_id != request.workspace_id
            or pointer.source_id != request.source_id
            or pointer.indexed_source_binding_id
            != request.indexed_source_binding_id
            or record.row_key
            != self._repository._active_materialization_pointer_key(
                workspace_id=request.workspace_id,
                source_id=request.source_id,
                indexed_source_binding_id=request.indexed_source_binding_id,
                remote_id=pointer.remote_id,
            )
        ):
            raise KnowledgeMaterializationPurgeError("ownership_mismatch")
        self._store.delete(record.partition_key, record.row_key)
        return counters.model_copy(
            update={"active_pointers_deleted": counters.active_pointers_deleted + 1}
        )

    def _process_delivery_receipt(
        self,
        record: DocumentRecord,
        request: KnowledgeMaterializationPurgeRequestV1,
        counters: KnowledgeMaterializationPurgeCountersV1,
    ) -> KnowledgeMaterializationPurgeCountersV1:
        try:
            receipt = ConnectedSourceDeliveryReceipt.model_validate(
                dict(record.data), strict=False
            )
        except (TypeError, ValueError) as exc:
            raise KnowledgeMaterializationPurgeError(
                "BLOCKED_CORRUPT_STATE"
            ) from exc
        if (
            receipt.tenant_id != request.tenant_id
            or receipt.workspace_id != request.workspace_id
            or receipt.source_id != request.source_id
            or receipt.indexed_source_binding_id != request.indexed_source_binding_id
            or receipt.knowledge_source_binding_ref
            != request.knowledge_source_binding_ref
        ):
            raise KnowledgeMaterializationPurgeError("ownership_mismatch")
        if receipt.materialization_sequence is None:
            raise KnowledgeMaterializationPurgeError("BLOCKED_CORRUPT_STATE")
        if (
            record.row_key
            != f"{request.workspace_id}:{request.source_id}:{receipt.delivery_id}"
        ):
            raise KnowledgeMaterializationPurgeError("BLOCKED_CORRUPT_STATE")
        assignment = self._repository.get_connected_source_delivery_sequence_assignment(
            tenant_id=request.tenant_id,
            workspace_id=request.workspace_id,
            source_id=request.source_id,
            indexed_source_binding_id=request.indexed_source_binding_id,
            delivery_id=receipt.delivery_id,
        )
        if (
            assignment is not None
            and assignment.materialization_sequence
            != receipt.materialization_sequence
        ):
            raise KnowledgeMaterializationPurgeError("BLOCKED_CORRUPT_STATE")
        index_entry = ConnectedSourceDeliveryReceiptOwnershipIndexEntryV1.for_receipt(
            receipt
        )
        self._store.delete(record.partition_key, record.row_key)
        self._repository.delete_connected_source_delivery_receipt_ownership_index_entry(
            index_entry
        )
        return counters.model_copy(
            update={"receipts_deleted": counters.receipts_deleted + 1}
        )

    def _process_sequence_assignment(
        self,
        record: DocumentRecord,
        request: KnowledgeMaterializationPurgeRequestV1,
        counters: KnowledgeMaterializationPurgeCountersV1,
    ) -> KnowledgeMaterializationPurgeCountersV1:
        try:
            assignment = ConnectedSourceDeliverySequenceAssignment.model_validate(
                dict(record.data), strict=False
            )
        except (TypeError, ValueError) as exc:
            raise KnowledgeMaterializationPurgeError(
                "BLOCKED_CORRUPT_STATE"
            ) from exc
        if (
            assignment.tenant_id != request.tenant_id
            or assignment.workspace_id != request.workspace_id
            or assignment.source_id != request.source_id
            or assignment.indexed_source_binding_id
            != request.indexed_source_binding_id
        ):
            raise KnowledgeMaterializationPurgeError("ownership_mismatch")
        receipt = self._repository.get_connected_source_delivery_receipt(
            tenant_id=request.tenant_id,
            workspace_id=request.workspace_id,
            source_id=request.source_id,
            delivery_id=assignment.delivery_id,
        )
        if (
            receipt is not None
            and receipt.materialization_sequence
            != assignment.materialization_sequence
        ):
            raise KnowledgeMaterializationPurgeError("BLOCKED_CORRUPT_STATE")
        self._store.delete(record.partition_key, record.row_key)
        return counters.model_copy(
            update={
                "sequence_assignments_deleted": (
                    counters.sequence_assignments_deleted + 1
                )
            }
        )

    def _delete_sequence_head(
        self, state: KnowledgeMaterializationPurgeStateV1
    ) -> KnowledgeMaterializationPurgeStateV1:
        request = state.request
        cursor = state.cursor
        if cursor is None:
            raise KnowledgeMaterializationPurgeError("BLOCKED_CORRUPT_STATE")
        head = self._repository.get_connected_source_delivery_sequence_head(
            tenant_id=request.tenant_id,
            workspace_id=request.workspace_id,
            source_id=request.source_id,
            indexed_source_binding_id=request.indexed_source_binding_id,
        )
        if head is not None:
            if (
                head.tenant_id,
                head.workspace_id,
                head.source_id,
                head.indexed_source_binding_id,
            ) != (
                request.tenant_id,
                request.workspace_id,
                request.source_id,
                request.indexed_source_binding_id,
            ):
                raise KnowledgeMaterializationPurgeError("ownership_mismatch")
            self._store.delete(
                _workspace_partition(
                    request.tenant_id, "connected_source_delivery_sequence_head"
                ),
                self._repository._delivery_sequence_head_key(
                    workspace_id=request.workspace_id,
                    source_id=request.source_id,
                    indexed_source_binding_id=request.indexed_source_binding_id,
                ),
            )
        return state.model_copy(
            update={
                "updated_at": self._now(),
                "cursor": cursor.model_copy(
                    update={
                        "phase": KnowledgeMaterializationPurgePhaseV1.PUBLICATION_CHAIN,
                        "record_kind": None,
                        "document_store_cursor": None,
                    }
                ),
            }
        )

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
                        update={
                            "phase": KnowledgeMaterializationPurgePhaseV1.COMPLETION_PROOF,
                            "record_kind": None,
                            "document_store_cursor": None,
                        }
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
        manifest_repository = ConnectedSourceMaterializationManifestRepository(
            self._store,
            publication_authority=self._publication_authority,
        )
        try:
            manifest_repository._load_immutable(node.descriptor)
        except ConnectedSourceMaterializationManifestConflict as exc:
            raise KnowledgeMaterializationPurgeError(
                "BLOCKED_CORRUPT_STATE"
            ) from exc
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
        self._require_legacy_migration_cleared(request)
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

    def _completion_proof(
        self, state: KnowledgeMaterializationPurgeStateV1
    ) -> KnowledgeMaterializationPurgeStateV1:
        request = state.request
        cursor = state.cursor
        assert cursor is not None
        try:
            documents = self._repository.list_document_refs_by_materialization_owner(
                tenant_id=request.tenant_id,
                workspace_id=request.workspace_id,
                source_id=request.source_id,
                indexed_source_binding_id=request.indexed_source_binding_id,
                knowledge_source_binding_ref=request.knowledge_source_binding_ref,
                limit=1,
                cursor=None,
            )
        except DocumentOwnershipIndexError as exc:
            raise KnowledgeMaterializationPurgeError(
                "BLOCKED_CORRUPT_STATE"
            ) from exc
        if documents.references or documents.orphan_index_entries:
            return state.model_copy(
                update={
                    "updated_at": self._now(),
                    "cursor": cursor.model_copy(
                        update={
                            "phase": KnowledgeMaterializationPurgePhaseV1.DOCUMENT_REFERENCES,
                            "document_store_cursor": None,
                            "record_kind": None,
                            "current_manifest_row_key": None,
                            "current_manifest_id": None,
                            "current_manifest_fingerprint": None,
                            "current_delivery_id": None,
                            "manifest_entry_offset": None,
                        }
                    ),
                }
            )
        if (
            cursor.current_manifest_id is not None
            or cursor.current_manifest_row_key is not None
            or cursor.current_manifest_fingerprint is not None
            or cursor.manifest_entry_offset is not None
        ):
            raise KnowledgeMaterializationPurgeError("BLOCKED_CORRUPT_STATE")
        for record_kind, kind_name in (
            (RecoveryRecordKindV1.INDEX_RECEIPT, "index_receipts"),
            (RecoveryRecordKindV1.DELIVERY_ACCOUNTING, "accounting"),
            (RecoveryRecordKindV1.ENQUEUE_INTENT, "enqueue_intents"),
        ):
            try:
                recovery_page = (
                    self._repository.list_connected_source_recovery_records_by_owner(
                        tenant_id=request.tenant_id,
                        workspace_id=request.workspace_id,
                        source_id=request.source_id,
                        indexed_source_binding_id=request.indexed_source_binding_id,
                        knowledge_source_binding_ref=request.knowledge_source_binding_ref,
                        record_kind=record_kind,
                        limit=1,
                        cursor=None,
                    )
                )
            except ConnectedSourceRecoveryOwnershipIndexError as exc:
                raise KnowledgeMaterializationPurgeError(
                    "BLOCKED_CORRUPT_STATE"
                ) from exc
            if recovery_page.index_entries or recovery_page.orphan_index_entries:
                return state.model_copy(
                    update={
                        "updated_at": self._now(),
                        "cursor": cursor.model_copy(
                            update={
                                "phase": KnowledgeMaterializationPurgePhaseV1.RECOVERY_RECORDS,
                                "document_store_cursor": None,
                                "record_kind": kind_name,
                                "current_manifest_row_key": None,
                                "current_manifest_id": None,
                                "current_manifest_fingerprint": None,
                                "current_delivery_id": None,
                                "manifest_entry_offset": None,
                            }
                        ),
                    }
                )
        manifest_repository = ConnectedSourceMaterializationManifestRepository(
            self._store,
            publication_authority=self._publication_authority,
        )
        partition = manifest_repository._immutable_partition(request.tenant_id)
        scope_prefix = manifest_repository._scope_prefix(
            workspace_id=request.workspace_id,
            source_id=request.source_id,
            indexed_source_binding_id=request.indexed_source_binding_id,
        )
        if self._store.query(
            partition,
            limit=1,
            row_key_prefix=scope_prefix + "manifest:",
        ).documents:
            return state.model_copy(
                update={
                    "updated_at": self._now(),
                    "cursor": cursor.model_copy(
                        update={
                            "phase": KnowledgeMaterializationPurgePhaseV1.MANIFESTS,
                            "document_store_cursor": None,
                            "record_kind": None,
                            "current_manifest_row_key": None,
                            "current_manifest_id": None,
                            "current_manifest_fingerprint": None,
                            "current_delivery_id": None,
                            "manifest_entry_offset": None,
                        }
                    ),
                }
            )
        for kind, prefix in (
            ("delivery_indexes", scope_prefix + "delivery:"),
            ("remote_candidates", scope_prefix + "remote:"),
        ):
            if self._store.query(partition, limit=1, row_key_prefix=prefix).documents:
                return state.model_copy(
                    update={
                        "updated_at": self._now(),
                        "cursor": cursor.model_copy(
                            update={
                                "phase": KnowledgeMaterializationPurgePhaseV1.DELIVERY_RECORDS,
                                "document_store_cursor": None,
                                "record_kind": kind,
                            }
                        ),
                    }
                )
        for kind, partition_key, prefix in (
            (
                "active_pointers",
                _workspace_partition(
                    request.tenant_id, "materialization_active_pointer"
                ),
                (
                    f"{request.workspace_id}:{request.source_id}:"
                    f"{request.indexed_source_binding_id}:"
                ),
            ),
            (
                "assignments",
                _workspace_partition(
                    request.tenant_id,
                    "connected_source_delivery_sequence_assignment",
                ),
                (
                    f"{request.workspace_id}:{request.source_id}:"
                    f"{request.indexed_source_binding_id}:"
                ),
            ),
        ):
            if self._store.query(
                partition_key, limit=1, row_key_prefix=prefix
            ).documents:
                return state.model_copy(
                    update={
                        "updated_at": self._now(),
                        "cursor": cursor.model_copy(
                            update={
                                "phase": KnowledgeMaterializationPurgePhaseV1.DELIVERY_RECORDS,
                                "document_store_cursor": None,
                                "record_kind": kind,
                            }
                        ),
                    }
                )
        try:
            receipt_proof = (
                self._repository.list_connected_source_delivery_receipts_by_owner(
                    tenant_id=request.tenant_id,
                    workspace_id=request.workspace_id,
                    source_id=request.source_id,
                    indexed_source_binding_id=request.indexed_source_binding_id,
                    knowledge_source_binding_ref=request.knowledge_source_binding_ref,
                    limit=1,
                )
            )
        except ConnectedSourceDeliveryReceiptOwnershipIndexError as exc:
            raise KnowledgeMaterializationPurgeError(
                "BLOCKED_CORRUPT_STATE"
            ) from exc
        if receipt_proof.documents or receipt_proof.orphan_index_entries:
            return state.model_copy(
                update={
                    "updated_at": self._now(),
                    "cursor": cursor.model_copy(
                        update={
                            "phase": KnowledgeMaterializationPurgePhaseV1.DELIVERY_RECORDS,
                            "document_store_cursor": None,
                            "record_kind": "receipts",
                        }
                    ),
                }
            )
        if self._repository.get_connected_source_delivery_sequence_head(
            tenant_id=request.tenant_id,
            workspace_id=request.workspace_id,
            source_id=request.source_id,
            indexed_source_binding_id=request.indexed_source_binding_id,
        ) is not None:
            return state.model_copy(
                update={
                    "updated_at": self._now(),
                    "cursor": cursor.model_copy(
                        update={
                            "phase": KnowledgeMaterializationPurgePhaseV1.DELIVERY_RECORDS,
                            "document_store_cursor": None,
                            "record_kind": "sequence_head",
                        }
                    ),
                }
            )
        head = self._publication_authority.read_publication_head(
            tenant_id=request.tenant_id,
            binding_id=request.knowledge_source_binding_ref,
        )
        if head is not None:
            self._validate_publication_node(
                self._publication_authority.read_publication_commit_node(
                    tenant_id=request.tenant_id,
                    binding_id=request.knowledge_source_binding_ref,
                    commit_id=head.publication_commit_id,
                ),
                request,
            )
            return state.model_copy(
                update={
                    "updated_at": self._now(),
                    "cursor": cursor.model_copy(
                        update={
                            "phase": KnowledgeMaterializationPurgePhaseV1.PUBLICATION_CHAIN,
                            "publication_commit_id": head.publication_commit_id,
                        }
                    ),
                }
            )
        self._require_legacy_migration_cleared(request)
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

    def _require_legacy_migration_cleared(
        self,
        request: KnowledgeMaterializationPurgeRequestV1,
    ) -> None:
        try:
            gate = self._repository.get_connected_source_recovery_migration_gate(
                tenant_id=request.tenant_id,
                workspace_id=request.workspace_id,
                source_id=request.source_id,
                indexed_source_binding_id=request.indexed_source_binding_id,
                knowledge_source_binding_ref=request.knowledge_source_binding_ref,
            )
        except ConnectedSourceRecoveryMigrationGateError as exc:
            raise KnowledgeMaterializationPurgeError(
                "BLOCKED_CORRUPT_STATE"
            ) from exc
        if gate is None:
            raise KnowledgeMaterializationPurgeError("BLOCKED_LEGACY_MIGRATION")
        if gate.ownership_scope != request.ownership_scope:
            raise KnowledgeMaterializationPurgeError("BLOCKED_CORRUPT_STATE")
        if gate.status is ConnectedSourceRecoveryMigrationGateStatusV1.REQUIRED:
            raise KnowledgeMaterializationPurgeError("BLOCKED_LEGACY_MIGRATION")
        if gate.status is not ConnectedSourceRecoveryMigrationGateStatusV1.CLEARED:
            raise KnowledgeMaterializationPurgeError("BLOCKED_CORRUPT_STATE")
        if gate.cleared_at is None or gate.evidence_revision is None:
            raise KnowledgeMaterializationPurgeError("BLOCKED_CORRUPT_STATE")

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
                phase=KnowledgeMaterializationPurgePhaseV1.DOCUMENT_REFERENCES,
                purge_id=purge_id,
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
