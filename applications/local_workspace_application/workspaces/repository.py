# © Artur Czarnecki. All rights reserved.

"""DocumentStore-backed persistence for managed workspaces (LKW-PRODUCT-1)."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from local_workspace_application.workspaces.connected_source_models import (
    ConnectedSourceDeliveryReceipt,
    ConnectedSourceOperationDeliveryAccounting,
    ConnectedSourceSyncEnqueueIntent,
)
from local_workspace_application.workspaces.knowledge_configuration_models import (
    WorkspaceCommittedQueryPolicy,
    WorkspaceConnectionAttachment,
    WorkspaceIndexedSourceBinding,
    WorkspaceKnowledgeConfigurationHead,
    WorkspaceKnowledgeMutationOperationV1,
    WorkspaceKnowledgeMutationRecord,
    WorkspaceLiveAccessBinding,
    parse_workspace_query_policy,
)
from local_workspace_application.workspaces.materialization_visibility import (
    KnowledgeMaterializationActivePointerV1,
)
from local_workspace_application.workspaces.models import (
    ActiveKnowledgeIngestionLocator,
    IntakeBatch,
    KnowledgeInput,
    ManagedFileObject,
    WebUrlSourceLocator,
    Workspace,
    WorkspaceDocumentReference,
    WorkspaceOperation,
    WorkspaceOperationStatus,
    WorkspaceOperationType,
    WorkspaceSource,
)
from pydantic import BaseModel

from intergrax.integrations.contracts.document_store import (
    ConditionalDocumentStore,
    DocumentRecord,
    DocumentStore,
)

T = TypeVar("T", bound=BaseModel)

_IDEMPOTENCY_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_QUERY_POLICY_ENTITY_ID = "query-policy"
_REVISION_ROW_KEY_MARKER = ":rev:"
_KNOWLEDGE_CONFIGURATION_MAX_REVISION = 10**20 - 1
_KNOWLEDGE_CONFIGURATION_REVISION_SCAN_LIMIT = 2000
_KNOWLEDGE_CONFIGURATION_MUTATION_SCAN_LIMIT = 2000
_KNOWLEDGE_CONFIGURATION_MUTATION_SCAN_PROBE_LIMIT = 2001

_ENTITY_KNOWLEDGE_CONFIGURATION_HEAD = "knowledge_configuration_head"
_ENTITY_KNOWLEDGE_CONFIGURATION_MUTATION = "knowledge_configuration_mutation"
_ENTITY_KNOWLEDGE_CONFIGURATION_CONNECTION_ATTACHMENT = (
    "knowledge_configuration_connection_attachment"
)
_ENTITY_KNOWLEDGE_CONFIGURATION_INDEXED_SOURCE = "knowledge_configuration_indexed_source"
_ENTITY_KNOWLEDGE_CONFIGURATION_LIVE_ACCESS = "knowledge_configuration_live_access"
_ENTITY_KNOWLEDGE_CONFIGURATION_QUERY_POLICY = "knowledge_configuration_query_policy"
_ENTITY_CONNECTED_SOURCE_DELIVERY = "connected_source_delivery"
_ENTITY_CONNECTED_SOURCE_SYNC_ENQUEUE = "connected_source_sync_enqueue"


class WorkspaceKnowledgeConfigurationRepositoryError(RuntimeError):
    def __init__(self, error_code: str) -> None:
        super().__init__(error_code)
        self.error_code = error_code


@dataclass(frozen=True)
class ActiveKnowledgeIngestionLocatorScan:
    locators: tuple[ActiveKnowledgeIngestionLocator, ...]
    malformed_seen: int
    malformed_removed: int

_ENTITY_WORKSPACE = "workspace"
_ENTITY_SOURCE = "source"
_ENTITY_OPERATION = "operation"
_ENTITY_DOCUMENT = "document"
_ENTITY_KNOWLEDGE_INPUT = "knowledge_input"
_ENTITY_MANAGED_FILE = "managed_file"
_ENTITY_WEB_URL_LOCATOR = "web_url_locator"
_ENTITY_INTAKE_BATCH = "intake_batch"
_ENTITY_CONNECTED_SOURCE_DELIVERY_RECEIPT = "connected_source_delivery_receipt"
_ENTITY_MATERIALIZATION_ACTIVE_POINTER = "materialization_active_pointer"
_ACTIVE_KNOWLEDGE_INGESTION_PARTITION = "lkw.managed_workspace:active_knowledge_ingestion"


def _partition(tenant_id: str, entity: str) -> str:
    return f"lkw.managed_workspace:{tenant_id}:{entity}"


def _revision_row_key(
    *,
    workspace_id: str,
    entity_id: str,
    revision: int,
) -> str:
    if revision < 1 or revision > _KNOWLEDGE_CONFIGURATION_MAX_REVISION:
        raise ValueError("knowledge_configuration_revision_invalid")
    return f"{workspace_id}:{entity_id}{_REVISION_ROW_KEY_MARKER}{revision:020d}"


def _mutation_row_key(
    *,
    workspace_id: str,
    operation: str,
    idempotency_key_hash: str,
) -> str:
    return f"{workspace_id}:{operation}:{idempotency_key_hash}"


def _parse_mutation_row_key(row_key: str) -> tuple[str, str, str]:
    hash_sep = row_key.rfind(":")
    if hash_sep < 0:
        raise ValueError("knowledge_configuration_record_identity_mismatch")
    idempotency_key_hash = row_key[hash_sep + 1 :]
    prefix = row_key[:hash_sep]
    colon_idx = prefix.find(":")
    if colon_idx < 0:
        raise ValueError("knowledge_configuration_record_identity_mismatch")
    workspace_id = prefix[:colon_idx]
    operation = prefix[colon_idx + 1 :]
    return workspace_id, operation, idempotency_key_hash


def _parse_revision_row_key(row_key: str) -> tuple[str, str, int]:
    if _REVISION_ROW_KEY_MARKER not in row_key:
        raise ValueError("knowledge_configuration_record_identity_mismatch")
    prefix, rev_part = row_key.rsplit(_REVISION_ROW_KEY_MARKER, 1)
    if len(rev_part) != 20 or not rev_part.isdigit():
        raise ValueError("knowledge_configuration_record_identity_mismatch")
    revision = int(rev_part)
    if revision < 1:
        raise ValueError("knowledge_configuration_record_identity_mismatch")
    colon_idx = prefix.find(":")
    if colon_idx < 0:
        raise ValueError("knowledge_configuration_record_identity_mismatch")
    workspace_id = prefix[:colon_idx]
    entity_id = prefix[colon_idx + 1 :]
    return workspace_id, entity_id, revision


def _assert_record_identity(
    *,
    tenant_id: str,
    workspace_id: str,
    model_tenant_id: str,
    model_workspace_id: str,
    entity_id: str | None = None,
    model_entity_id: str | None = None,
    revision: int | None = None,
    model_revision: int | None = None,
) -> None:
    if (
        model_tenant_id != tenant_id
        or model_workspace_id != workspace_id
        or (entity_id is not None and model_entity_id != entity_id)
        or (revision is not None and model_revision != revision)
    ):
        raise ValueError("knowledge_configuration_record_identity_mismatch")


def _validate_idempotency_key_hash(idempotency_key_hash: str) -> None:
    if _IDEMPOTENCY_HASH_RE.fullmatch(idempotency_key_hash) is None:
        raise WorkspaceKnowledgeConfigurationRepositoryError(
            "knowledge_configuration_idempotency_hash_invalid"
        )


def _to_document_record(
    model: BaseModel,
    *,
    partition_key: str,
    row_key: str,
) -> DocumentRecord:
    return DocumentRecord(
        partition_key=partition_key,
        row_key=row_key,
        data=model.model_dump(mode="json"),
    )


class ManagedWorkspaceRepository:
    """Tier-3 repository over the provider-neutral DocumentStore contract."""

    def __init__(self, document_store: DocumentStore) -> None:
        self._store = document_store

    @property
    def document_store(self) -> DocumentStore:
        return self._store

    def _put(self, partition_key: str, row_key: str, model: BaseModel) -> None:
        self._store.put(
            DocumentRecord(
                partition_key=partition_key,
                row_key=row_key,
                data=model.model_dump(mode="json"),
            )
        )

    def _get(self, partition_key: str, row_key: str, model_type: type[T]) -> T | None:
        record = self._store.get(partition_key, row_key)
        if record is None:
            return None
        return model_type.model_validate(dict(record.data))

    def _list(self, partition_key: str, model_type: type[T], *, limit: int = 500) -> list[T]:
        result = self._store.query(partition_key, limit=limit)
        return [model_type.model_validate(dict(doc.data)) for doc in result.documents]

    # --- Workspace ---

    def put_workspace(self, workspace: Workspace) -> Workspace:
        self._put(
            _partition(workspace.tenant_id, _ENTITY_WORKSPACE),
            workspace.workspace_id,
            workspace,
        )
        return workspace

    def get_workspace(self, *, tenant_id: str, workspace_id: str) -> Workspace | None:
        return self._get(
            _partition(tenant_id, _ENTITY_WORKSPACE),
            workspace_id,
            Workspace,
        )

    def delete_workspace(self, *, tenant_id: str, workspace_id: str) -> None:
        self._store.delete(_partition(tenant_id, _ENTITY_WORKSPACE), workspace_id)

    def list_workspaces(self, *, tenant_id: str) -> list[Workspace]:
        items = self._list(_partition(tenant_id, _ENTITY_WORKSPACE), Workspace)
        return sorted(items, key=lambda item: item.created_at, reverse=True)

    # --- Source ---

    def put_source(self, source: WorkspaceSource) -> WorkspaceSource:
        self._put(
            _partition(source.tenant_id, _ENTITY_SOURCE),
            f"{source.workspace_id}:{source.source_id}",
            source,
        )
        return source

    def put_source_if_absent(self, source: WorkspaceSource) -> bool:
        return self._put_if_absent(
            source,
            partition_key=_partition(source.tenant_id, _ENTITY_SOURCE),
            row_key=f"{source.workspace_id}:{source.source_id}",
        )

    def delete_source_if_match(self, source: WorkspaceSource) -> bool:
        return self._delete_if_match(
            source,
            partition_key=_partition(source.tenant_id, _ENTITY_SOURCE),
            row_key=f"{source.workspace_id}:{source.source_id}",
        )

    def get_source(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        source_id: str,
    ) -> WorkspaceSource | None:
        return self._get(
            _partition(tenant_id, _ENTITY_SOURCE),
            f"{workspace_id}:{source_id}",
            WorkspaceSource,
        )

    def list_sources(self, *, tenant_id: str, workspace_id: str) -> list[WorkspaceSource]:
        prefix = f"{workspace_id}:"
        result = self._store.query(
            _partition(tenant_id, _ENTITY_SOURCE),
            limit=500,
            row_key_prefix=prefix,
        )
        items = [WorkspaceSource.model_validate(dict(doc.data)) for doc in result.documents]
        return sorted(items, key=lambda item: item.created_at)

    def delete_source(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        source_id: str,
    ) -> None:
        self._store.delete(
            _partition(tenant_id, _ENTITY_SOURCE),
            f"{workspace_id}:{source_id}",
        )

    def delete_sources_for_workspace(self, *, tenant_id: str, workspace_id: str) -> int:
        deleted = 0
        for source in self.list_sources(tenant_id=tenant_id, workspace_id=workspace_id):
            self.delete_source(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                source_id=source.source_id,
            )
            deleted += 1
        return deleted

    # --- Connected source delivery receipts ---

    def put_connected_source_delivery_receipt(
        self,
        receipt: ConnectedSourceDeliveryReceipt,
    ) -> ConnectedSourceDeliveryReceipt:
        partition_key = _partition(receipt.tenant_id, _ENTITY_CONNECTED_SOURCE_DELIVERY_RECEIPT)
        row_key = (
            f"{receipt.workspace_id}:{receipt.source_id}:{receipt.delivery_id}"
        )
        self._put(partition_key, row_key, receipt)
        return receipt

    def put_connected_source_delivery_receipt_if_absent(
        self,
        receipt: ConnectedSourceDeliveryReceipt,
    ) -> bool:
        partition_key = _partition(receipt.tenant_id, _ENTITY_CONNECTED_SOURCE_DELIVERY_RECEIPT)
        row_key = (
            f"{receipt.workspace_id}:{receipt.source_id}:{receipt.delivery_id}"
        )
        return self._put_if_absent(receipt, partition_key=partition_key, row_key=row_key)

    def complete_connected_source_delivery_receipt_if_in_progress(
        self,
        *,
        expected: ConnectedSourceDeliveryReceipt,
        replacement: ConnectedSourceDeliveryReceipt,
    ) -> bool:
        partition_key = _partition(expected.tenant_id, _ENTITY_CONNECTED_SOURCE_DELIVERY_RECEIPT)
        row_key = f"{expected.workspace_id}:{expected.source_id}:{expected.delivery_id}"
        return self._replace_if_match(
            expected=expected,
            replacement=replacement,
            partition_key=partition_key,
            row_key=row_key,
        )

    def get_connected_source_delivery_receipt(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        source_id: str,
        delivery_id: str,
    ) -> ConnectedSourceDeliveryReceipt | None:
        return self._get(
            _partition(tenant_id, _ENTITY_CONNECTED_SOURCE_DELIVERY_RECEIPT),
            f"{workspace_id}:{source_id}:{delivery_id}",
            ConnectedSourceDeliveryReceipt,
        )

    def put_active_materialization_pointer_if_absent(
        self,
        pointer: KnowledgeMaterializationActivePointerV1,
    ) -> bool:
        return self._put_if_absent(
            pointer,
            partition_key=_partition(
                pointer.tenant_id, _ENTITY_MATERIALIZATION_ACTIVE_POINTER
            ),
            row_key=self._active_materialization_pointer_key(
                workspace_id=pointer.workspace_id,
                source_id=pointer.source_id,
                indexed_source_binding_id=pointer.indexed_source_binding_id,
                remote_id=pointer.remote_id,
            ),
        )

    def get_active_materialization_pointer(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        source_id: str,
        indexed_source_binding_id: str,
        remote_id: str,
    ) -> KnowledgeMaterializationActivePointerV1 | None:
        return self._get(
            _partition(tenant_id, _ENTITY_MATERIALIZATION_ACTIVE_POINTER),
            self._active_materialization_pointer_key(
                workspace_id=workspace_id,
                source_id=source_id,
                indexed_source_binding_id=indexed_source_binding_id,
                remote_id=remote_id,
            ),
            KnowledgeMaterializationActivePointerV1,
        )

    def replace_active_materialization_pointer(
        self,
        *,
        expected: KnowledgeMaterializationActivePointerV1,
        replacement: KnowledgeMaterializationActivePointerV1,
    ) -> bool:
        if (
            expected.tenant_id != replacement.tenant_id
            or self._active_materialization_pointer_key(
                workspace_id=expected.workspace_id,
                source_id=expected.source_id,
                indexed_source_binding_id=expected.indexed_source_binding_id,
                remote_id=expected.remote_id,
            )
            != self._active_materialization_pointer_key(
                workspace_id=replacement.workspace_id,
                source_id=replacement.source_id,
                indexed_source_binding_id=replacement.indexed_source_binding_id,
                remote_id=replacement.remote_id,
            )
        ):
            raise ValueError("active_materialization_pointer_identity_mismatch")
        return self._replace_if_match(
            expected=expected,
            replacement=replacement,
            partition_key=_partition(
                expected.tenant_id, _ENTITY_MATERIALIZATION_ACTIVE_POINTER
            ),
            row_key=self._active_materialization_pointer_key(
                workspace_id=expected.workspace_id,
                source_id=expected.source_id,
                indexed_source_binding_id=expected.indexed_source_binding_id,
                remote_id=expected.remote_id,
            ),
        )

    @staticmethod
    def _active_materialization_pointer_key(
        *,
        workspace_id: str,
        source_id: str,
        indexed_source_binding_id: str,
        remote_id: str,
    ) -> str:
        return (
            f"{workspace_id}:{source_id}:"
            f"{indexed_source_binding_id}:{remote_id}"
        )

    def delete_connected_source_delivery_receipt(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        source_id: str,
        delivery_id: str,
    ) -> None:
        self._store.delete(
            _partition(tenant_id, _ENTITY_CONNECTED_SOURCE_DELIVERY_RECEIPT),
            f"{workspace_id}:{source_id}:{delivery_id}",
        )

    # --- Connected source delivery accounting ---

    def put_connected_source_delivery_accounting_if_absent(
        self,
        accounting: ConnectedSourceOperationDeliveryAccounting,
    ) -> bool:
        partition_key = _partition(accounting.tenant_id, _ENTITY_CONNECTED_SOURCE_DELIVERY)
        row_key = f"{accounting.operation_id}:{accounting.delivery_id}"
        return self._put_if_absent(
            accounting,
            partition_key=partition_key,
            row_key=row_key,
        )

    def get_connected_source_delivery_accounting(
        self,
        *,
        tenant_id: str,
        operation_id: str,
        delivery_id: str,
    ) -> ConnectedSourceOperationDeliveryAccounting | None:
        return self._get(
            _partition(tenant_id, _ENTITY_CONNECTED_SOURCE_DELIVERY),
            f"{operation_id}:{delivery_id}",
            ConnectedSourceOperationDeliveryAccounting,
        )

    def delete_connected_source_delivery_accounting(
        self,
        *,
        tenant_id: str,
        operation_id: str,
        delivery_id: str,
    ) -> None:
        self._store.delete(
            _partition(tenant_id, _ENTITY_CONNECTED_SOURCE_DELIVERY),
            f"{operation_id}:{delivery_id}",
        )

    def list_connected_source_delivery_accounting(
        self,
        *,
        tenant_id: str,
        operation_id: str,
    ) -> list[ConnectedSourceOperationDeliveryAccounting]:
        result = self._store.query(
            _partition(tenant_id, _ENTITY_CONNECTED_SOURCE_DELIVERY),
            limit=500,
            row_key_prefix=f"{operation_id}:",
        )
        items = [
            ConnectedSourceOperationDeliveryAccounting.model_validate(dict(doc.data))
            for doc in result.documents
        ]
        return sorted(items, key=lambda item: (item.accounted_at, item.delivery_id))

    # --- Connected source sync enqueue intent ---

    def put_connected_source_sync_enqueue_intent(
        self,
        intent: ConnectedSourceSyncEnqueueIntent,
    ) -> ConnectedSourceSyncEnqueueIntent:
        partition_key = _partition(intent.tenant_id, _ENTITY_CONNECTED_SOURCE_SYNC_ENQUEUE)
        self._put(partition_key, intent.operation_id, intent)
        return intent

    def put_connected_source_sync_enqueue_intent_if_absent(
        self,
        intent: ConnectedSourceSyncEnqueueIntent,
    ) -> bool:
        partition_key = _partition(intent.tenant_id, _ENTITY_CONNECTED_SOURCE_SYNC_ENQUEUE)
        return self._put_if_absent(
            intent,
            partition_key=partition_key,
            row_key=intent.operation_id,
        )

    def allocate_connected_source_sync_enqueue_generation(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        source_id: str,
        operation_id: str,
        max_attempts: int = 3,
    ) -> ConnectedSourceSyncEnqueueIntent:
        from datetime import UTC, datetime

        partition_key = _partition(tenant_id, _ENTITY_CONNECTED_SOURCE_SYNC_ENQUEUE)
        for _ in range(max_attempts):
            existing = self.get_connected_source_sync_enqueue_intent(
                tenant_id=tenant_id,
                operation_id=operation_id,
            )
            now = datetime.now(UTC)
            if existing is None:
                intent = ConnectedSourceSyncEnqueueIntent(
                    tenant_id=tenant_id,
                    workspace_id=workspace_id,
                    source_id=source_id,
                    operation_id=operation_id,
                    enqueue_generation=1,
                    last_enqueued_generation=0,
                    updated_at=now,
                )
                if self.put_connected_source_sync_enqueue_intent_if_absent(intent):
                    return intent
                continue
            updated = existing.model_copy(
                update={
                    "enqueue_generation": existing.enqueue_generation + 1,
                    "updated_at": now,
                }
            )
            if self._replace_if_match(
                expected=existing,
                replacement=updated,
                partition_key=partition_key,
                row_key=operation_id,
            ):
                return updated
        raise RuntimeError("connected_source_enqueue_generation_allocation_failed")

    def get_connected_source_sync_enqueue_intent(
        self,
        *,
        tenant_id: str,
        operation_id: str,
    ) -> ConnectedSourceSyncEnqueueIntent | None:
        return self._get(
            _partition(tenant_id, _ENTITY_CONNECTED_SOURCE_SYNC_ENQUEUE),
            operation_id,
            ConnectedSourceSyncEnqueueIntent,
        )

    def mark_connected_source_sync_enqueued(
        self,
        *,
        tenant_id: str,
        operation_id: str,
        expected_generation: int,
        task_id: str,
        queue_provider: str,
    ) -> bool:
        intent = self.get_connected_source_sync_enqueue_intent(
            tenant_id=tenant_id,
            operation_id=operation_id,
        )
        if intent is None or intent.enqueue_generation != expected_generation:
            return False
        if intent.last_enqueued_generation >= expected_generation:
            return True
        from datetime import UTC, datetime

        updated = intent.model_copy(
            update={
                "last_enqueued_generation": expected_generation,
                "last_task_id": task_id,
                "last_queue_provider": queue_provider,
                "updated_at": datetime.now(UTC),
            }
        )
        return self._replace_if_match(
            expected=intent,
            replacement=updated,
            partition_key=_partition(tenant_id, _ENTITY_CONNECTED_SOURCE_SYNC_ENQUEUE),
            row_key=operation_id,
        )

    def delete_connected_source_sync_enqueue_intent(
        self,
        *,
        tenant_id: str,
        operation_id: str,
    ) -> None:
        self._store.delete(
            _partition(tenant_id, _ENTITY_CONNECTED_SOURCE_SYNC_ENQUEUE),
            operation_id,
        )

    # --- Operation ---

    def put_operation(self, operation: WorkspaceOperation) -> WorkspaceOperation:
        self._put(
            _partition(operation.tenant_id, _ENTITY_OPERATION),
            operation.operation_id,
            operation,
        )
        return operation

    def replace_operation_if_match(
        self,
        *,
        expected: WorkspaceOperation,
        replacement: WorkspaceOperation,
    ) -> bool:
        return self._replace_if_match(
            expected=expected,
            replacement=replacement,
            partition_key=_partition(expected.tenant_id, _ENTITY_OPERATION),
            row_key=expected.operation_id,
        )

    def claim_operation_if_queued(
        self,
        *,
        expected: WorkspaceOperation,
        replacement: WorkspaceOperation,
    ) -> bool:
        partition_key = _partition(expected.tenant_id, _ENTITY_OPERATION)
        return self._replace_if_match(
            expected=expected,
            replacement=replacement,
            partition_key=partition_key,
            row_key=expected.operation_id,
        )

    def get_operation(self, *, tenant_id: str, operation_id: str) -> WorkspaceOperation | None:
        return self._get(
            _partition(tenant_id, _ENTITY_OPERATION),
            operation_id,
            WorkspaceOperation,
        )

    def delete_operation(self, *, tenant_id: str, operation_id: str) -> None:
        self._store.delete(_partition(tenant_id, _ENTITY_OPERATION), operation_id)

    def delete_operations_for_workspace(self, *, tenant_id: str, workspace_id: str) -> int:
        deleted = 0
        for operation in self.list_operations(tenant_id=tenant_id):
            if operation.workspace_id != workspace_id:
                continue
            self.delete_operation(tenant_id=tenant_id, operation_id=operation.operation_id)
            deleted += 1
        return deleted

    # --- Document references ---

    def put_document_ref(self, ref: WorkspaceDocumentReference) -> WorkspaceDocumentReference:
        self._put(
            _partition(ref.tenant_id, _ENTITY_DOCUMENT),
            f"{ref.workspace_id}:{ref.document_id}",
            ref,
        )
        # Secondary index by path identity for idempotent sync lookups.
        path_row = f"path:{ref.workspace_id}:{ref.source_id}:{ref.source_path}"
        self._store.put(
            DocumentRecord(
                partition_key=_partition(ref.tenant_id, _ENTITY_DOCUMENT),
                row_key=path_row,
                data={"document_id": ref.document_id, "content_hash": ref.content_hash},
            )
        )
        return ref

    def get_document_ref_by_path(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        source_id: str,
        source_path: str,
    ) -> WorkspaceDocumentReference | None:
        path_row = f"path:{workspace_id}:{source_id}:{source_path}"
        index = self._store.get(_partition(tenant_id, _ENTITY_DOCUMENT), path_row)
        if index is None:
            return None
        document_id = str(index.data.get("document_id") or "").strip()
        if not document_id:
            return None
        return self._get(
            _partition(tenant_id, _ENTITY_DOCUMENT),
            f"{workspace_id}:{document_id}",
            WorkspaceDocumentReference,
        )

    def list_document_refs(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> list[WorkspaceDocumentReference]:
        result = self._store.query(
            _partition(tenant_id, _ENTITY_DOCUMENT),
            limit=2000,
            row_key_prefix=f"{workspace_id}:",
        )
        refs: list[WorkspaceDocumentReference] = []
        for doc in result.documents:
            if str(doc.row_key).startswith(f"{workspace_id}:") and not str(doc.row_key).startswith(
                f"{workspace_id}:path:"
            ):
                # skip path index rows which use path: prefix under same partition
                if "document_id" in doc.data and "source_path" in doc.data:
                    refs.append(WorkspaceDocumentReference.model_validate(dict(doc.data)))
        return refs

    def get_document_ref(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        document_id: str,
    ) -> WorkspaceDocumentReference | None:
        return self._get(
            _partition(tenant_id, _ENTITY_DOCUMENT),
            f"{workspace_id}:{document_id}",
            WorkspaceDocumentReference,
        )

    def delete_document_refs_for_workspace(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> int:
        """Remove primary refs and path-index rows for one workspace."""
        partition = _partition(tenant_id, _ENTITY_DOCUMENT)
        deleted = 0
        refs = self.list_document_refs(tenant_id=tenant_id, workspace_id=workspace_id)
        for ref in refs:
            self._store.delete(partition, f"{workspace_id}:{ref.document_id}")
            path_row = f"path:{workspace_id}:{ref.source_id}:{ref.source_path}"
            self._store.delete(partition, path_row)
            deleted += 1
        # Sweep any leftover path-index rows for this workspace.
        path_prefix = f"path:{workspace_id}:"
        leftover = self._store.query(partition, limit=2000, row_key_prefix=path_prefix)
        for doc in leftover.documents:
            self._store.delete(partition, doc.row_key)
            deleted += 1
        return deleted

    def list_operations(self, *, tenant_id: str) -> list[WorkspaceOperation]:
        return self._list(_partition(tenant_id, _ENTITY_OPERATION), WorkspaceOperation)

    def list_ingestion_operations(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        statuses: set[WorkspaceOperationStatus] | None = None,
    ) -> list[WorkspaceOperation]:
        items = [
            op
            for op in self.list_operations(tenant_id=tenant_id)
            if op.workspace_id == workspace_id
            and op.operation_type is WorkspaceOperationType.KNOWLEDGE_INGESTION
            and (statuses is None or op.status in statuses)
        ]
        return sorted(items, key=lambda item: item.created_at or item.operation_id)

    # --- Knowledge Input ---

    def put_knowledge_input(self, knowledge_input: KnowledgeInput) -> KnowledgeInput:
        self._put(
            _partition(knowledge_input.tenant_id, _ENTITY_KNOWLEDGE_INPUT),
            f"{knowledge_input.workspace_id}:{knowledge_input.input_id}",
            knowledge_input,
        )
        return knowledge_input

    def get_knowledge_input(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        input_id: str,
    ) -> KnowledgeInput | None:
        return self._get(
            _partition(tenant_id, _ENTITY_KNOWLEDGE_INPUT),
            f"{workspace_id}:{input_id}",
            KnowledgeInput,
        )

    def list_knowledge_inputs(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> list[KnowledgeInput]:
        result = self._store.query(
            _partition(tenant_id, _ENTITY_KNOWLEDGE_INPUT),
            limit=500,
            row_key_prefix=f"{workspace_id}:",
        )
        items = [KnowledgeInput.model_validate(dict(doc.data)) for doc in result.documents]
        return sorted(items, key=lambda item: item.created_at)

    def delete_knowledge_inputs_for_workspace(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> int:
        deleted = 0
        for knowledge_input in self.list_knowledge_inputs(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        ):
            self._store.delete(
                _partition(tenant_id, _ENTITY_KNOWLEDGE_INPUT),
                f"{workspace_id}:{knowledge_input.input_id}",
            )
            deleted += 1
        return deleted

    def find_active_sync_operation(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        source_id: str,
    ) -> WorkspaceOperation | None:
        active = {
            WorkspaceOperationStatus.QUEUED,
            WorkspaceOperationStatus.RUNNING,
        }
        candidates = [
            op
            for op in self.list_operations(tenant_id=tenant_id)
            if op.workspace_id == workspace_id
            and op.source_id == source_id
            and op.operation_type is WorkspaceOperationType.SOURCE_SYNC
            and op.status in active
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda item: item.started_at or item.completed_at or item.operation_id)
        return candidates[0]

    def mark_running_operations_failed_for_tenant(
        self,
        *,
        tenant_id: str,
        error: str = "interrupted_by_host_restart",
    ) -> int:
        from datetime import UTC, datetime

        now = datetime.now(UTC)
        recovered = 0
        for operation in self.list_operations(tenant_id=tenant_id):
            if operation.status is not WorkspaceOperationStatus.RUNNING:
                continue
            self.put_operation(
                operation.model_copy(
                    update={
                        "status": WorkspaceOperationStatus.FAILED,
                        "error": error,
                        "completed_at": now,
                    }
                )
            )
            recovered += 1
        return recovered

    # --- Managed file ---

    def put_managed_file(self, managed_file: ManagedFileObject) -> ManagedFileObject:
        self._put(
            _partition(managed_file.tenant_id, _ENTITY_MANAGED_FILE),
            f"{managed_file.workspace_id}:{managed_file.input_id}",
            managed_file,
        )
        return managed_file

    def get_managed_file(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        input_id: str,
    ) -> ManagedFileObject | None:
        return self._get(
            _partition(tenant_id, _ENTITY_MANAGED_FILE),
            f"{workspace_id}:{input_id}",
            ManagedFileObject,
        )

    def list_managed_files(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> list[ManagedFileObject]:
        result = self._store.query(
            _partition(tenant_id, _ENTITY_MANAGED_FILE),
            limit=2000,
            row_key_prefix=f"{workspace_id}:",
        )
        items = [ManagedFileObject.model_validate(dict(doc.data)) for doc in result.documents]
        return sorted(items, key=lambda item: item.created_at)

    def delete_managed_file(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        input_id: str,
    ) -> None:
        self._store.delete(
            _partition(tenant_id, _ENTITY_MANAGED_FILE),
            f"{workspace_id}:{input_id}",
        )

    def delete_managed_files_for_workspace(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> int:
        deleted = 0
        for managed_file in self.list_managed_files(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        ):
            self.delete_managed_file(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                input_id=managed_file.input_id,
            )
            deleted += 1
        return deleted

    # --- Web URL locators ---

    def put_web_url_locator(self, locator: WebUrlSourceLocator) -> WebUrlSourceLocator:
        self._put(
            _partition(locator.tenant_id, _ENTITY_WEB_URL_LOCATOR),
            f"{locator.workspace_id}:{locator.requested_url_fingerprint}",
            locator,
        )
        return locator

    def get_web_url_locator(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        requested_url_fingerprint: str,
    ) -> WebUrlSourceLocator | None:
        return self._get(
            _partition(tenant_id, _ENTITY_WEB_URL_LOCATOR),
            f"{workspace_id}:{requested_url_fingerprint}",
            WebUrlSourceLocator,
        )

    def list_web_url_locators(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> list[WebUrlSourceLocator]:
        result = self._store.query(
            _partition(tenant_id, _ENTITY_WEB_URL_LOCATOR),
            limit=2000,
            row_key_prefix=f"{workspace_id}:",
        )
        items: list[WebUrlSourceLocator] = []
        for doc in result.documents:
            try:
                items.append(WebUrlSourceLocator.model_validate(dict(doc.data)))
            except Exception as exc:
                raise ValueError("web_url_locator_malformed") from exc
        return sorted(items, key=lambda item: item.created_at)

    def delete_web_url_locator(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        requested_url_fingerprint: str,
    ) -> None:
        self._store.delete(
            _partition(tenant_id, _ENTITY_WEB_URL_LOCATOR),
            f"{workspace_id}:{requested_url_fingerprint}",
        )

    def delete_web_url_locators_for_workspace(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> int:
        deleted = 0
        for locator in self.list_web_url_locators(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        ):
            self.delete_web_url_locator(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                requested_url_fingerprint=locator.requested_url_fingerprint,
            )
            deleted += 1
        return deleted

    # --- Intake batch ---

    def put_intake_batch(self, batch: IntakeBatch) -> IntakeBatch:
        self._put(
            _partition(batch.tenant_id, _ENTITY_INTAKE_BATCH),
            f"{batch.workspace_id}:{batch.batch_id}",
            batch,
        )
        return batch

    def get_intake_batch(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        batch_id: str,
    ) -> IntakeBatch | None:
        return self._get(
            _partition(tenant_id, _ENTITY_INTAKE_BATCH),
            f"{workspace_id}:{batch_id}",
            IntakeBatch,
        )

    def list_intake_batches(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> list[IntakeBatch]:
        result = self._store.query(
            _partition(tenant_id, _ENTITY_INTAKE_BATCH),
            limit=500,
            row_key_prefix=f"{workspace_id}:",
        )
        items = [IntakeBatch.model_validate(dict(doc.data)) for doc in result.documents]
        return sorted(items, key=lambda item: item.created_at)

    def delete_intake_batches_for_workspace(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> int:
        deleted = 0
        for batch in self.list_intake_batches(tenant_id=tenant_id, workspace_id=workspace_id):
            self._store.delete(
                _partition(tenant_id, _ENTITY_INTAKE_BATCH),
                f"{workspace_id}:{batch.batch_id}",
            )
            deleted += 1
        return deleted

    # --- Active knowledge-ingestion locators ---

    def put_active_ingestion_locator(
        self,
        locator: ActiveKnowledgeIngestionLocator,
    ) -> ActiveKnowledgeIngestionLocator:
        self._put(
            _ACTIVE_KNOWLEDGE_INGESTION_PARTITION,
            locator.operation_id,
            locator,
        )
        return locator

    def scan_active_ingestion_locators(
        self,
        *,
        limit: int = 5000,
    ) -> ActiveKnowledgeIngestionLocatorScan:
        result = self._store.query(_ACTIVE_KNOWLEDGE_INGESTION_PARTITION, limit=limit)
        locators: list[ActiveKnowledgeIngestionLocator] = []
        malformed_seen = 0
        malformed_removed = 0
        for doc in result.documents:
            try:
                locators.append(
                    ActiveKnowledgeIngestionLocator.model_validate(dict(doc.data))
                )
            except Exception:  # noqa: BLE001 - isolate malformed locator rows
                malformed_seen += 1
                try:
                    self._store.delete(_ACTIVE_KNOWLEDGE_INGESTION_PARTITION, doc.row_key)
                    malformed_removed += 1
                except Exception:  # noqa: BLE001 - continue scan after delete failure
                    pass
        locators.sort(key=lambda item: (item.created_at, item.operation_id))
        return ActiveKnowledgeIngestionLocatorScan(
            locators=tuple(locators),
            malformed_seen=malformed_seen,
            malformed_removed=malformed_removed,
        )

    def list_active_ingestion_locators(
        self,
        *,
        limit: int = 5000,
    ) -> list[ActiveKnowledgeIngestionLocator]:
        return list(self.scan_active_ingestion_locators(limit=limit).locators)

    def delete_active_ingestion_locator(self, operation_id: str) -> None:
        self._store.delete(_ACTIVE_KNOWLEDGE_INGESTION_PARTITION, operation_id)

    # --- Workspace Knowledge Configuration ---

    def _require_conditional_store(self) -> ConditionalDocumentStore:
        if not isinstance(self._store, ConditionalDocumentStore):
            raise WorkspaceKnowledgeConfigurationRepositoryError(
                "configuration_conditional_store_required"
            )
        return self._store

    def _put_if_absent(self, model: BaseModel, *, partition_key: str, row_key: str) -> bool:
        store = self._require_conditional_store()
        return store.put_if_absent(
            _to_document_record(model, partition_key=partition_key, row_key=row_key)
        )

    def _replace_if_match(
        self,
        *,
        expected: BaseModel,
        replacement: BaseModel,
        partition_key: str,
        row_key: str,
    ) -> bool:
        store = self._require_conditional_store()
        return store.replace_if_match(
            expected=_to_document_record(expected, partition_key=partition_key, row_key=row_key),
            replacement=_to_document_record(
                replacement,
                partition_key=partition_key,
                row_key=row_key,
            ),
        )

    def _delete_if_match(self, model: BaseModel, *, partition_key: str, row_key: str) -> bool:
        store = self._require_conditional_store()
        return store.delete_if_match(
            expected=_to_document_record(model, partition_key=partition_key, row_key=row_key)
        )

    def _list_revision_versions(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        partition_key: str,
        model_type: type[T],
        entity_id_field: str | None = None,
        fixed_entity_id: str | None = None,
        sort_key: Callable[[T], tuple[object, ...]],
    ) -> list[T]:
        result = self._store.query(
            partition_key,
            limit=_KNOWLEDGE_CONFIGURATION_REVISION_SCAN_LIMIT + 1,
            row_key_prefix=f"{workspace_id}:",
        )
        if len(result.documents) > _KNOWLEDGE_CONFIGURATION_REVISION_SCAN_LIMIT:
            raise WorkspaceKnowledgeConfigurationRepositoryError(
                "knowledge_configuration_revision_scan_limit_exceeded"
            )
        items: list[T] = []
        for doc in result.documents:
            model = model_type.model_validate(dict(doc.data))
            row_workspace_id, row_entity_id, row_revision = _parse_revision_row_key(doc.row_key)
            if row_workspace_id != workspace_id:
                raise ValueError("knowledge_configuration_record_identity_mismatch")
            if fixed_entity_id is not None:
                model_entity_id = fixed_entity_id
            else:
                assert entity_id_field is not None
                model_entity_id = getattr(model, entity_id_field)
            _assert_record_identity(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                model_tenant_id=model.tenant_id,
                model_workspace_id=model.workspace_id,
                entity_id=row_entity_id,
                model_entity_id=model_entity_id,
                revision=row_revision,
                model_revision=model.effective_revision,
            )
            items.append(model)
        return sorted(items, key=sort_key)

    def get_knowledge_configuration_head(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> WorkspaceKnowledgeConfigurationHead | None:
        record = self._store.get(
            _partition(tenant_id, _ENTITY_KNOWLEDGE_CONFIGURATION_HEAD),
            workspace_id,
        )
        if record is None:
            return None
        head = WorkspaceKnowledgeConfigurationHead.model_validate(dict(record.data))
        _assert_record_identity(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            model_tenant_id=head.tenant_id,
            model_workspace_id=head.workspace_id,
        )
        return head

    def put_knowledge_configuration_head_if_absent(
        self,
        head: WorkspaceKnowledgeConfigurationHead,
    ) -> bool:
        partition_key = _partition(head.tenant_id, _ENTITY_KNOWLEDGE_CONFIGURATION_HEAD)
        return self._put_if_absent(head, partition_key=partition_key, row_key=head.workspace_id)

    def replace_knowledge_configuration_head_if_match(
        self,
        *,
        expected: WorkspaceKnowledgeConfigurationHead,
        replacement: WorkspaceKnowledgeConfigurationHead,
    ) -> bool:
        if (
            expected.tenant_id != replacement.tenant_id
            or expected.workspace_id != replacement.workspace_id
        ):
            raise ValueError("knowledge_configuration_conditional_key_mismatch")
        partition_key = _partition(expected.tenant_id, _ENTITY_KNOWLEDGE_CONFIGURATION_HEAD)
        row_key = expected.workspace_id
        return self._replace_if_match(
            expected=expected,
            replacement=replacement,
            partition_key=partition_key,
            row_key=row_key,
        )

    def get_knowledge_configuration_mutation(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        operation: WorkspaceKnowledgeMutationOperationV1,
        idempotency_key_hash: str,
    ) -> WorkspaceKnowledgeMutationRecord | None:
        _validate_idempotency_key_hash(idempotency_key_hash)
        row_key = _mutation_row_key(
            workspace_id=workspace_id,
            operation=operation.value,
            idempotency_key_hash=idempotency_key_hash,
        )
        record = self._store.get(
            _partition(tenant_id, _ENTITY_KNOWLEDGE_CONFIGURATION_MUTATION),
            row_key,
        )
        if record is None:
            return None
        mutation = WorkspaceKnowledgeMutationRecord.model_validate(dict(record.data))
        _assert_record_identity(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            model_tenant_id=mutation.tenant_id,
            model_workspace_id=mutation.workspace_id,
            entity_id=operation.value,
            model_entity_id=mutation.operation.value,
            revision=None,
            model_revision=None,
        )
        if mutation.idempotency_key_hash != idempotency_key_hash:
            raise ValueError("knowledge_configuration_record_identity_mismatch")
        return mutation

    def put_knowledge_configuration_mutation_if_absent(
        self,
        mutation: WorkspaceKnowledgeMutationRecord,
    ) -> bool:
        partition_key = _partition(mutation.tenant_id, _ENTITY_KNOWLEDGE_CONFIGURATION_MUTATION)
        row_key = _mutation_row_key(
            workspace_id=mutation.workspace_id,
            operation=mutation.operation.value,
            idempotency_key_hash=mutation.idempotency_key_hash,
        )
        return self._put_if_absent(mutation, partition_key=partition_key, row_key=row_key)

    def replace_knowledge_configuration_mutation_if_match(
        self,
        *,
        expected: WorkspaceKnowledgeMutationRecord,
        replacement: WorkspaceKnowledgeMutationRecord,
    ) -> bool:
        if (
            expected.tenant_id != replacement.tenant_id
            or expected.workspace_id != replacement.workspace_id
            or expected.operation != replacement.operation
            or expected.idempotency_key_hash != replacement.idempotency_key_hash
        ):
            raise ValueError("knowledge_configuration_conditional_key_mismatch")
        partition_key = _partition(expected.tenant_id, _ENTITY_KNOWLEDGE_CONFIGURATION_MUTATION)
        row_key = _mutation_row_key(
            workspace_id=expected.workspace_id,
            operation=expected.operation.value,
            idempotency_key_hash=expected.idempotency_key_hash,
        )
        return self._replace_if_match(
            expected=expected,
            replacement=replacement,
            partition_key=partition_key,
            row_key=row_key,
        )

    def list_knowledge_configuration_mutations(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> list[WorkspaceKnowledgeMutationRecord]:
        partition_key = _partition(tenant_id, _ENTITY_KNOWLEDGE_CONFIGURATION_MUTATION)
        result = self._store.query(
            partition_key,
            limit=_KNOWLEDGE_CONFIGURATION_MUTATION_SCAN_PROBE_LIMIT,
            row_key_prefix=f"{workspace_id}:",
        )
        if len(result.documents) > _KNOWLEDGE_CONFIGURATION_MUTATION_SCAN_LIMIT:
            raise WorkspaceKnowledgeConfigurationRepositoryError(
                "knowledge_configuration_mutation_scan_limit_exceeded"
            )
        items: list[WorkspaceKnowledgeMutationRecord] = []
        for doc in result.documents:
            row_workspace_id, row_operation, row_hash = _parse_mutation_row_key(doc.row_key)
            if row_workspace_id != workspace_id:
                raise ValueError("knowledge_configuration_record_identity_mismatch")
            mutation = WorkspaceKnowledgeMutationRecord.model_validate(dict(doc.data))
            _assert_record_identity(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                model_tenant_id=mutation.tenant_id,
                model_workspace_id=mutation.workspace_id,
                entity_id=row_operation,
                model_entity_id=mutation.operation.value,
            )
            if mutation.idempotency_key_hash != row_hash:
                raise ValueError("knowledge_configuration_record_identity_mismatch")
            items.append(mutation)
        return sorted(
            items,
            key=lambda item: (
                item.created_at,
                item.operation.value,
                item.idempotency_key_hash,
                item.mutation_id,
            ),
        )

    def find_knowledge_configuration_mutation_by_id(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        mutation_id: str,
    ) -> WorkspaceKnowledgeMutationRecord | None:
        matches = [
            mutation
            for mutation in self.list_knowledge_configuration_mutations(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
            )
            if mutation.mutation_id == mutation_id
        ]
        if not matches:
            return None
        if len(matches) > 1:
            raise WorkspaceKnowledgeConfigurationRepositoryError(
                "knowledge_configuration_mutation_id_not_unique"
            )
        return matches[0]

    def put_knowledge_connection_attachment_version_if_absent(
        self,
        attachment: WorkspaceConnectionAttachment,
    ) -> bool:
        partition_key = _partition(
            attachment.tenant_id,
            _ENTITY_KNOWLEDGE_CONFIGURATION_CONNECTION_ATTACHMENT,
        )
        row_key = _revision_row_key(
            workspace_id=attachment.workspace_id,
            entity_id=attachment.attachment_id,
            revision=attachment.effective_revision,
        )
        return self._put_if_absent(attachment, partition_key=partition_key, row_key=row_key)

    def get_knowledge_connection_attachment_version(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        attachment_id: str,
        effective_revision: int,
    ) -> WorkspaceConnectionAttachment | None:
        partition_key = _partition(
            tenant_id,
            _ENTITY_KNOWLEDGE_CONFIGURATION_CONNECTION_ATTACHMENT,
        )
        row_key = _revision_row_key(
            workspace_id=workspace_id,
            entity_id=attachment_id,
            revision=effective_revision,
        )
        record = self._store.get(partition_key, row_key)
        if record is None:
            return None
        attachment = WorkspaceConnectionAttachment.model_validate(dict(record.data))
        _assert_record_identity(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            model_tenant_id=attachment.tenant_id,
            model_workspace_id=attachment.workspace_id,
            entity_id=attachment_id,
            model_entity_id=attachment.attachment_id,
            revision=effective_revision,
            model_revision=attachment.effective_revision,
        )
        return attachment

    def list_knowledge_connection_attachment_versions(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> list[WorkspaceConnectionAttachment]:
        return self._list_revision_versions(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            partition_key=_partition(
                tenant_id,
                _ENTITY_KNOWLEDGE_CONFIGURATION_CONNECTION_ATTACHMENT,
            ),
            model_type=WorkspaceConnectionAttachment,
            entity_id_field="attachment_id",
            sort_key=lambda item: (item.attachment_id, item.effective_revision),
        )

    def delete_knowledge_connection_attachment_version_if_match(
        self,
        attachment: WorkspaceConnectionAttachment,
    ) -> bool:
        partition_key = _partition(
            attachment.tenant_id,
            _ENTITY_KNOWLEDGE_CONFIGURATION_CONNECTION_ATTACHMENT,
        )
        row_key = _revision_row_key(
            workspace_id=attachment.workspace_id,
            entity_id=attachment.attachment_id,
            revision=attachment.effective_revision,
        )
        return self._delete_if_match(attachment, partition_key=partition_key, row_key=row_key)

    def put_knowledge_indexed_source_version_if_absent(
        self,
        binding: WorkspaceIndexedSourceBinding,
    ) -> bool:
        partition_key = _partition(
            binding.tenant_id,
            _ENTITY_KNOWLEDGE_CONFIGURATION_INDEXED_SOURCE,
        )
        row_key = _revision_row_key(
            workspace_id=binding.workspace_id,
            entity_id=binding.indexed_source_binding_id,
            revision=binding.effective_revision,
        )
        return self._put_if_absent(binding, partition_key=partition_key, row_key=row_key)

    def get_knowledge_indexed_source_version(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        indexed_source_binding_id: str,
        effective_revision: int,
    ) -> WorkspaceIndexedSourceBinding | None:
        partition_key = _partition(
            tenant_id,
            _ENTITY_KNOWLEDGE_CONFIGURATION_INDEXED_SOURCE,
        )
        row_key = _revision_row_key(
            workspace_id=workspace_id,
            entity_id=indexed_source_binding_id,
            revision=effective_revision,
        )
        record = self._store.get(partition_key, row_key)
        if record is None:
            return None
        binding = WorkspaceIndexedSourceBinding.model_validate(dict(record.data))
        _assert_record_identity(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            model_tenant_id=binding.tenant_id,
            model_workspace_id=binding.workspace_id,
            entity_id=indexed_source_binding_id,
            model_entity_id=binding.indexed_source_binding_id,
            revision=effective_revision,
            model_revision=binding.effective_revision,
        )
        return binding

    def list_knowledge_indexed_source_versions(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> list[WorkspaceIndexedSourceBinding]:
        return self._list_revision_versions(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            partition_key=_partition(
                tenant_id,
                _ENTITY_KNOWLEDGE_CONFIGURATION_INDEXED_SOURCE,
            ),
            model_type=WorkspaceIndexedSourceBinding,
            entity_id_field="indexed_source_binding_id",
            sort_key=lambda item: (item.indexed_source_binding_id, item.effective_revision),
        )

    def delete_knowledge_indexed_source_version_if_match(
        self,
        binding: WorkspaceIndexedSourceBinding,
    ) -> bool:
        partition_key = _partition(
            binding.tenant_id,
            _ENTITY_KNOWLEDGE_CONFIGURATION_INDEXED_SOURCE,
        )
        row_key = _revision_row_key(
            workspace_id=binding.workspace_id,
            entity_id=binding.indexed_source_binding_id,
            revision=binding.effective_revision,
        )
        return self._delete_if_match(binding, partition_key=partition_key, row_key=row_key)

    def put_knowledge_live_access_version_if_absent(
        self,
        binding: WorkspaceLiveAccessBinding,
    ) -> bool:
        partition_key = _partition(
            binding.tenant_id,
            _ENTITY_KNOWLEDGE_CONFIGURATION_LIVE_ACCESS,
        )
        row_key = _revision_row_key(
            workspace_id=binding.workspace_id,
            entity_id=binding.live_access_binding_id,
            revision=binding.effective_revision,
        )
        return self._put_if_absent(binding, partition_key=partition_key, row_key=row_key)

    def get_knowledge_live_access_version(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        live_access_binding_id: str,
        effective_revision: int,
    ) -> WorkspaceLiveAccessBinding | None:
        partition_key = _partition(
            tenant_id,
            _ENTITY_KNOWLEDGE_CONFIGURATION_LIVE_ACCESS,
        )
        row_key = _revision_row_key(
            workspace_id=workspace_id,
            entity_id=live_access_binding_id,
            revision=effective_revision,
        )
        record = self._store.get(partition_key, row_key)
        if record is None:
            return None
        binding = WorkspaceLiveAccessBinding.model_validate(dict(record.data))
        _assert_record_identity(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            model_tenant_id=binding.tenant_id,
            model_workspace_id=binding.workspace_id,
            entity_id=live_access_binding_id,
            model_entity_id=binding.live_access_binding_id,
            revision=effective_revision,
            model_revision=binding.effective_revision,
        )
        return binding

    def list_knowledge_live_access_versions(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> list[WorkspaceLiveAccessBinding]:
        return self._list_revision_versions(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            partition_key=_partition(
                tenant_id,
                _ENTITY_KNOWLEDGE_CONFIGURATION_LIVE_ACCESS,
            ),
            model_type=WorkspaceLiveAccessBinding,
            entity_id_field="live_access_binding_id",
            sort_key=lambda item: (item.live_access_binding_id, item.effective_revision),
        )

    def delete_knowledge_live_access_version_if_match(
        self,
        binding: WorkspaceLiveAccessBinding,
    ) -> bool:
        partition_key = _partition(
            binding.tenant_id,
            _ENTITY_KNOWLEDGE_CONFIGURATION_LIVE_ACCESS,
        )
        row_key = _revision_row_key(
            workspace_id=binding.workspace_id,
            entity_id=binding.live_access_binding_id,
            revision=binding.effective_revision,
        )
        return self._delete_if_match(binding, partition_key=partition_key, row_key=row_key)

    def put_knowledge_query_policy_version_if_absent(
        self,
        policy: WorkspaceCommittedQueryPolicy,
    ) -> bool:
        partition_key = _partition(
            policy.tenant_id,
            _ENTITY_KNOWLEDGE_CONFIGURATION_QUERY_POLICY,
        )
        row_key = _revision_row_key(
            workspace_id=policy.workspace_id,
            entity_id=_QUERY_POLICY_ENTITY_ID,
            revision=policy.effective_revision,
        )
        return self._put_if_absent(policy, partition_key=partition_key, row_key=row_key)

    def get_knowledge_query_policy_version(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        effective_revision: int,
    ) -> WorkspaceCommittedQueryPolicy | None:
        partition_key = _partition(
            tenant_id,
            _ENTITY_KNOWLEDGE_CONFIGURATION_QUERY_POLICY,
        )
        row_key = _revision_row_key(
            workspace_id=workspace_id,
            entity_id=_QUERY_POLICY_ENTITY_ID,
            revision=effective_revision,
        )
        record = self._store.get(partition_key, row_key)
        if record is None:
            return None
        try:
            policy = parse_workspace_query_policy(dict(record.data))
        except ValueError as exc:
            raise WorkspaceKnowledgeConfigurationRepositoryError(
                "query_policy_schema_version_unknown"
            ) from exc
        _assert_record_identity(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            model_tenant_id=policy.tenant_id,
            model_workspace_id=policy.workspace_id,
            revision=effective_revision,
            model_revision=policy.effective_revision,
        )
        return policy

    def list_knowledge_query_policy_versions(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> list[WorkspaceCommittedQueryPolicy]:
        partition_key = _partition(
            tenant_id,
            _ENTITY_KNOWLEDGE_CONFIGURATION_QUERY_POLICY,
        )
        result = self._store.query(
            partition_key,
            limit=_KNOWLEDGE_CONFIGURATION_REVISION_SCAN_LIMIT + 1,
            row_key_prefix=f"{workspace_id}:",
        )
        if len(result.documents) > _KNOWLEDGE_CONFIGURATION_REVISION_SCAN_LIMIT:
            raise WorkspaceKnowledgeConfigurationRepositoryError(
                "knowledge_configuration_revision_scan_limit_exceeded"
            )
        items: list[WorkspaceCommittedQueryPolicy] = []
        for doc in result.documents:
            try:
                policy = parse_workspace_query_policy(dict(doc.data))
            except ValueError as exc:
                raise WorkspaceKnowledgeConfigurationRepositoryError(
                    "query_policy_schema_version_unknown"
                ) from exc
            row_workspace_id, row_entity_id, row_revision = _parse_revision_row_key(doc.row_key)
            if row_workspace_id != workspace_id:
                raise ValueError("knowledge_configuration_record_identity_mismatch")
            if row_entity_id != _QUERY_POLICY_ENTITY_ID:
                raise ValueError("knowledge_configuration_record_identity_mismatch")
            _assert_record_identity(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                model_tenant_id=policy.tenant_id,
                model_workspace_id=policy.workspace_id,
                entity_id=row_entity_id,
                model_entity_id=_QUERY_POLICY_ENTITY_ID,
                revision=row_revision,
                model_revision=policy.effective_revision,
            )
            items.append(policy)
        return sorted(items, key=lambda item: (item.effective_revision,))

    def delete_knowledge_query_policy_version_if_match(
        self,
        policy: WorkspaceCommittedQueryPolicy,
    ) -> bool:
        partition_key = _partition(
            policy.tenant_id,
            _ENTITY_KNOWLEDGE_CONFIGURATION_QUERY_POLICY,
        )
        row_key = _revision_row_key(
            workspace_id=policy.workspace_id,
            entity_id=_QUERY_POLICY_ENTITY_ID,
            revision=policy.effective_revision,
        )
        return self._delete_if_match(policy, partition_key=partition_key, row_key=row_key)
