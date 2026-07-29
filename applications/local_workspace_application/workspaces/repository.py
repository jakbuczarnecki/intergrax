# © Artur Czarnecki. All rights reserved.

"""DocumentStore-backed persistence for managed workspaces (LKW-PRODUCT-1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

from pydantic import BaseModel

from intergrax.integrations.contracts.document_store import DocumentRecord, DocumentStore
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

T = TypeVar("T", bound=BaseModel)


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
_ACTIVE_KNOWLEDGE_INGESTION_PARTITION = "lkw.managed_workspace:active_knowledge_ingestion"


def _partition(tenant_id: str, entity: str) -> str:
    return f"lkw.managed_workspace:{tenant_id}:{entity}"


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

    # --- Operation ---

    def put_operation(self, operation: WorkspaceOperation) -> WorkspaceOperation:
        self._put(
            _partition(operation.tenant_id, _ENTITY_OPERATION),
            operation.operation_id,
            operation,
        )
        return operation

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
