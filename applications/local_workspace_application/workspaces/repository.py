# © Artur Czarnecki. All rights reserved.

"""DocumentStore-backed persistence for managed workspaces (LKW-PRODUCT-1)."""

from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

from intergrax.integrations.contracts.document_store import DocumentRecord, DocumentStore
from local_workspace_application.workspaces.models import (
    Workspace,
    WorkspaceDocumentReference,
    WorkspaceOperation,
    WorkspaceOperationStatus,
    WorkspaceSource,
)

T = TypeVar("T", bound=BaseModel)

_ENTITY_WORKSPACE = "workspace"
_ENTITY_SOURCE = "source"
_ENTITY_OPERATION = "operation"
_ENTITY_DOCUMENT = "document"


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
