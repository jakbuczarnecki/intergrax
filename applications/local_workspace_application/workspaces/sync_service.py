# © Artur Czarnecki. All rights reserved.

"""Source synchronization orchestration for managed workspaces (LKW-PRODUCT-1)."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from intergrax.tools.providers.filesystem.allowlist import read_allowlist_roots_from_env
from local_workspace_application.workspaces.discovery import discover_source_files
from local_workspace_application.workspaces.document_indexing import (
    WorkspaceDocumentIndexingError,
    WorkspaceDocumentIndexingService,
    extract_ingest_summary,
)
from local_workspace_application.workspaces.idempotency import (
    content_hash_for_file,
    normalize_source_path,
)
from local_workspace_application.workspaces.models import (
    WorkspaceOperation,
    WorkspaceOperationStatus,
    WorkspaceSourceStatus,
    WorkspaceSourceType,
)
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository

logger = logging.getLogger(__name__)


class TaskExecutorPort(Protocol):
    async def execute(self, task: Any) -> Any: ...


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _safe_error_message(exc: BaseException) -> str:
    message = str(exc).strip() or exc.__class__.__name__
    # Never return traceback-like multi-line dumps to API consumers.
    first_line = message.splitlines()[0].strip()
    return first_line[:500]


class ManagedWorkspaceSyncService:
    """Runs source sync via the existing LocalWorkspaceTaskExecutor / indexer path."""

    def __init__(
        self,
        repository: ManagedWorkspaceRepository,
        task_executor: TaskExecutorPort,
        *,
        allowlist_roots: frozenset[str] | None = None,
        indexing_service: WorkspaceDocumentIndexingService | None = None,
    ) -> None:
        self._repository = repository
        self._task_executor = task_executor
        self._allowlist_roots = allowlist_roots
        self._indexing_service = indexing_service or WorkspaceDocumentIndexingService(
            repository,
            task_executor,
        )

    async def run_operation(self, *, tenant_id: str, operation_id: str) -> WorkspaceOperation:
        operation = self._repository.get_operation(tenant_id=tenant_id, operation_id=operation_id)
        if operation is None:
            raise LookupError("operation_not_found")

        # Duplicate delivery after terminal state: safe no-op.
        if operation.status in {
            WorkspaceOperationStatus.COMPLETED,
            WorkspaceOperationStatus.FAILED,
        }:
            return operation

        # Duplicate delivery while already running: fail-closed, do not re-ingest.
        if operation.status is WorkspaceOperationStatus.RUNNING:
            logger.warning(
                "managed_workspace_sync_duplicate_while_running operation_id=%s",
                operation.operation_id,
            )
            return operation

        if operation.status is not WorkspaceOperationStatus.QUEUED:
            return self._fail(operation, f"unexpected_operation_status:{operation.status.value}")

        source = self._repository.get_source(
            tenant_id=tenant_id,
            workspace_id=operation.workspace_id,
            source_id=operation.source_id,
        )
        if source is None:
            return self._fail(operation, "source_not_found")
        if source.source_type is not WorkspaceSourceType.LOCAL_FOLDER:
            return self._fail(operation, "source_sync_unsupported_for_source_type")

        started = _utc_now()
        operation = operation.model_copy(
            update={
                "status": WorkspaceOperationStatus.RUNNING,
                "started_at": started,
                "error": None,
            }
        )
        self._repository.put_operation(operation)
        self._repository.put_source(
            source.model_copy(update={"status": WorkspaceSourceStatus.SYNCING})
        )

        roots = self._allowlist_roots
        if roots is None:
            roots = read_allowlist_roots_from_env()

        try:
            root = Path(source.path)
            discovered, skipped = discover_source_files(
                root,
                recursive=source.recursive,
                allowlist_roots=roots,
            )
            files_discovered = len(discovered) + len(skipped)
            files_processed = 0
            files_failed = len(skipped)
            documents_indexed = 0
            documents_unchanged = 0

            for path in discovered:
                files_processed += 1
                normalized = normalize_source_path(path)
                digest = content_hash_for_file(path)
                try:
                    index_result = await self._indexing_service.index_one(
                        tenant_id=tenant_id,
                        workspace_id=operation.workspace_id,
                        source_id=operation.source_id,
                        operation_id=operation.operation_id,
                        physical_path=path,
                        logical_source_path=normalized,
                        safe_file_name=path.name,
                        content_hash=digest,
                    )
                except WorkspaceDocumentIndexingError as exc:
                    files_failed += 1
                    logger.warning(
                        "managed_workspace_sync_file_failed operation_id=%s path=%s reason=%s",
                        operation.operation_id,
                        normalized,
                        exc.error_code,
                    )
                    continue

                if index_result.unchanged:
                    documents_unchanged += 1
                    continue
                if index_result.indexed:
                    documents_indexed += 1
                else:
                    files_failed += 1

            completed = _utc_now()
            final_status = (
                WorkspaceOperationStatus.FAILED
                if files_discovered > 0 and documents_indexed == 0 and documents_unchanged == 0
                else WorkspaceOperationStatus.COMPLETED
            )
            error = None
            if final_status == WorkspaceOperationStatus.FAILED:
                error = "sync_produced_no_documents"
            operation = operation.model_copy(
                update={
                    "status": final_status,
                    "files_discovered": files_discovered,
                    "files_processed": files_processed,
                    "files_failed": files_failed,
                    "documents_indexed": documents_indexed,
                    "documents_unchanged": documents_unchanged,
                    "completed_at": completed,
                    "error": error,
                }
            )
            self._repository.put_operation(operation)
            self._repository.put_source(
                source.model_copy(
                    update={
                        "status": (
                            WorkspaceSourceStatus.ERROR
                            if final_status == WorkspaceOperationStatus.FAILED
                            else WorkspaceSourceStatus.READY
                        ),
                        "last_sync_at": completed,
                    }
                )
            )
            return operation
        except Exception as exc:
            logger.exception(
                "managed_workspace_sync_failed operation_id=%s",
                operation.operation_id,
            )
            failed = self._fail(operation, _safe_error_message(exc))
            self._repository.put_source(
                source.model_copy(update={"status": WorkspaceSourceStatus.ERROR})
            )
            return failed

    def _fail(self, operation: WorkspaceOperation, error: str) -> WorkspaceOperation:
        failed = operation.model_copy(
            update={
                "status": WorkspaceOperationStatus.FAILED,
                "error": error,
                "started_at": operation.started_at or _utc_now(),
                "completed_at": _utc_now(),
            }
        )
        self._repository.put_operation(failed)
        return failed


# Backward-compatible alias for callers that imported the private helper.
_extract_ingest_summary = extract_ingest_summary
