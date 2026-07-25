# © Artur Czarnecki. All rights reserved.

"""Source synchronization orchestration for managed workspaces (LKW-PRODUCT-1)."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from intergrax.runtime.task.task import Task, TaskContext
from intergrax.runtime.task.task_run_bridge import new_run_id
from intergrax.tools.providers.filesystem.allowlist import read_allowlist_roots_from_env
from local_workspace_application.workspaces.discovery import discover_source_files
from local_workspace_application.workspaces.idempotency import (
    content_hash_for_file,
    logical_document_id,
    normalize_source_path,
)
from local_workspace_application.workspaces.models import (
    WorkspaceDocumentReference,
    WorkspaceOperation,
    WorkspaceOperationStatus,
    WorkspaceSourceStatus,
    WorkspaceSourceType,
)
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository

logger = logging.getLogger(__name__)


class TaskExecutorPort(Protocol):
    async def execute(self, task: Task) -> Any: ...


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
    ) -> None:
        self._repository = repository
        self._task_executor = task_executor
        self._allowlist_roots = allowlist_roots

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
                existing = self._repository.get_document_ref_by_path(
                    tenant_id=tenant_id,
                    workspace_id=operation.workspace_id,
                    source_id=operation.source_id,
                    source_path=normalized,
                )
                if existing is not None and existing.content_hash == digest:
                    documents_unchanged += 1
                    continue

                document_id = logical_document_id(
                    tenant_id=tenant_id,
                    workspace_id=operation.workspace_id,
                    source_id=operation.source_id,
                    normalized_source_path=normalized,
                    content_hash=digest,
                )
                run_id = new_run_id()
                task = Task(
                    task_id=run_id,
                    tenant_id=tenant_id,
                    user_id="lkw.managed_workspace",
                    message=f"Index managed workspace source file {path.name}",
                    context=TaskContext(capability="local.workspace.index"),
                    metadata={
                        "tenant_id": tenant_id,
                        "workspace_id": operation.workspace_id,
                        "source_id": operation.source_id,
                        "collection_id": operation.workspace_id,
                        "document_id": document_id,
                        "source_paths": [str(path)],
                        "content_hash": digest,
                        "operation_id": operation.operation_id,
                        "requested_by": "lkw.managed_workspace.sync",
                    },
                )
                result = await self._task_executor.execute(task)
                ingest_summary = _extract_ingest_summary(result)
                if not ingest_summary.get("used"):
                    files_failed += 1
                    reason = str(ingest_summary.get("reason") or "ingest_failed")
                    logger.warning(
                        "managed_workspace_sync_file_failed operation_id=%s path=%s reason=%s",
                        operation.operation_id,
                        normalized,
                        reason,
                    )
                    continue

                self._repository.put_document_ref(
                    WorkspaceDocumentReference(
                        document_id=document_id,
                        tenant_id=tenant_id,
                        workspace_id=operation.workspace_id,
                        source_id=operation.source_id,
                        source_path=normalized,
                        file_name=path.name,
                        content_hash=digest,
                        indexed_at=_utc_now(),
                    )
                )
                documents_indexed += 1

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


def _extract_ingest_summary(result: Any) -> dict[str, Any]:
    metadata = getattr(result, "metadata", None)
    if isinstance(metadata, dict):
        summary = metadata.get("ingest_summary")
        if isinstance(summary, dict):
            return summary
        for key in ("domain_summary", "result"):
            nested = metadata.get(key)
            if isinstance(nested, dict) and isinstance(nested.get("ingest_summary"), dict):
                return nested["ingest_summary"]
        evidence = metadata.get("lkw_evidence.v1")
        if isinstance(evidence, dict):
            diagnostics = evidence.get("diagnostics")
            if isinstance(diagnostics, dict):
                index_diag = diagnostics.get("lkw.index_summary.v1")
                if isinstance(index_diag, dict):
                    ingested_count = int(index_diag.get("ingested_count") or 0)
                    return {
                        "used": ingested_count > 0,
                        "reason": "ingest_complete" if ingested_count > 0 else "ingest_failed",
                    }

    execution = getattr(result, "execution_result", None)
    if execution is not None:
        structured = getattr(execution, "structured_data", None)
        if isinstance(structured, dict):
            summary = structured.get("ingest_summary")
            if isinstance(summary, dict):
                return summary

    answer = str(getattr(result, "answer", "") or "")
    if "index failed" in answer:
        return {"used": False, "reason": "index_failed"}
    if "ingested=" in answer:
        # Parse ``ingested=N`` from the indexer answer line.
        try:
            token = next(part for part in answer.split(",") if "ingested=" in part)
            count = int(token.split("ingested=", 1)[1].strip())
            return {
                "used": count > 0,
                "reason": "ingest_complete" if count > 0 else "no_paths_ingested",
            }
        except (StopIteration, ValueError):
            pass
    if "ingest_complete" in answer:
        return {"used": True, "reason": "ingest_complete"}
    return {"used": False, "reason": "ingest_summary_missing"}
