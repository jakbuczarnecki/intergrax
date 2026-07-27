# © Artur Czarnecki. All rights reserved.

"""Shared single-file document indexing orchestration for managed workspaces."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from intergrax.runtime.task.task import Task, TaskContext
from intergrax.runtime.task.task_run_bridge import new_run_id
from local_workspace_application.workspaces.idempotency import logical_document_id
from local_workspace_application.workspaces.models import WorkspaceDocumentReference
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository


class TaskExecutorPort(Protocol):
    async def execute(self, task: Task) -> Any: ...


def _utc_now() -> datetime:
    return datetime.now(UTC)


def extract_ingest_summary(result: Any) -> dict[str, Any]:
    """Normalize indexer / task-executor result shapes into an ingest summary."""
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


class WorkspaceDocumentIndexingResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    indexed: bool
    unchanged: bool
    document_id: str = ""
    documents_indexed: int = Field(default=0, ge=0)
    num_chunks: int = Field(default=0, ge=0)
    reason: str = ""


class WorkspaceDocumentIndexingError(RuntimeError):
    def __init__(self, error_code: str) -> None:
        code = (error_code or "").strip()
        if not code:
            raise ValueError("error_code_required")
        self.error_code = code
        super().__init__(code)


class WorkspaceDocumentIndexingService:
    """Owns document-ref lookup, unchanged detection and local.workspace.index invocation."""

    def __init__(
        self,
        repository: ManagedWorkspaceRepository,
        task_executor: TaskExecutorPort,
    ) -> None:
        self._repository = repository
        self._task_executor = task_executor

    async def index_one(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        source_id: str,
        operation_id: str,
        physical_path: Path,
        logical_source_path: str,
        safe_file_name: str,
        content_hash: str,
    ) -> WorkspaceDocumentIndexingResult:
        existing = self._repository.get_document_ref_by_path(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            source_id=source_id,
            source_path=logical_source_path,
        )
        if existing is not None and existing.content_hash == content_hash:
            return WorkspaceDocumentIndexingResult(
                indexed=False,
                unchanged=True,
                document_id=existing.document_id,
                documents_indexed=0,
                num_chunks=0,
                reason="unchanged",
            )

        document_id = logical_document_id(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            source_id=source_id,
            normalized_source_path=logical_source_path,
            content_hash=content_hash,
        )
        run_id = new_run_id()
        task = Task(
            task_id=run_id,
            tenant_id=tenant_id,
            user_id="lkw.managed_workspace",
            message=f"Index managed workspace source file {safe_file_name}",
            context=TaskContext(capability="local.workspace.index"),
            metadata={
                "tenant_id": tenant_id,
                "workspace_id": workspace_id,
                "source_id": source_id,
                "collection_id": workspace_id,
                "document_id": document_id,
                "source_paths": [str(physical_path)],
                "logical_source_path": logical_source_path,
                "display_file_name": safe_file_name,
                "content_hash": content_hash,
                "operation_id": operation_id,
                "requested_by": "lkw.managed_workspace.index",
            },
        )
        result = await self._task_executor.execute(task)
        ingest_summary = extract_ingest_summary(result)
        if not ingest_summary.get("used"):
            reason = str(ingest_summary.get("reason") or "ingest_failed")
            raise WorkspaceDocumentIndexingError(reason)

        self._repository.put_document_ref(
            WorkspaceDocumentReference(
                document_id=document_id,
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                source_id=source_id,
                source_path=logical_source_path,
                file_name=safe_file_name,
                content_hash=content_hash,
                indexed_at=_utc_now(),
            )
        )
        return WorkspaceDocumentIndexingResult(
            indexed=True,
            unchanged=False,
            document_id=document_id,
            documents_indexed=1,
            num_chunks=int(ingest_summary.get("num_chunks") or 0),
            reason=str(ingest_summary.get("reason") or "ingest_complete"),
        )
