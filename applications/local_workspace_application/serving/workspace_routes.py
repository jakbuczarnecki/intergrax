# © Artur Czarnecki. All rights reserved.

"""Public managed workspace HTTP routes (LKW-PRODUCT-1)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, Header, HTTPException, Request, status

from intergrax.contracts.acp_metadata_keys import AcpStructuredDataKey
from intergrax.fastapi_core.context import get_request_context
from intergrax.runtime.task.task import Task, TaskContext
from intergrax.runtime.task.task_run_bridge import new_run_id
from intergrax.tools.providers.filesystem.allowlist import read_allowlist_roots_from_env
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings
from local_workspace_application.host.task_executor import LocalWorkspaceTaskExecutor
from local_workspace_application.serving.run_metadata import attach_lkw_evidence_metadata
from local_workspace_application.serving.workspace_schemas import (
    CreateWorkspaceRequestV1,
    OperationResponseV1,
    RegisterSourceRequestV1,
    SourceListResponseV1,
    SourceResponseV1,
    SyncOperationAcceptedV1,
    WorkspaceListResponseV1,
    WorkspaceResponseV1,
    WorkspaceSearchHitV1,
    WorkspaceSearchRequestV1,
    WorkspaceSearchResponseV1,
)
from local_workspace_application.workspaces.document_store_factory import (
    resolve_managed_workspace_document_store,
)
from local_workspace_application.workspaces.idempotency import normalize_source_path
from local_workspace_application.workspaces.models import (
    Workspace,
    WorkspaceOperation,
    WorkspaceSource,
)
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository
from local_workspace_application.workspaces.service import ManagedWorkspaceService
from local_workspace_application.workspaces.sync_service import ManagedWorkspaceSyncService


def _workspace_response(workspace: Workspace) -> WorkspaceResponseV1:
    return WorkspaceResponseV1(
        workspace_id=workspace.workspace_id,
        tenant_id=workspace.tenant_id,
        name=workspace.name,
        description=workspace.description,
        status=workspace.status.value,
        created_at=workspace.created_at,
        updated_at=workspace.updated_at,
    )


def _source_response(source: WorkspaceSource) -> SourceResponseV1:
    return SourceResponseV1(
        source_id=source.source_id,
        workspace_id=source.workspace_id,
        source_type=source.source_type.value,
        path=source.path,
        status=source.status.value,
        recursive=source.recursive,
        created_at=source.created_at,
        last_sync_at=source.last_sync_at,
    )


def _operation_response(operation: WorkspaceOperation) -> OperationResponseV1:
    return OperationResponseV1(
        operation_id=operation.operation_id,
        operation_type=operation.operation_type.value,
        status=operation.status.value,
        workspace_id=operation.workspace_id,
        source_id=operation.source_id,
        files_discovered=operation.files_discovered,
        files_processed=operation.files_processed,
        files_failed=operation.files_failed,
        documents_indexed=operation.documents_indexed,
        documents_unchanged=operation.documents_unchanged,
        started_at=operation.started_at,
        completed_at=operation.completed_at,
        error=operation.error,
    )


def resolve_tenant_id(
    request: Request,
    *,
    body_tenant_id: str | None = None,
    header_tenant_id: str | None = None,
    default_tenant_id: str = "default",
) -> str:
    """Prefer auth/request context; never accept cross-tenant body override when context exists."""
    try:
        context = get_request_context(request)
    except RuntimeError:
        context = None
    if context is not None and context.tenant_id:
        return str(context.tenant_id)
    if header_tenant_id and header_tenant_id.strip():
        return header_tenant_id.strip()
    if body_tenant_id and body_tenant_id.strip():
        return body_tenant_id.strip()
    return default_tenant_id


def mount_managed_workspace_routes(
    app: FastAPI,
    *,
    task_executor: LocalWorkspaceTaskExecutor,
    settings: LocalWorkspaceBackendSettings,
    prefix: str = "/v1/local_workspace",
    repository: ManagedWorkspaceRepository | None = None,
) -> ManagedWorkspaceService:
    allowlist = settings.allowed_read_roots or read_allowlist_roots_from_env()
    shadow_roots = (Path(settings.shadow_workspaces_dir),)
    if repository is None:
        repository = ManagedWorkspaceRepository(resolve_managed_workspace_document_store())
    service = ManagedWorkspaceService(
        repository,
        allowlist_roots=frozenset(allowlist) if allowlist else None,
        shadow_roots=shadow_roots,
    )
    sync_service = ManagedWorkspaceSyncService(
        repository,
        task_executor,
        allowlist_roots=frozenset(allowlist) if allowlist else None,
    )
    app.state.lkw_managed_workspace_service = service
    app.state.lkw_managed_workspace_repository = repository

    router = APIRouter(prefix=prefix, tags=["local_workspace_managed"])

    def _not_found() -> HTTPException:
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="not_found")

    @router.post(
        "/workspaces",
        response_model=WorkspaceResponseV1,
        status_code=status.HTTP_201_CREATED,
    )
    async def create_workspace(
        request: Request,
        body: CreateWorkspaceRequestV1,
        x_tenant_id: str | None = Header(default=None, alias="X-Tenant-Id"),
    ) -> WorkspaceResponseV1:
        tenant_id = resolve_tenant_id(
            request,
            body_tenant_id=body.tenant_id,
            header_tenant_id=x_tenant_id,
        )
        workspace = service.create_workspace(
            tenant_id=tenant_id,
            name=body.name,
            description=body.description,
        )
        return _workspace_response(workspace)

    @router.get("/workspaces", response_model=WorkspaceListResponseV1)
    async def list_workspaces(
        request: Request,
        x_tenant_id: str | None = Header(default=None, alias="X-Tenant-Id"),
    ) -> WorkspaceListResponseV1:
        tenant_id = resolve_tenant_id(request, header_tenant_id=x_tenant_id)
        items = service.list_workspaces(tenant_id=tenant_id)
        return WorkspaceListResponseV1(workspaces=[_workspace_response(item) for item in items])

    @router.get("/workspaces/{workspace_id}", response_model=WorkspaceResponseV1)
    async def get_workspace(
        request: Request,
        workspace_id: str,
        x_tenant_id: str | None = Header(default=None, alias="X-Tenant-Id"),
    ) -> WorkspaceResponseV1:
        tenant_id = resolve_tenant_id(request, header_tenant_id=x_tenant_id)
        workspace = service.get_workspace(tenant_id=tenant_id, workspace_id=workspace_id)
        if workspace is None:
            raise _not_found()
        return _workspace_response(workspace)

    @router.post(
        "/workspaces/{workspace_id}/sources",
        response_model=SourceResponseV1,
        status_code=status.HTTP_201_CREATED,
    )
    async def register_source(
        request: Request,
        workspace_id: str,
        body: RegisterSourceRequestV1,
        x_tenant_id: str | None = Header(default=None, alias="X-Tenant-Id"),
    ) -> SourceResponseV1:
        tenant_id = resolve_tenant_id(request, header_tenant_id=x_tenant_id)
        try:
            source = service.register_local_folder_source(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                path=body.path,
                recursive=body.recursive,
            )
        except LookupError:
            raise _not_found() from None
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        return _source_response(source)

    @router.get(
        "/workspaces/{workspace_id}/sources",
        response_model=SourceListResponseV1,
    )
    async def list_sources(
        request: Request,
        workspace_id: str,
        x_tenant_id: str | None = Header(default=None, alias="X-Tenant-Id"),
    ) -> SourceListResponseV1:
        tenant_id = resolve_tenant_id(request, header_tenant_id=x_tenant_id)
        sources = service.list_sources(tenant_id=tenant_id, workspace_id=workspace_id)
        if sources is None:
            raise _not_found()
        return SourceListResponseV1(sources=[_source_response(item) for item in sources])

    @router.post(
        "/workspaces/{workspace_id}/sources/{source_id}/sync",
        response_model=SyncOperationAcceptedV1,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def start_sync(
        request: Request,
        workspace_id: str,
        source_id: str,
        x_tenant_id: str | None = Header(default=None, alias="X-Tenant-Id"),
    ) -> SyncOperationAcceptedV1:
        tenant_id = resolve_tenant_id(request, header_tenant_id=x_tenant_id)
        try:
            operation = service.create_sync_operation(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                source_id=source_id,
            )
        except LookupError:
            raise _not_found() from None

        task = asyncio.create_task(
            sync_service.run_operation(
                tenant_id=tenant_id,
                operation_id=operation.operation_id,
            ),
            name=f"lkw-managed-workspace-sync-{operation.operation_id}",
        )

        def _log_sync_task_result(done: asyncio.Task[Any]) -> None:
            try:
                done.result()
            except Exception:
                # Failure is persisted on the operation record; avoid silent task warnings.
                pass

        task.add_done_callback(_log_sync_task_result)
        return SyncOperationAcceptedV1(
            operation_id=operation.operation_id,
            workspace_id=operation.workspace_id,
            source_id=operation.source_id,
            status=operation.status.value,
        )

    @router.get("/operations/{operation_id}", response_model=OperationResponseV1)
    async def get_operation(
        request: Request,
        operation_id: str,
        x_tenant_id: str | None = Header(default=None, alias="X-Tenant-Id"),
    ) -> OperationResponseV1:
        tenant_id = resolve_tenant_id(request, header_tenant_id=x_tenant_id)
        operation = service.get_operation(tenant_id=tenant_id, operation_id=operation_id)
        if operation is None:
            raise _not_found()
        return _operation_response(operation)

    @router.post(
        "/workspaces/{workspace_id}/search",
        response_model=WorkspaceSearchResponseV1,
    )
    async def search_workspace(
        request: Request,
        workspace_id: str,
        body: WorkspaceSearchRequestV1,
        x_tenant_id: str | None = Header(default=None, alias="X-Tenant-Id"),
    ) -> WorkspaceSearchResponseV1:
        tenant_id = resolve_tenant_id(request, header_tenant_id=x_tenant_id)
        workspace = service.get_workspace(tenant_id=tenant_id, workspace_id=workspace_id)
        if workspace is None:
            raise _not_found()

        run_id = new_run_id()
        task = Task(
            task_id=run_id,
            tenant_id=tenant_id,
            user_id="lkw.managed_workspace",
            message=body.query,
            context=TaskContext(capability="local.workspace.search"),
            metadata={
                "tenant_id": tenant_id,
                "workspace_id": workspace_id,
                "collection_id": workspace_id,
                "query": body.query,
                "top_k": max(body.limit, 10),
                "requested_by": "lkw.managed_workspace.search",
            },
        )
        try:
            result = await task_executor.execute(task)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"search_error: {exc.__class__.__name__}",
            ) from exc

        # Populate curated diagnostics on result metadata (same read model as /run).
        result_metadata = dict(getattr(result, "metadata", None) or {})
        attach_lkw_evidence_metadata(
            result_metadata,
            task_result=result,
            capability="local.workspace.search",
        )
        result = result.model_copy(update={"metadata": result_metadata})

        hits = _map_search_hits(
            repository=repository,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            task_result=result,
            limit=body.limit,
        )
        return WorkspaceSearchResponseV1(
            workspace_id=workspace_id,
            query=body.query,
            results=hits,
        )

    app.include_router(router)
    return service


def _map_search_hits(
    *,
    repository: ManagedWorkspaceRepository,
    tenant_id: str,
    workspace_id: str,
    task_result: Any,
    limit: int,
) -> list[WorkspaceSearchHitV1]:
    evidence = _extract_search_evidence(task_result)
    refs_by_path = {
        normalize_source_path(ref.source_path): ref
        for ref in repository.list_document_refs(tenant_id=tenant_id, workspace_id=workspace_id)
    }
    # Also index by basename for Windows path-shape mismatches in tool payloads.
    refs_by_name: dict[str, Any] = {}
    for ref in refs_by_path.values():
        refs_by_name.setdefault(ref.file_name, ref)

    hits: list[WorkspaceSearchHitV1] = []
    for item in evidence:
        if not isinstance(item, dict):
            continue
        raw_path = item.get("source_path")
        source_path = normalize_source_path(str(raw_path)) if raw_path else ""
        ref = refs_by_path.get(source_path) if source_path else None
        if ref is None and source_path:
            basename = source_path.replace("\\", "/").rsplit("/", 1)[-1]
            ref = refs_by_name.get(basename)
        document_id = str(item.get("document_id") or (ref.document_id if ref else "")).strip()
        source_id = str(item.get("source_id") or (ref.source_id if ref else "")).strip()
        file_name = str(
            item.get("file_name")
            or (ref.file_name if ref else "")
            or (source_path.replace("\\", "/").rsplit("/", 1)[-1] if source_path else "")
        ).strip()
        resolved_path = (ref.source_path if ref is not None else source_path).strip()
        if not document_id or not source_id or not resolved_path or not file_name:
            continue
        score_raw = item.get("score")
        score = float(score_raw) if isinstance(score_raw, (int, float)) else 0.0
        snippet = str(item.get("text") or item.get("snippet") or "").strip()
        metadata = item.get("metadata")
        hits.append(
            WorkspaceSearchHitV1(
                document_id=document_id,
                source_id=source_id,
                workspace_id=workspace_id,
                source_path=resolved_path,
                file_name=file_name,
                score=score,
                snippet=snippet,
                metadata=dict(metadata) if isinstance(metadata, dict) else {},
            )
        )
        if len(hits) >= limit:
            break

    if hits:
        return hits

    # Fallback when structured evidence text was redacted but source refs are present.
    for raw_path in _extract_source_refs(task_result):
        source_path = normalize_source_path(str(raw_path))
        ref = refs_by_path.get(source_path)
        if ref is None:
            basename = source_path.replace("\\", "/").rsplit("/", 1)[-1]
            ref = refs_by_name.get(basename)
        if ref is None:
            continue
        snippet = _read_snippet(ref.source_path)
        hits.append(
            WorkspaceSearchHitV1(
                document_id=ref.document_id,
                source_id=ref.source_id,
                workspace_id=workspace_id,
                source_path=ref.source_path,
                file_name=ref.file_name,
                score=1.0,
                snippet=snippet,
                metadata={},
            )
        )
        if len(hits) >= limit:
            break
    return hits


def _read_snippet(source_path: str, *, limit: int = 280) -> str:
    try:
        path = Path(source_path)
        if not path.is_file():
            return ""
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if len(text) <= limit:
            return text
        return text[:limit].rstrip() + "…"
    except OSError:
        return ""


def _extract_search_evidence(task_result: Any) -> list[Any]:
    execution = getattr(task_result, "execution_result", None)
    if execution is not None:
        structured = getattr(execution, "structured_data", None)
        if isinstance(structured, dict):
            summary = structured.get("search_summary")
            if isinstance(summary, dict) and isinstance(summary.get("evidence"), list):
                return list(summary["evidence"])
            for value in structured.values():
                if isinstance(value, dict) and isinstance(value.get("evidence"), list):
                    return list(value["evidence"])
                if isinstance(value, dict):
                    nested = value.get("search_summary")
                    if isinstance(nested, dict) and isinstance(nested.get("evidence"), list):
                        return list(nested["evidence"])

    metadata = getattr(task_result, "metadata", None)
    if isinstance(metadata, dict):
        for key in ("search_summary", "lkw_search_summary"):
            summary = metadata.get(key)
            if isinstance(summary, dict) and isinstance(summary.get("evidence"), list):
                return list(summary["evidence"])
        evidence = metadata.get("evidence")
        if isinstance(evidence, list):
            return list(evidence)
    return []


def _extract_source_refs(task_result: Any) -> list[str]:
    metadata = getattr(task_result, "metadata", None)
    if isinstance(metadata, dict):
        evidence = metadata.get("lkw_evidence.v1")
        if isinstance(evidence, dict):
            diagnostics = evidence.get("diagnostics")
            if isinstance(diagnostics, dict):
                search_diag = diagnostics.get("lkw.search_summary.v1")
                if isinstance(search_diag, dict):
                    refs = search_diag.get("source_refs")
                    if isinstance(refs, list):
                        return [str(item) for item in refs if str(item).strip()]

    execution = getattr(task_result, "execution_result", None)
    if execution is None:
        return []
    structured = getattr(execution, "structured_data", None)
    if not isinstance(structured, dict):
        return []
    summary = structured.get("search_summary")
    if isinstance(summary, dict):
        evidence = summary.get("evidence")
        if isinstance(evidence, list):
            refs = [
                str(item.get("source_path"))
                for item in evidence
                if isinstance(item, dict) and item.get("source_path")
            ]
            if refs:
                return refs
    trace = structured.get(AcpStructuredDataKey.TRACE_SUMMARY)
    step_diagnostics: dict[str, Any] = {}
    if isinstance(trace, dict):
        raw = trace.get("step_diagnostics")
        if isinstance(raw, dict):
            step_diagnostics = raw
    search_diag = step_diagnostics.get("lkw.search_summary.v1")
    if isinstance(search_diag, dict):
        refs = search_diag.get("source_refs")
        if isinstance(refs, list):
            return [str(item) for item in refs if str(item).strip()]
    return []
