# © Artur Czarnecki. All rights reserved.

"""Public managed workspace HTTP routes (LKW-PRODUCT-1 / LKW-PRODUCT-1-HARDENING)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, FastAPI, Header, HTTPException, Request, status

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
    WorkspaceOperationStatus,
    WorkspaceSource,
)
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository
from local_workspace_application.workspaces.service import (
    ConcurrentSyncError,
    ManagedWorkspaceService,
)
from local_workspace_application.workspaces.sync_enqueue import enqueue_managed_workspace_sync
from local_workspace_application.workspaces.sync_jobs import ManagedWorkspaceSyncJob
from local_workspace_application.workspaces.sync_runtime import (
    ManagedWorkspaceSyncRuntime,
    build_managed_workspace_sync_runtime,
)
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
    sync_runtime: ManagedWorkspaceSyncRuntime | None = None,
) -> ManagedWorkspaceService:
    from pathlib import Path

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
    if sync_runtime is None:
        sync_runtime = build_managed_workspace_sync_runtime(
            document_store=repository.document_store,
            sync_service=sync_service,
            repository=repository,
        )
        app.state.lkw_managed_workspace_sync_runtime = sync_runtime

        @app.on_event("startup")
        async def _start_managed_workspace_sync_runtime() -> None:
            import asyncio

            sync_runtime.bind_main_loop(asyncio.get_running_loop())
            sync_runtime.start()

        @app.on_event("shutdown")
        async def _stop_managed_workspace_sync_runtime() -> None:
            sync_runtime.stop()

    app.state.lkw_managed_workspace_service = service
    app.state.lkw_managed_workspace_repository = repository
    app.state.lkw_managed_workspace_sync_runtime = sync_runtime

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
        except ConcurrentSyncError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "error": "sync_already_in_progress",
                    "operation_id": exc.active.operation_id,
                    "status": exc.active.status.value,
                },
            ) from exc

        job = ManagedWorkspaceSyncJob(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            source_id=source_id,
            operation_id=operation.operation_id,
            operation_type="source_sync",
        )
        try:
            enqueue_managed_workspace_sync(sync_runtime.wiring_context, job)
        except Exception as exc:
            service.repository.put_operation(
                operation.model_copy(
                    update={
                        "status": WorkspaceOperationStatus.FAILED,
                        "error": f"enqueue_failed:{exc.__class__.__name__}",
                    }
                )
            )
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="sync_enqueue_failed",
            ) from exc

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

        result_metadata = dict(getattr(result, "metadata", None) or {})
        attach_lkw_evidence_metadata(
            result_metadata,
            task_result=result,
            capability="local.workspace.search",
        )
        result = result.model_copy(update={"metadata": result_metadata})

        try:
            hits = _map_search_hits(
                repository=repository,
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                task_result=result,
                limit=body.limit,
            )
        except SearchEvidenceIncompleteError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="search_evidence_incomplete",
            ) from exc

        return WorkspaceSearchResponseV1(
            workspace_id=workspace_id,
            query=body.query,
            results=hits,
        )

    app.include_router(router)
    return service


class SearchEvidenceIncompleteError(RuntimeError):
    """Platform result lacked required typed search evidence fields."""


def _map_search_hits(
    *,
    repository: ManagedWorkspaceRepository,
    tenant_id: str,
    workspace_id: str,
    task_result: Any,
    limit: int,
) -> list[WorkspaceSearchHitV1]:
    summary = _extract_search_summary(task_result)
    if summary is None:
        raise SearchEvidenceIncompleteError("search_summary_missing")
    evidence = summary.get("evidence")
    if not isinstance(evidence, list):
        raise SearchEvidenceIncompleteError("search_evidence_missing")

    hits: list[WorkspaceSearchHitV1] = []
    incomplete = False
    for item in evidence:
        if not isinstance(item, dict):
            continue
        document_id = str(item.get("document_id") or "").strip()
        source_id = str(item.get("source_id") or "").strip()
        item_workspace_id = str(item.get("workspace_id") or workspace_id).strip()
        source_path = str(item.get("source_path") or "").strip()
        file_name = str(item.get("file_name") or "").strip()
        score_raw = item.get("score")
        snippet = str(item.get("snippet") or item.get("text") or "").strip()
        metadata = item.get("metadata")

        if not document_id or not source_id or not source_path or not file_name:
            incomplete = True
            continue
        if not isinstance(score_raw, (int, float)):
            incomplete = True
            continue
        if not snippet:
            incomplete = True
            continue
        if item_workspace_id != workspace_id:
            continue

        ref = repository.get_document_ref(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            document_id=document_id,
        )
        if ref is None:
            continue
        if ref.source_id != source_id or ref.tenant_id != tenant_id:
            continue
        if normalize_source_path(ref.source_path) != normalize_source_path(source_path):
            # Provenance mismatch — drop rather than fabricate.
            continue

        hits.append(
            WorkspaceSearchHitV1(
                document_id=document_id,
                source_id=source_id,
                workspace_id=workspace_id,
                source_path=ref.source_path,
                file_name=file_name,
                score=float(score_raw),
                snippet=snippet,
                metadata=dict(metadata) if isinstance(metadata, dict) else {},
            )
        )
        if len(hits) >= limit:
            break

    if not hits and incomplete:
        raise SearchEvidenceIncompleteError("search_evidence_incomplete")
    if not hits and evidence:
        raise SearchEvidenceIncompleteError("search_evidence_unverified")
    return hits


def _extract_search_summary(task_result: Any) -> dict[str, Any] | None:
    execution = getattr(task_result, "execution_result", None)
    if execution is not None:
        structured = getattr(execution, "structured_data", None)
        if isinstance(structured, dict):
            summary = structured.get("search_summary")
            if isinstance(summary, dict):
                return summary
    return None
