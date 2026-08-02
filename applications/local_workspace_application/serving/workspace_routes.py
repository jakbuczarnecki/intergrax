# © Artur Czarnecki. All rights reserved.

"""Public managed workspace HTTP routes (LKW-PRODUCT-1 / LKW-PRODUCT-1-HARDENING)."""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from fastapi import APIRouter, FastAPI, File, Header, HTTPException, Request, UploadFile, status

from intergrax.fastapi_core.context import get_request_context
from intergrax.integrations.contracts.object_storage import ObjectStorage
from intergrax.runtime.task.task import Task, TaskContext
from intergrax.runtime.task.task_run_bridge import new_run_id
from intergrax.tools.providers.filesystem.allowlist import read_allowlist_roots_from_env
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings
from local_workspace_application.host.task_executor import LocalWorkspaceTaskExecutor
from local_workspace_application.serving.run_metadata import attach_lkw_evidence_metadata
from local_workspace_application.serving.source_projection import safe_source_label
from local_workspace_application.serving.workspace_schemas import (
    CreateWorkspaceRequestV1,
    ManagedFileBatchAcceptedV1,
    ManagedFileBatchItemAcceptedV1,
    OperationResponseV1,
    RegisterSourceRequestV1,
    SourceCandidateAcceptedV1,
    SourceCandidateListResponseV1,
    SourceCandidateSummaryV1,
    SourceListResponseV1,
    SourceResponseV1,
    SourceSummaryResponseV1,
    SyncOperationAcceptedV1,
    WebUrlAcceptedV1,
    WebUrlIntakeRequestV1,
    WorkspaceAskCitationLocationV1,
    WorkspaceAskCitationV1,
    WorkspaceAskErrorV1,
    WorkspaceAskRequestV1,
    WorkspaceAskResponseV1,
    WorkspaceListResponseV1,
    WorkspaceResponseV1,
    WorkspaceSearchRequestV1,
    WorkspaceSearchResponseV1,
)
from local_workspace_application.workspaces.ask_models import WorkspaceAskRun
from local_workspace_application.workspaces.ask_repository import WorkspaceAskRepository
from local_workspace_application.workspaces.ask_service import (
    WorkspaceAskLookupError,
    WorkspaceAskNotFoundError,
    WorkspaceAskPersistenceError,
    WorkspaceAskService,
)
from local_workspace_application.workspaces.document_store_factory import (
    resolve_managed_workspace_document_store,
)
from local_workspace_application.workspaces.document_indexing import (
    WorkspaceDocumentIndexingService,
)
from local_workspace_application.workspaces.ingestion_recovery import (
    KnowledgeIngestionRecoveryService,
)
from local_workspace_application.workspaces.knowledge_ingestion import (
    KnowledgeIngestionProcessorRouter,
    KnowledgeIngestionService,
)
from local_workspace_application.workspaces.knowledge_intake import (
    KnowledgeInputSourceResolverRouter,
    KnowledgeIntakeDispatchError,
    KnowledgeIntakeService,
)
from local_workspace_application.workspaces.local_folder_indexing import (
    LocalFolderIndexingService,
)
from local_workspace_application.workspaces.managed_file_ingestion import (
    ManagedFileKnowledgeIngestionProcessor,
    ManagedObjectMaterializer,
)
from local_workspace_application.workspaces.managed_files import (
    IntakeBatchIdempotencyConflict,
    ManagedFileBatchCandidate,
    ManagedFileIntakeService,
    ManagedFileObjectCleanup,
    ManagedFileSourceResolver,
    ManagedFileValidationError,
    managed_file_request_fingerprint,
    normalize_managed_file_item_error_code,
)
from local_workspace_application.workspaces.models import (
    IntakeBatchItemStatus,
    IntakeBatchStatus,
    KnowledgeInputKind,
    Workspace,
    WorkspaceOperation,
    WorkspaceOperationStatus,
    WorkspaceSource,
)
from local_workspace_application.workspaces.source_candidates import (
    SourceCandidateAlreadyRegistered,
    SourceCandidateIdempotencyConflict,
    SourceCandidateIntakeService,
    SourceCandidateKnowledgeIngestionProcessor,
    SourceCandidateRegistry,
    SourceCandidateRegistryError,
    SourceCandidateSourceResolver,
    SourceCandidateUnavailable,
)
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository
from local_workspace_application.workspaces.search_evidence import (
    SearchEvidenceIncompleteError,
    map_search_hits,
)
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
from local_workspace_application.workspaces.web_url_ingestion import (
    WebUrlAlreadyRegistered,
    WebUrlIdempotencyConflict,
    WebUrlIntakeService,
    WebUrlKnowledgeIngestionProcessor,
    WebUrlSourceResolver,
    WebUrlStateConflict,
    WebUrlTextMaterializer,
    WebUrlValidationError,
    http_status_for_web_url_error,
)
from local_workspace_application.workspaces.sync_service import ManagedWorkspaceSyncService
from local_workspace_application.workspaces.knowledge_configuration_handlers import (
    CreateIndexedSourceMutationHandler,
)
from local_workspace_application.workspaces.knowledge_configuration_mutation_engine import (
    WorkspaceKnowledgeConfigurationMutationEngine,
)
from local_workspace_application.workspaces.knowledge_configuration_models import (
    WorkspaceKnowledgeMutationOperationV1,
)
from local_workspace_application.workspaces.knowledge_configuration_service import (
    WorkspaceKnowledgeConfigurationService,
)
from local_workspace_application.workspaces.connected_source_wiring import (
    ConnectedSourceWiring,
)
from local_workspace_application.serving.knowledge_connected_source_routes import (
    mount_connected_source_knowledge_routes,
)
from local_workspace_application.workspaces.vector_cleanup import (
    VectorstoreManagerWorkspaceCleanup,
)

logger = logging.getLogger(__name__)


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


def _source_summary_response(source: WorkspaceSource) -> SourceSummaryResponseV1:
    source_type = source.source_type.value
    return SourceSummaryResponseV1(
        source_id=source.source_id,
        workspace_id=source.workspace_id,
        source_type=source_type,
        label=safe_source_label(source_type=source_type, path=source.path),
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


async def _prepare_managed_file_batch_candidate(
    upload: UploadFile,
    *,
    max_bytes: int,
) -> ManagedFileBatchCandidate:
    raw_file_name = upload.filename or ""
    raw_content_type = upload.content_type or ""
    hasher = hashlib.sha256()
    chunks: list[bytes] = []
    total = 0
    exceeded = False
    read_failed = False
    try:
        while True:
            chunk = await upload.read(64 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
            total += len(chunk)
            if exceeded:
                continue
            if total > max_bytes:
                exceeded = True
                chunks.clear()
            else:
                chunks.append(chunk)
    except Exception:
        read_failed = True
    try:
        await upload.close()
    except Exception:
        read_failed = True

    body_hash = f"sha256:{hasher.hexdigest()}"
    request_state = "read_failed" if read_failed else "complete"
    request_fingerprint = managed_file_request_fingerprint(
        raw_file_name=raw_file_name,
        raw_content_type=raw_content_type,
        size_bytes=total,
        body_hash=body_hash,
        request_state=request_state,
    )
    if read_failed:
        return ManagedFileBatchCandidate(
            raw_file_name=raw_file_name,
            raw_content_type=raw_content_type,
            body=None,
            size_bytes=total,
            body_hash=body_hash,
            request_fingerprint=request_fingerprint,
            preflight_error_code="managed_file_upload_read_failed",
        )
    if exceeded:
        return ManagedFileBatchCandidate(
            raw_file_name=raw_file_name,
            raw_content_type=raw_content_type,
            body=None,
            size_bytes=total,
            body_hash=body_hash,
            request_fingerprint=request_fingerprint,
            preflight_error_code="managed_file_too_large",
        )
    return ManagedFileBatchCandidate(
        raw_file_name=raw_file_name,
        raw_content_type=raw_content_type,
        body=b"".join(chunks),
        size_bytes=total,
        body_hash=body_hash,
        request_fingerprint=request_fingerprint,
        preflight_error_code=None,
    )


def mount_managed_workspace_routes(
    app: FastAPI,
    *,
    task_executor: LocalWorkspaceTaskExecutor,
    settings: LocalWorkspaceBackendSettings,
    prefix: str = "/v1/local_workspace",
    repository: ManagedWorkspaceRepository | None = None,
    sync_runtime: ManagedWorkspaceSyncRuntime | None = None,
    ask_service: WorkspaceAskService | None = None,
    llm_adapter: Any | None = None,
    vectorstore_manager: Any | None = None,
    object_storage: ObjectStorage | None = None,
    web_url_access_policy: Any | None = None,
    web_content_capture: Any | None = None,
    indexing_service: WorkspaceDocumentIndexingService | None = None,
    connected_source_wiring: ConnectedSourceWiring | None = None,
    shared_slack_integration: Any | None = None,
) -> ManagedWorkspaceService:
    from pathlib import Path

    from intergrax.runtime.wiring.llm_resolver import resolve_llm_adapter
    from intergrax.websearch.capture import SecureHttpWebContentCapture, WebUrlAccessPolicy

    configured = settings.allowed_read_roots or read_allowlist_roots_from_env()
    allowlist = frozenset(
        set(configured) | {settings.managed_upload_staging_dir, settings.web_url_staging_dir}
    )
    shadow_roots = (Path(settings.shadow_workspaces_dir),)
    if repository is None:
        repository = ManagedWorkspaceRepository(resolve_managed_workspace_document_store())
    ask_repository = WorkspaceAskRepository(repository.document_store)
    vector_cleanup = None
    if vectorstore_manager is not None:
        vector_cleanup = VectorstoreManagerWorkspaceCleanup(vectorstore_manager)

    managed_file_cleanup = None
    if object_storage is not None:
        managed_file_cleanup = ManagedFileObjectCleanup(repository, object_storage)

    service = ManagedWorkspaceService(
        repository,
        allowlist_roots=frozenset(allowlist) if allowlist else None,
        shadow_roots=shadow_roots,
        ask_repository=ask_repository,
        vector_cleanup=vector_cleanup,
        managed_file_cleanup=managed_file_cleanup,
    )
    indexing_service = indexing_service or WorkspaceDocumentIndexingService(
        repository,
        task_executor,
    )
    folder_indexing = LocalFolderIndexingService(
        indexing_service,
        allowlist_roots=frozenset(allowlist) if allowlist else None,
    )
    configuration_service = WorkspaceKnowledgeConfigurationService(repository, service)
    mutation_engine = WorkspaceKnowledgeConfigurationMutationEngine(
        repository,
        service,
        configuration_service,
        {
            WorkspaceKnowledgeMutationOperationV1.CREATE_INDEXED_SOURCE: (
                CreateIndexedSourceMutationHandler()
            ),
        },
    )
    connected_wiring = connected_source_wiring
    if connected_wiring is None and (
        shared_slack_integration is not None
        or settings.connected_source_opaque_ref_signing_key.strip()
        or settings.slack_companion_enabled
    ):
        from local_workspace_application.workspaces.connected_source_host_wiring import (
            build_connected_source_host_bundle,
        )

        host_bundle = build_connected_source_host_bundle(
            settings=settings,
            repository=repository,
            workspace_service=service,
            configuration_service=configuration_service,
            mutation_engine=mutation_engine,
            indexing_service=indexing_service,
            slack_integration=shared_slack_integration,
            sync_runtime=sync_runtime,
        )
        connected_wiring = host_bundle.wiring
        app.state.lkw_connected_source_readiness = host_bundle.readiness
    recovery_tenant_ids: tuple[str, ...] = ()
    if connected_wiring is not None and settings.slack_tenant_id.strip():
        recovery_tenant_ids = (settings.slack_tenant_id.strip(),)
    sync_service = ManagedWorkspaceSyncService(
        repository,
        task_executor,
        allowlist_roots=frozenset(allowlist) if allowlist else None,
        indexing_service=indexing_service,
        folder_indexing=folder_indexing,
        connected_source_sync=(
            connected_wiring.connected_source_sync_service if connected_wiring is not None else None
        ),
    )
    owns_runtime = sync_runtime is None
    if sync_runtime is None:
        sync_runtime = build_managed_workspace_sync_runtime(
            document_store=repository.document_store,
            sync_service=sync_service,
            repository=repository,
            connected_source_recovery_tenant_ids=recovery_tenant_ids,
        )
        app.state.lkw_managed_workspace_sync_runtime = sync_runtime

    if connected_wiring is not None:
        connected_wiring.connected_source_sync_service.attach_continuation(
            __import__(
                "local_workspace_application.workspaces.connected_source_wiring",
                fromlist=["_SyncRuntimeContinuation"],
            )._SyncRuntimeContinuation(sync_runtime)
        )
        connected_wiring.connected_source_sync_service.attach_sync_enqueue_context(
            sync_runtime.wiring_context
        )
        app.state.lkw_connected_source_wiring = connected_wiring
        mount_connected_source_knowledge_routes(
            app,
            wiring=connected_wiring,
            workspace_service=service,
            sync_runtime=sync_runtime,
            prefix=prefix,
        )

    source_candidate_registry = SourceCandidateRegistry.load(settings.source_candidates_file)
    resolver_map: dict[KnowledgeInputKind, object] = {
        KnowledgeInputKind.SOURCE_CANDIDATE: SourceCandidateSourceResolver(
            repository,
            source_candidate_registry,
            allowlist_roots=frozenset(allowlist) if allowlist else None,
            shadow_roots=shadow_roots,
        ),
    }
    processor_map: dict[KnowledgeInputKind, object] = {
        KnowledgeInputKind.SOURCE_CANDIDATE: SourceCandidateKnowledgeIngestionProcessor(
            folder_indexing,
        ),
    }

    if object_storage is not None:
        materializer = ManagedObjectMaterializer(
            object_storage,
            Path(settings.managed_upload_staging_dir),
        )
        processor_map[KnowledgeInputKind.MANAGED_FILE] = ManagedFileKnowledgeIngestionProcessor(
            repository,
            materializer,
            indexing_service,
        )
        resolver_map[KnowledgeInputKind.MANAGED_FILE] = ManagedFileSourceResolver(repository)

    web_url_policy = web_url_access_policy or WebUrlAccessPolicy()
    web_url_capture = web_content_capture or SecureHttpWebContentCapture(policy=web_url_policy)
    web_url_materializer = WebUrlTextMaterializer(Path(settings.web_url_staging_dir))
    processor_map[KnowledgeInputKind.WEB_URL] = WebUrlKnowledgeIngestionProcessor(
        repository,
        web_url_capture,
        indexing_service,
        web_url_materializer,
    )
    resolver_map[KnowledgeInputKind.WEB_URL] = WebUrlSourceResolver(repository)

    source_resolver = KnowledgeInputSourceResolverRouter(resolver_map)  # type: ignore[arg-type]
    processor = KnowledgeIngestionProcessorRouter(processor_map)  # type: ignore[arg-type]
    knowledge_ingestion_service = KnowledgeIngestionService(repository, processor)
    sync_runtime.register_knowledge_ingestion_service(knowledge_ingestion_service)
    knowledge_intake_service = KnowledgeIntakeService(
        repository,
        source_resolver,
        sync_runtime.wiring_context,
    )
    managed_file_intake_service: ManagedFileIntakeService | None = None
    if object_storage is not None:
        managed_file_intake_service = ManagedFileIntakeService(
            repository,
            object_storage,
            knowledge_intake_service,
            max_bytes=settings.managed_file_max_bytes,
            max_batch_files=settings.managed_file_max_batch_files,
        )
    recovery = KnowledgeIngestionRecoveryService(repository, knowledge_intake_service)
    sync_runtime.attach_recovery_service(recovery)
    source_candidate_intake_service = SourceCandidateIntakeService(
        repository,
        source_candidate_registry,
        knowledge_intake_service,
        allowlist_roots=frozenset(allowlist) if allowlist else None,
        shadow_roots=shadow_roots,
    )
    web_url_intake_service = WebUrlIntakeService(
        repository,
        knowledge_intake_service,
        web_url_policy,
        preflight_timeout_seconds=settings.web_url_preflight_timeout_seconds,
    )

    if owns_runtime:

        @app.on_event("startup")
        async def _start_managed_workspace_sync_runtime() -> None:
            import asyncio

            sync_runtime.bind_main_loop(asyncio.get_running_loop())
            sync_runtime.start()

        @app.on_event("shutdown")
        async def _stop_managed_workspace_sync_runtime() -> None:
            sync_runtime.stop()

    if ask_service is None:
        ask_service = WorkspaceAskService(
            workspace_service=service,
            workspace_repository=repository,
            ask_repository=ask_repository,
            task_executor=task_executor,
            llm_adapter=llm_adapter,
            llm_adapter_factory=(None if llm_adapter is not None else (lambda: resolve_llm_adapter(None))),
        )

    app.state.lkw_managed_workspace_service = service
    app.state.lkw_managed_workspace_repository = repository
    app.state.lkw_managed_workspace_sync_runtime = sync_runtime
    app.state.lkw_ask_service = ask_service
    app.state.lkw_managed_file_intake_service = managed_file_intake_service
    app.state.lkw_knowledge_intake_service = knowledge_intake_service
    app.state.lkw_knowledge_ingestion_service = knowledge_ingestion_service
    app.state.lkw_source_candidate_registry = source_candidate_registry
    app.state.lkw_source_candidate_intake_service = source_candidate_intake_service
    app.state.lkw_web_url_intake_service = web_url_intake_service
    app.state.lkw_local_folder_indexing_service = folder_indexing

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

    @router.delete(
        "/workspaces/{workspace_id}",
        status_code=status.HTTP_204_NO_CONTENT,
        response_model=None,
    )
    async def delete_workspace(
        request: Request,
        workspace_id: str,
        x_tenant_id: str | None = Header(default=None, alias="X-Tenant-Id"),
    ) -> None:
        tenant_id = resolve_tenant_id(request, header_tenant_id=x_tenant_id)
        current: ManagedWorkspaceService = getattr(
            request.app.state, "lkw_managed_workspace_service", service
        )
        try:
            deleted = current.delete_workspace(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
            )
        except RuntimeError as exc:
            logger.warning(
                "workspace_delete_failed kind=%s",
                type(exc).__name__,
            )
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="workspace_delete_failed",
            ) from exc
        if not deleted:
            raise _not_found()

    @router.post(
        "/workspaces/{workspace_id}/knowledge/files",
        response_model=ManagedFileBatchAcceptedV1,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def upload_managed_files(
        request: Request,
        workspace_id: str,
        files: list[UploadFile] = File(...),
        x_tenant_id: str | None = Header(default=None, alias="X-Tenant-Id"),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> ManagedFileBatchAcceptedV1:
        tenant_id = resolve_tenant_id(request, header_tenant_id=x_tenant_id)
        intake: ManagedFileIntakeService | None = getattr(
            request.app.state, "lkw_managed_file_intake_service", managed_file_intake_service
        )
        if intake is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="managed_file_storage_unavailable",
            )
        if not idempotency_key or not idempotency_key.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="idempotency_key_required",
            )
        if not files:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="managed_file_batch_empty",
            )
        if len(files) > settings.managed_file_max_batch_files:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail="managed_file_batch_too_large",
            )
        if service.require_workspace(tenant_id=tenant_id, workspace_id=workspace_id) is None:
            raise _not_found()

        candidates: list[ManagedFileBatchCandidate] = []
        for upload in files:
            candidates.append(
                await _prepare_managed_file_batch_candidate(
                    upload,
                    max_bytes=settings.managed_file_max_bytes,
                )
            )

        try:
            batch = intake.accept_prepared_many(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                idempotency_key=idempotency_key.strip(),
                candidates=candidates,
            )
        except LookupError:
            raise _not_found() from None
        except IntakeBatchIdempotencyConflict:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="intake_batch_idempotency_conflict",
            ) from None
        except ManagedFileValidationError as exc:
            status_code = status.HTTP_400_BAD_REQUEST
            if exc.error_code == "managed_file_batch_too_large":
                status_code = status.HTTP_413_CONTENT_TOO_LARGE
            raise HTTPException(status_code=status_code, detail=exc.error_code) from None

        status_map = {
            IntakeBatchStatus.ACCEPTED: "accepted",
            IntakeBatchStatus.PARTIAL: "partial",
            IntakeBatchStatus.FAILED: "failed",
            IntakeBatchStatus.ACCEPTING: "partial",
        }
        items: list[ManagedFileBatchItemAcceptedV1] = []
        for item in batch.items:
            op_status = None
            if item.operation_id:
                operation = repository.get_operation(
                    tenant_id=tenant_id,
                    operation_id=item.operation_id,
                )
                if operation is not None:
                    op_status = operation.status.value
            items.append(
                ManagedFileBatchItemAcceptedV1(
                    position=item.position,
                    file_name=item.safe_file_name,
                    status=(
                        "accepted"
                        if item.status is IntakeBatchItemStatus.ACCEPTED
                        else "failed"
                    ),
                    input_id=item.input_id,
                    source_id=item.source_id,
                    operation_id=item.operation_id,
                    operation_status=op_status,
                    error_code=(
                        normalize_managed_file_item_error_code(item.error_code)
                        if item.error_code
                        else None
                    ),
                )
            )
        accepted_count = sum(1 for item in items if item.status == "accepted")
        failed_count = sum(1 for item in items if item.status == "failed")
        return ManagedFileBatchAcceptedV1(
            batch_id=batch.batch_id,
            workspace_id=batch.workspace_id,
            status=status_map[batch.status],  # type: ignore[arg-type]
            accepted_count=accepted_count,
            failed_count=failed_count,
            items=items,
        )

    @router.get(
        "/workspaces/{workspace_id}/source-candidates",
        response_model=SourceCandidateListResponseV1,
    )
    async def list_source_candidates(
        request: Request,
        workspace_id: str,
        x_tenant_id: str | None = Header(default=None, alias="X-Tenant-Id"),
    ) -> SourceCandidateListResponseV1:
        tenant_id = resolve_tenant_id(request, header_tenant_id=x_tenant_id)
        try:
            candidates = source_candidate_intake_service.list_candidates(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
            )
        except LookupError:
            raise _not_found() from None
        except SourceCandidateRegistryError:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="source_candidate_registry_unavailable",
            ) from None
        return SourceCandidateListResponseV1(
            workspace_id=workspace_id,
            candidates=[
                SourceCandidateSummaryV1(
                    candidate_id=item.candidate_id,
                    label=item.label,
                    description=item.description,
                    source_type="local_folder",
                    available=item.available,
                )
                for item in candidates
            ],
        )

    @router.post(
        "/workspaces/{workspace_id}/knowledge/source-candidates/{candidate_id}",
        response_model=SourceCandidateAcceptedV1,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def accept_source_candidate(
        request: Request,
        workspace_id: str,
        candidate_id: str,
        x_tenant_id: str | None = Header(default=None, alias="X-Tenant-Id"),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> SourceCandidateAcceptedV1:
        tenant_id = resolve_tenant_id(request, header_tenant_id=x_tenant_id)
        if idempotency_key is None or not idempotency_key.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="idempotency_key_required",
            )
        try:
            accepted = source_candidate_intake_service.accept(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                candidate_id=candidate_id,
                idempotency_key=idempotency_key.strip(),
            )
        except LookupError:
            raise _not_found() from None
        except SourceCandidateRegistryError:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="source_candidate_registry_unavailable",
            ) from None
        except SourceCandidateUnavailable:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="source_candidate_unavailable",
            ) from None
        except SourceCandidateIdempotencyConflict:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="source_candidate_idempotency_conflict",
            ) from None
        except SourceCandidateAlreadyRegistered:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="source_candidate_already_registered",
            ) from None
        except KnowledgeIntakeDispatchError:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="source_candidate_dispatch_failed",
            ) from None
        return SourceCandidateAcceptedV1(
            candidate_id=accepted.candidate_id,
            label=accepted.label,
            workspace_id=accepted.workspace_id,
            source_id=accepted.source_id,
            operation_id=accepted.operation_id,
            status=accepted.status,  # type: ignore[arg-type]
        )

    @router.post(
        "/workspaces/{workspace_id}/knowledge/web-urls",
        response_model=WebUrlAcceptedV1,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def accept_web_url(
        request: Request,
        workspace_id: str,
        body: WebUrlIntakeRequestV1,
        x_tenant_id: str | None = Header(default=None, alias="X-Tenant-Id"),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> WebUrlAcceptedV1:
        tenant_id = resolve_tenant_id(request, header_tenant_id=x_tenant_id)
        if idempotency_key is None or not idempotency_key.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="idempotency_key_required",
            )
        try:
            accepted = await web_url_intake_service.accept(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                raw_url=body.url,
                idempotency_key=idempotency_key.strip(),
            )
        except LookupError:
            raise _not_found() from None
        except WebUrlValidationError as exc:
            raise HTTPException(
                status_code=http_status_for_web_url_error(exc.error_code),
                detail=exc.error_code,
            ) from None
        except WebUrlIdempotencyConflict:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="web_url_idempotency_conflict",
            ) from None
        except WebUrlAlreadyRegistered:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="web_url_already_registered",
            ) from None
        except WebUrlStateConflict as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(exc) or "web_url_state_conflict",
            ) from None
        except KnowledgeIntakeDispatchError:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="web_url_dispatch_failed",
            ) from None
        return WebUrlAcceptedV1(
            input_id=accepted.input_id,
            workspace_id=accepted.workspace_id,
            source_id=accepted.source_id,
            operation_id=accepted.operation_id,
            status=accepted.status,  # type: ignore[arg-type]
            safe_display_url=accepted.safe_display_url,
        )

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
        return SourceListResponseV1(
            sources=[_source_summary_response(item) for item in sources]
        )

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
            hits = map_search_hits(
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

    @router.post(
        "/workspaces/{workspace_id}/ask",
        response_model=WorkspaceAskResponseV1,
    )
    async def ask_workspace(
        request: Request,
        workspace_id: str,
        body: WorkspaceAskRequestV1,
        x_tenant_id: str | None = Header(default=None, alias="X-Tenant-Id"),
    ) -> WorkspaceAskResponseV1:
        header_tenant = (x_tenant_id or "").strip() or None
        tenant_id = resolve_tenant_id(request, header_tenant_id=x_tenant_id)
        if header_tenant and header_tenant != tenant_id:
            logger.warning(
                "ask_tenant_resolution reason=tenant_scope_mismatch"
            )
        current_ask: WorkspaceAskService = getattr(
            request.app.state, "lkw_ask_service", ask_service
        )
        managed_service: ManagedWorkspaceService = getattr(
            request.app.state, "lkw_managed_workspace_service", service
        )
        managed_repository: ManagedWorkspaceRepository | None = getattr(
            request.app.state, "lkw_managed_workspace_repository", None
        )
        current_ask.use_workspace_authority(managed_service, managed_repository)
        try:
            run = await current_ask.ask(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                question=body.question,
                limit=body.limit,
            )
        except WorkspaceAskLookupError as exc:
            reason = exc.reason
            if (
                header_tenant
                and header_tenant != tenant_id
                and reason == "workspace_lookup_failed"
            ):
                reason = "tenant_scope_mismatch"
            logger.warning("ask_workspace_http_404 reason=%s", reason)
            raise _not_found() from exc
        except SearchEvidenceIncompleteError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="search_evidence_incomplete",
            ) from exc
        except WorkspaceAskPersistenceError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="ask_persistence_failed",
            ) from exc
        except Exception as exc:
            # KeyError/IndexError are LookupError subclasses — must not become 404.
            logger.warning(
                "ask_workspace_failed kind=%s",
                type(exc).__name__,
            )
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"ask_error: {exc.__class__.__name__}",
            ) from exc
        return _ask_response(run)

    @router.get(
        "/asks/{run_id}",
        response_model=WorkspaceAskResponseV1,
    )
    async def get_ask_run(
        request: Request,
        run_id: str,
        x_tenant_id: str | None = Header(default=None, alias="X-Tenant-Id"),
    ) -> WorkspaceAskResponseV1:
        tenant_id = resolve_tenant_id(request, header_tenant_id=x_tenant_id)
        current_ask: WorkspaceAskService = getattr(
            request.app.state, "lkw_ask_service", ask_service
        )
        managed_service: ManagedWorkspaceService = getattr(
            request.app.state, "lkw_managed_workspace_service", service
        )
        managed_repository: ManagedWorkspaceRepository | None = getattr(
            request.app.state, "lkw_managed_workspace_repository", None
        )
        current_ask.use_workspace_authority(managed_service, managed_repository)
        try:
            run = current_ask.get_run(tenant_id=tenant_id, run_id=run_id)
        except WorkspaceAskNotFoundError as exc:
            raise _not_found() from exc
        return _ask_response(run)

    app.include_router(router)
    return service


def _ask_response(run: WorkspaceAskRun) -> WorkspaceAskResponseV1:
    return WorkspaceAskResponseV1(
        run_id=run.run_id,
        workspace_id=run.workspace_id,
        status=run.status.value,  # type: ignore[arg-type]
        question=run.question,
        answer=run.answer,
        citations=[
            WorkspaceAskCitationV1(
                evidence_id=item.evidence_id,
                document_id=item.document_id,
                source_id=item.source_id,
                workspace_id=item.workspace_id,
                source_path=item.source_path,
                file_name=item.file_name,
                excerpt=item.excerpt,
                score=item.score,
                chunk_id=item.chunk_id,
                location=(
                    WorkspaceAskCitationLocationV1(page=item.location.page)
                    if item.location is not None
                    else None
                ),
            )
            for item in run.citations
        ],
        created_at=run.created_at,
        completed_at=run.completed_at,
        error=(
            WorkspaceAskErrorV1(code=run.error.code, message=run.error.message)
            if run.error is not None
            else None
        ),
    )


# Backward-compatible aliases for existing hardening tests.
_map_search_hits = map_search_hits
__all__ = ("SearchEvidenceIncompleteError", "mount_managed_workspace_routes", "resolve_tenant_id")
