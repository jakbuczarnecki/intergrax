# © Artur Czarnecki. All rights reserved.

"""HTTP routes for connected workspace knowledge sources."""

from __future__ import annotations

from fastapi import (
    APIRouter,
    FastAPI,
    Header,
    HTTPException,
    Query,
    Request,
    Response,
    status,
)
from local_workspace_application.serving.knowledge_configuration_http import (
    hash_knowledge_configuration_idempotency_key,
    map_knowledge_configuration_mutation_error,
    parse_knowledge_configuration_if_match,
    require_knowledge_configuration_idempotency_key,
    resolve_knowledge_configuration_tenant_id,
)
from local_workspace_application.serving.knowledge_connected_source_schemas import (
    ConnectedIndexedSourceSyncAcceptedV1,
    CreateConnectedIndexedSourceRequestV1,
    CreateConnectedIndexedSourceResponseV1,
    RemoteResourceCandidateResponseV1,
    RemoteResourceDiscoveryResponseV1,
)
from local_workspace_application.workspaces.connected_source_models import (
    ConnectedSourceBindingError,
    ConnectedSourceDiscoveryError,
)
from local_workspace_application.workspaces.connected_source_wiring import (
    ConnectedSourceWiring,
)
from local_workspace_application.workspaces.knowledge_access_service import (
    CreateConnectedIndexedSourceRequest,
)
from local_workspace_application.workspaces.knowledge_configuration_models import (
    WorkspaceIndexedSourceBindingStatusV1,
)
from local_workspace_application.workspaces.knowledge_configuration_mutation_engine import (
    WorkspaceKnowledgeConfigurationMutationError,
    WorkspaceKnowledgeMutationExecutionDispositionV1,
)
from local_workspace_application.workspaces.knowledge_configuration_service import (
    is_workspace_source_product_visible,
)
from local_workspace_application.workspaces.knowledge_indexed_source_lifecycle_service import (
    DisableWorkspaceIndexedSourceCommand,
    WorkspaceIndexedSourceLifecycleError,
)
from local_workspace_application.workspaces.models import (
    WorkspaceOperationStatus,
    WorkspaceSourceStatus,
    WorkspaceSourceType,
)
from local_workspace_application.workspaces.service import (
    ConcurrentSyncError,
    ManagedWorkspaceService,
)
from local_workspace_application.workspaces.sync_enqueue import (
    enqueue_managed_workspace_sync,
)
from local_workspace_application.workspaces.sync_jobs import ManagedWorkspaceSyncJob
from local_workspace_application.workspaces.sync_runtime import (
    ManagedWorkspaceSyncRuntime,
)


def _map_discovery_error(exc: ConnectedSourceDiscoveryError) -> HTTPException:
    code = exc.error_code
    if code in {"workspace_not_found", "connection_not_attached", "candidate_inaccessible"}:
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=code)
    if code in {
        "candidate_ref_invalid",
        "candidate_ref_tampered",
        "discovery_cursor_invalid",
        "discovery_limit_invalid",
        "resource_type_unsupported",
    }:
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=code)
    if code == "connection_unavailable":
        return HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=code)
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=code)


def _map_lifecycle_error(exc: WorkspaceIndexedSourceLifecycleError) -> HTTPException:
    code = exc.error_code
    if code in {
        "workspace_not_found",
        "indexed_source_not_found",
        "knowledge_source_binding_not_found",
        "connection_not_attached",
    }:
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=code)
    if code in {
        "configuration_revision_conflict",
        "configuration_idempotency_conflict",
        "knowledge_source_binding_unavailable",
    }:
        return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=code)
    if code in {
        "connection_unavailable",
        "knowledge_source_binding_invalid",
        "indexed_source_projection_incomplete",
        "configuration_recovery_required",
        "configuration_mutation_cleanup_failed",
    }:
        return HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=code)
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=code)


def _indexed_source_response(
    *,
    workspace_id: str,
    binding: object,
    configuration_revision: int,
) -> CreateConnectedIndexedSourceResponseV1:
    safe_label = binding.cached_safe_display_label
    if not safe_label or not str(safe_label).strip():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="indexed_source_projection_incomplete",
        )
    return CreateConnectedIndexedSourceResponseV1(
        workspace_id=workspace_id,
        indexed_source_binding_id=binding.indexed_source_binding_id,
        source_id=binding.source_id,
        knowledge_source_binding_ref=binding.knowledge_source_binding_ref,
        safe_display_label=str(safe_label),
        status=binding.status.value,
        sync_mode=binding.sync_mode.value,
        audience_eligibility=binding.audience_eligibility.value,
        configuration_revision=configuration_revision,
    )


def _resolve_historical_indexed_binding(
    workspace_service: ManagedWorkspaceService,
    *,
    tenant_id: str,
    workspace_id: str,
    indexed_source_binding_id: str,
    configuration_revision: int,
) -> object:
    binding = None
    for version in workspace_service.repository.list_knowledge_indexed_source_versions(
        tenant_id=tenant_id,
        workspace_id=workspace_id,
    ):
        if version.indexed_source_binding_id != indexed_source_binding_id:
            continue
        if (
            version.effective_revision <= configuration_revision
            and (binding is None or version.effective_revision > binding.effective_revision)
        ):
            binding = version
    if binding is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="indexed_source_not_found",
        )
    return binding


def _resolve_committed_indexed_binding(
    workspace_service: ManagedWorkspaceService,
    *,
    tenant_id: str,
    workspace_id: str,
    indexed_source_binding_id: str,
) -> object:
    configuration = workspace_service.repository.get_knowledge_configuration_head(
        tenant_id=tenant_id,
        workspace_id=workspace_id,
    )
    committed = configuration.committed_revision if configuration is not None else 0
    return _resolve_historical_indexed_binding(
        workspace_service,
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        indexed_source_binding_id=indexed_source_binding_id,
        configuration_revision=committed,
    )


def mount_connected_source_knowledge_routes(
    app: FastAPI,
    *,
    wiring: ConnectedSourceWiring,
    workspace_service: ManagedWorkspaceService,
    sync_runtime: ManagedWorkspaceSyncRuntime,
    prefix: str = "/v1/local_workspace",
) -> None:
    router = APIRouter(prefix=prefix, tags=["connected-source-knowledge"])

    @router.get(
        "/workspaces/{workspace_id}/knowledge/connections/{connection_ref}/remote-resources",
        response_model=RemoteResourceDiscoveryResponseV1,
    )
    async def list_remote_resources(
        request: Request,
        workspace_id: str,
        connection_ref: str,
        resource_type: str = Query(..., min_length=1, max_length=64),
        cursor: str | None = Query(default=None),
        limit: int = Query(default=50, ge=1, le=100),
        x_tenant_id: str | None = Header(default=None, alias="X-Tenant-Id"),
    ) -> RemoteResourceDiscoveryResponseV1:
        tenant_id = resolve_knowledge_configuration_tenant_id(request, header_tenant_id=x_tenant_id)
        from local_workspace_application.workspaces.connected_source_models import (
            RemoteResourceTypeV1,
        )

        try:
            resource_enum = RemoteResourceTypeV1(resource_type)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="resource_type_unsupported",
            ) from None
        try:
            page = await wiring.discovery_service.list_remote_resources(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                connection_ref=connection_ref,
                resource_type=resource_enum,
                cursor=cursor,
                limit=limit,
            )
        except ConnectedSourceDiscoveryError as exc:
            raise _map_discovery_error(exc) from exc
        return RemoteResourceDiscoveryResponseV1(
            items=[
                RemoteResourceCandidateResponseV1(
                    opaque_candidate_ref=item.opaque_candidate_ref,
                    resource_type=item.resource_type.value,
                    safe_display_label=item.safe_display_label,
                    conversation_kind=(
                        item.conversation_kind.value if item.conversation_kind is not None else None
                    ),
                    is_archived=item.is_archived,
                    is_private=item.is_private,
                    safe_description=item.safe_description,
                )
                for item in page.items
            ],
            next_cursor=page.next_cursor,
        )

    @router.post(
        "/workspaces/{workspace_id}/knowledge/indexed-sources",
        response_model=CreateConnectedIndexedSourceResponseV1,
    )
    async def create_indexed_source(
        request: Request,
        workspace_id: str,
        body: CreateConnectedIndexedSourceRequestV1,
        x_tenant_id: str | None = Header(default=None, alias="X-Tenant-Id"),
        if_match: str | None = Header(default=None, alias="If-Match"),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> Response:
        tenant_id = resolve_knowledge_configuration_tenant_id(request, header_tenant_id=x_tenant_id)
        expected_revision = parse_knowledge_configuration_if_match(if_match)
        idem_hash = hash_knowledge_configuration_idempotency_key(
            require_knowledge_configuration_idempotency_key(idempotency_key)
        )
        try:
            result = await wiring.knowledge_access_service.create_indexed_source_from_candidate(
                CreateConnectedIndexedSourceRequest(
                    tenant_id=tenant_id,
                    workspace_id=workspace_id,
                    connection_ref=body.connection_ref,
                    opaque_candidate_ref=body.opaque_candidate_ref,
                    expected_revision=expected_revision,
                    idempotency_key_hash=idem_hash,
                    root_oldest=body.root_oldest,
                    root_latest=body.root_latest,
                )
            )
        except ConnectedSourceDiscoveryError as exc:
            raise _map_discovery_error(exc) from exc
        except WorkspaceIndexedSourceLifecycleError as exc:
            raise _map_lifecycle_error(exc) from exc
        except ConnectedSourceBindingError as exc:
            raise _map_lifecycle_error(
                WorkspaceIndexedSourceLifecycleError(exc.error_code)
            ) from exc
        except WorkspaceKnowledgeConfigurationMutationError as exc:
            raise map_knowledge_configuration_mutation_error(exc) from exc

        binding = _resolve_historical_indexed_binding(
            workspace_service,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            indexed_source_binding_id=result.binding_id,
            configuration_revision=result.configuration_revision,
        )
        disposition = result.mutation_result.disposition
        status_code = status.HTTP_200_OK
        if (
            disposition is WorkspaceKnowledgeMutationExecutionDispositionV1.APPLIED
            and result.created_new_source
        ):
            status_code = status.HTTP_201_CREATED
        payload = _indexed_source_response(
            workspace_id=workspace_id,
            binding=binding,
            configuration_revision=result.configuration_revision,
        )
        return Response(
            content=payload.model_dump_json(),
            media_type="application/json",
            status_code=status_code,
        )

    @router.delete(
        "/workspaces/{workspace_id}/knowledge/indexed-sources/{indexed_source_binding_id}",
        response_model=CreateConnectedIndexedSourceResponseV1,
    )
    async def disable_indexed_source(
        request: Request,
        workspace_id: str,
        indexed_source_binding_id: str,
        x_tenant_id: str | None = Header(default=None, alias="X-Tenant-Id"),
        if_match: str | None = Header(default=None, alias="If-Match"),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> Response:
        tenant_id = resolve_knowledge_configuration_tenant_id(request, header_tenant_id=x_tenant_id)
        expected_revision = parse_knowledge_configuration_if_match(if_match)
        idem_hash = hash_knowledge_configuration_idempotency_key(
            require_knowledge_configuration_idempotency_key(idempotency_key)
        )
        try:
            result = wiring.indexed_source_lifecycle_service.disable_indexed_source(
                DisableWorkspaceIndexedSourceCommand(
                    tenant_id=tenant_id,
                    workspace_id=workspace_id,
                    indexed_source_binding_id=indexed_source_binding_id,
                    expected_revision=expected_revision,
                    idempotency_key_hash=idem_hash,
                )
            )
        except WorkspaceIndexedSourceLifecycleError as exc:
            raise _map_lifecycle_error(exc) from exc
        except WorkspaceKnowledgeConfigurationMutationError as exc:
            raise map_knowledge_configuration_mutation_error(exc) from exc

        binding = _resolve_historical_indexed_binding(
            workspace_service,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            indexed_source_binding_id=result.binding.indexed_source_binding_id,
            configuration_revision=result.configuration_revision,
        )
        payload = _indexed_source_response(
            workspace_id=workspace_id,
            binding=binding,
            configuration_revision=result.configuration_revision,
        )
        return Response(
            content=payload.model_dump_json(),
            media_type="application/json",
            status_code=status.HTTP_200_OK,
        )

    @router.post(
        "/workspaces/{workspace_id}/knowledge/indexed-sources/{indexed_source_binding_id}/sync",
        response_model=ConnectedIndexedSourceSyncAcceptedV1,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def sync_indexed_source(
        request: Request,
        workspace_id: str,
        indexed_source_binding_id: str,
        x_tenant_id: str | None = Header(default=None, alias="X-Tenant-Id"),
    ) -> ConnectedIndexedSourceSyncAcceptedV1:
        tenant_id = resolve_knowledge_configuration_tenant_id(request, header_tenant_id=x_tenant_id)
        binding = _resolve_committed_indexed_binding(
            workspace_service,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            indexed_source_binding_id=indexed_source_binding_id,
        )
        if binding.status is not WorkspaceIndexedSourceBindingStatusV1.ACTIVE:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="indexed_source_inactive",
            )
        source = workspace_service.repository.get_source(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            source_id=binding.source_id,
        )
        if source is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="source_not_found")
        if source.source_type is not WorkspaceSourceType.CONNECTED_SOURCE:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="source_sync_unsupported_for_source_type",
            )
        configuration = workspace_service.repository.get_knowledge_configuration_head(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        )
        committed = configuration.committed_revision if configuration is not None else 0
        if not is_workspace_source_product_visible(
            source,
            committed_configuration_revision=committed,
        ):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="source_not_product_visible",
            )
        if source.status is WorkspaceSourceStatus.ERROR:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="source_unavailable",
            )
        try:
            operation = workspace_service.create_sync_operation(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                source_id=binding.source_id,
            )
        except LookupError:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="source_not_found") from None
        except ConcurrentSyncError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "error": "sync_already_in_progress",
                    "operation_id": exc.active.operation_id,
                },
            ) from exc
        job = ManagedWorkspaceSyncJob(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            source_id=binding.source_id,
            operation_id=operation.operation_id,
            operation_type="source_sync",
        )
        try:
            enqueue_managed_workspace_sync(sync_runtime.wiring_context, job)
        except Exception:  # noqa: BLE001
            workspace_service.repository.put_operation(
                operation.model_copy(
                    update={
                        "status": WorkspaceOperationStatus.FAILED,
                        "error": "sync_enqueue_failed",
                    }
                )
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="sync_enqueue_failed",
            ) from None
        return ConnectedIndexedSourceSyncAcceptedV1(
            operation_id=operation.operation_id,
            workspace_id=workspace_id,
            source_id=binding.source_id,
            status=operation.status.value,
        )

    app.include_router(router)
