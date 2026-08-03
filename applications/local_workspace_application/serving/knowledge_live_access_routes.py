# © Artur Czarnecki. All rights reserved.

"""HTTP routes for workspace Live Access Binding lifecycle."""

from __future__ import annotations

from fastapi import APIRouter, FastAPI, Header, HTTPException, Request, Response, status

from local_workspace_application.serving.knowledge_configuration_http import (
    hash_knowledge_configuration_idempotency_key,
    map_knowledge_configuration_mutation_error,
    parse_knowledge_configuration_if_match,
    require_knowledge_configuration_idempotency_key,
    resolve_knowledge_configuration_tenant_id,
)
from local_workspace_application.serving.knowledge_live_access_schemas import (
    CreateWorkspaceLiveAccessBindingRequestV1,
    WorkspaceLiveAccessBindingResponseV1,
)
from local_workspace_application.workspaces.knowledge_configuration_mutation_engine import (
    WorkspaceKnowledgeConfigurationMutationError,
    WorkspaceKnowledgeMutationExecutionDispositionV1,
)
from local_workspace_application.workspaces.knowledge_live_access_service import (
    CreateWorkspaceLiveAccessBindingCommand,
    DisableWorkspaceLiveAccessBindingCommand,
    WorkspaceLiveAccessBindingError,
    WorkspaceLiveAccessBindingService,
)
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository


def _map_live_access_error(exc: WorkspaceLiveAccessBindingError) -> HTTPException:
    code = exc.error_code
    if code in {
        "workspace_not_found",
        "connection_not_found",
        "connection_not_attached",
        "live_access_binding_not_found",
    }:
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=code)
    if code in {
        "configuration_revision_conflict",
        "configuration_idempotency_conflict",
    }:
        return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=code)
    if code in {
        "capability_not_found",
        "capability_not_read_only",
        "capability_catalog_invalid",
        "remote_resource_required",
        "remote_resource_not_found",
        "remote_resource_unavailable",
        "remote_resource_connection_mismatch",
        "remote_resource_capability_mismatch",
        "remote_resource_type_unsupported",
        "blank_capability_id",
        "blank_remote_resource_id",
        "allowed_capability_ids_required",
    }:
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=code)
    if code in {
        "connection_unavailable",
        "capability_catalog_unavailable",
        "remote_resource_lookup_unavailable",
        "remote_resource_lookup_invalid",
        "live_access_projection_incomplete",
        "configuration_recovery_required",
        "configuration_mutation_cleanup_failed",
    }:
        return HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=code)
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=code)


def _resolve_historical_live_binding(
    repository: ManagedWorkspaceRepository,
    *,
    tenant_id: str,
    workspace_id: str,
    live_access_binding_id: str,
    configuration_revision: int,
) -> object:
    binding = None
    for version in repository.list_knowledge_live_access_versions(
        tenant_id=tenant_id,
        workspace_id=workspace_id,
    ):
        if version.live_access_binding_id != live_access_binding_id:
            continue
        if version.effective_revision <= configuration_revision:
            if binding is None or version.effective_revision > binding.effective_revision:
                binding = version
    if binding is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="live_access_binding_not_found",
        )
    return binding


def _live_access_response(
    *,
    workspace_id: str,
    binding: object,
    configuration_revision: int,
) -> WorkspaceLiveAccessBindingResponseV1:
    return WorkspaceLiveAccessBindingResponseV1(
        workspace_id=workspace_id,
        live_access_binding_id=binding.live_access_binding_id,
        connection_ref=binding.connection_ref,
        remote_resource_id=binding.remote_resource_id,
        allowed_capability_ids=binding.allowed_capability_ids,
        derived_provider_id=binding.derived_provider_id,
        derived_integration_kind=binding.derived_integration_kind.value,
        derived_resource_type=binding.derived_resource_type,
        derived_safe_display_label=binding.derived_safe_display_label,
        status=binding.status.value,
        audience_eligibility=binding.audience_eligibility.value,
        configuration_revision=configuration_revision,
    )


def mount_knowledge_live_access_routes(
    app: FastAPI,
    *,
    live_access_service: WorkspaceLiveAccessBindingService,
    repository: ManagedWorkspaceRepository,
    prefix: str = "/v1/local_workspace",
) -> None:
    router = APIRouter(prefix=prefix, tags=["knowledge-live-access"])

    @router.post(
        "/workspaces/{workspace_id}/knowledge/live-access-bindings",
        response_model=WorkspaceLiveAccessBindingResponseV1,
    )
    async def create_live_access_binding(
        request: Request,
        workspace_id: str,
        body: CreateWorkspaceLiveAccessBindingRequestV1,
        x_tenant_id: str | None = Header(default=None, alias="X-Tenant-Id"),
        if_match: str | None = Header(default=None, alias="If-Match"),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> Response:
        tenant_id = resolve_knowledge_configuration_tenant_id(
            request,
            header_tenant_id=x_tenant_id,
        )
        expected_revision = parse_knowledge_configuration_if_match(if_match)
        idem_hash = hash_knowledge_configuration_idempotency_key(
            require_knowledge_configuration_idempotency_key(idempotency_key)
        )
        try:
            result = await live_access_service.create_live_access_binding(
                CreateWorkspaceLiveAccessBindingCommand(
                    tenant_id=tenant_id,
                    workspace_id=workspace_id,
                    connection_ref=body.connection_ref,
                    remote_resource_id=body.remote_resource_id,
                    allowed_capability_ids=body.allowed_capability_ids,
                    expected_revision=expected_revision,
                    idempotency_key_hash=idem_hash,
                    audience_eligibility=body.audience_eligibility,
                )
            )
        except WorkspaceLiveAccessBindingError as exc:
            raise _map_live_access_error(exc) from exc
        except WorkspaceKnowledgeConfigurationMutationError as exc:
            raise map_knowledge_configuration_mutation_error(exc) from exc
        except ValueError as exc:
            code = str(exc)
            if code in {
                "blank_capability_id",
                "blank_remote_resource_id",
                "allowed_capability_ids_required",
            }:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=code) from exc
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="invalid_request") from exc

        binding = _resolve_historical_live_binding(
            repository,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            live_access_binding_id=result.binding.live_access_binding_id,
            configuration_revision=result.configuration_revision,
        )
        status_code = status.HTTP_200_OK
        if (
            result.disposition is WorkspaceKnowledgeMutationExecutionDispositionV1.APPLIED
            and result.created_new_binding
        ):
            status_code = status.HTTP_201_CREATED
        payload = _live_access_response(
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
        "/workspaces/{workspace_id}/knowledge/live-access-bindings/{live_access_binding_id}",
        response_model=WorkspaceLiveAccessBindingResponseV1,
    )
    async def disable_live_access_binding(
        request: Request,
        workspace_id: str,
        live_access_binding_id: str,
        x_tenant_id: str | None = Header(default=None, alias="X-Tenant-Id"),
        if_match: str | None = Header(default=None, alias="If-Match"),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> Response:
        tenant_id = resolve_knowledge_configuration_tenant_id(
            request,
            header_tenant_id=x_tenant_id,
        )
        expected_revision = parse_knowledge_configuration_if_match(if_match)
        idem_hash = hash_knowledge_configuration_idempotency_key(
            require_knowledge_configuration_idempotency_key(idempotency_key)
        )
        try:
            result = live_access_service.disable_live_access_binding(
                DisableWorkspaceLiveAccessBindingCommand(
                    tenant_id=tenant_id,
                    workspace_id=workspace_id,
                    live_access_binding_id=live_access_binding_id,
                    expected_revision=expected_revision,
                    idempotency_key_hash=idem_hash,
                )
            )
        except WorkspaceLiveAccessBindingError as exc:
            raise _map_live_access_error(exc) from exc
        except WorkspaceKnowledgeConfigurationMutationError as exc:
            raise map_knowledge_configuration_mutation_error(exc) from exc

        binding = _resolve_historical_live_binding(
            repository,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            live_access_binding_id=result.binding.live_access_binding_id,
            configuration_revision=result.configuration_revision,
        )
        payload = _live_access_response(
            workspace_id=workspace_id,
            binding=binding,
            configuration_revision=result.configuration_revision,
        )
        return Response(
            content=payload.model_dump_json(),
            media_type="application/json",
            status_code=status.HTTP_200_OK,
        )

    app.include_router(router)
