# © Artur Czarnecki. All rights reserved.

"""HTTP routes for workspace Query Policy and knowledge configuration projection."""

from __future__ import annotations

from fastapi import APIRouter, FastAPI, Header, HTTPException, Request, Response, status
from pydantic import ValidationError

from local_workspace_application.serving.knowledge_configuration_http import (
    hash_knowledge_configuration_idempotency_key,
    map_knowledge_configuration_mutation_error,
    parse_knowledge_configuration_if_match,
    require_knowledge_configuration_idempotency_key,
    resolve_knowledge_configuration_tenant_id,
)
from local_workspace_application.serving.knowledge_query_policy_schemas import (
    ConnectionAttachmentProjectionV1,
    IndexedSourceBindingProjectionV1,
    LiveAccessBindingProjectionV1,
    QueryPolicyProjectionV1,
    UpdateQueryPolicyRequestV1,
    WorkspaceKnowledgeConfigurationResponseV1,
    WorkspaceQueryPolicyResponseV1,
)
from local_workspace_application.workspaces.knowledge_configuration_models import (
    LiveResultRetentionV1,
    QueryPolicyModeV1,
    WorkspaceKnowledgeConfigurationV1,
    WorkspaceQueryPolicy,
)
from local_workspace_application.workspaces.knowledge_configuration_mutation_engine import (
    WorkspaceKnowledgeConfigurationMutationError,
)
from local_workspace_application.workspaces.knowledge_configuration_service import (
    WorkspaceKnowledgeConfigurationService,
    WorkspaceKnowledgeConfigurationServiceError,
)
from local_workspace_application.workspaces.knowledge_query_policy_service import (
    UpdateWorkspaceQueryPolicyCommand,
    WorkspaceQueryPolicyError,
    WorkspaceQueryPolicyService,
)


def _parse_query_policy_mode(value: str) -> QueryPolicyModeV1:
    try:
        return QueryPolicyModeV1(value)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="query_policy_mode_unsupported",
        ) from exc


def _parse_live_result_retention(value: str) -> LiveResultRetentionV1:
    try:
        return LiveResultRetentionV1(value)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="query_policy_invalid",
        ) from exc


def _map_query_policy_error(exc: WorkspaceQueryPolicyError) -> HTTPException:
    code = exc.error_code
    if code == "workspace_not_found":
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=code)
    if code in {"query_policy_mode_unsupported", "query_policy_invalid"}:
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=code)
    if code in {
        "query_policy_projection_incomplete",
        "configuration_projection_unstable",
        "configuration_projection_invalid",
        "configuration_recovery_required",
        "configuration_mutation_cleanup_failed",
    }:
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=code,
        )
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=code)


def _map_configuration_service_error(
    exc: WorkspaceKnowledgeConfigurationServiceError,
) -> HTTPException:
    if exc.error_code in {
        "configuration_projection_unstable",
        "configuration_projection_invalid",
    }:
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=exc.error_code,
        )
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=exc.error_code,
    )


def _query_policy_response(
    *,
    workspace_id: str,
    policy: WorkspaceQueryPolicy,
    configuration_revision: int,
) -> WorkspaceQueryPolicyResponseV1:
    return WorkspaceQueryPolicyResponseV1(
        workspace_id=workspace_id,
        mode=policy.mode.value,
        allowed_connection_refs=policy.allowed_connection_refs,
        allowed_capability_ids=policy.allowed_capability_ids,
        max_live_calls=policy.max_live_calls,
        max_total_duration_ms=policy.max_total_duration_ms,
        max_result_items=policy.max_result_items,
        max_result_bytes=policy.max_result_bytes,
        live_result_retention=policy.live_result_retention.value,
        effective_revision=policy.effective_revision,
        configuration_revision=configuration_revision,
        updated_at=policy.updated_at,
    )


def _query_policy_projection(policy: WorkspaceQueryPolicy) -> QueryPolicyProjectionV1:
    return QueryPolicyProjectionV1(
        mode=policy.mode.value,
        allowed_connection_refs=policy.allowed_connection_refs,
        allowed_capability_ids=policy.allowed_capability_ids,
        max_live_calls=policy.max_live_calls,
        max_total_duration_ms=policy.max_total_duration_ms,
        max_result_items=policy.max_result_items,
        max_result_bytes=policy.max_result_bytes,
        live_result_retention=policy.live_result_retention.value,
        effective_revision=policy.effective_revision,
        updated_at=policy.updated_at,
    )


def _configuration_response(
    configuration: WorkspaceKnowledgeConfigurationV1,
) -> WorkspaceKnowledgeConfigurationResponseV1:
    return WorkspaceKnowledgeConfigurationResponseV1(
        tenant_id=configuration.tenant_id,
        workspace_id=configuration.workspace_id,
        configuration_revision=configuration.configuration_revision,
        connection_attachments=tuple(
            ConnectionAttachmentProjectionV1(
                attachment_id=item.attachment_id,
                connection_ref=item.connection_ref,
                safe_display_label=item.safe_display_label,
                status=item.status.value,
                effective_revision=item.effective_revision,
                created_at=item.created_at,
                updated_at=item.updated_at,
            )
            for item in configuration.connection_attachments
        ),
        indexed_sources=tuple(
            IndexedSourceBindingProjectionV1(
                indexed_source_binding_id=item.indexed_source_binding_id,
                knowledge_source_binding_ref=item.knowledge_source_binding_ref,
                source_id=item.source_id,
                sync_mode=item.sync_mode.value,
                status=item.status.value,
                audience_eligibility=item.audience_eligibility.value,
                effective_revision=item.effective_revision,
                cached_safe_display_label=item.cached_safe_display_label,
                created_at=item.created_at,
                updated_at=item.updated_at,
            )
            for item in configuration.indexed_sources
        ),
        live_access_bindings=tuple(
            LiveAccessBindingProjectionV1(
                live_access_binding_id=item.live_access_binding_id,
                connection_ref=item.connection_ref,
                remote_resource_id=item.remote_resource_id,
                allowed_capability_ids=item.allowed_capability_ids,
                derived_provider_id=item.derived_provider_id,
                derived_integration_kind=item.derived_integration_kind.value,
                derived_resource_type=item.derived_resource_type,
                derived_safe_display_label=item.derived_safe_display_label,
                status=item.status.value,
                audience_eligibility=item.audience_eligibility.value,
                effective_revision=item.effective_revision,
                created_at=item.created_at,
                updated_at=item.updated_at,
            )
            for item in configuration.live_access_bindings
        ),
        query_policy=(
            _query_policy_projection(configuration.query_policy)
            if configuration.query_policy is not None
            else None
        ),
        updated_at=configuration.updated_at,
    )


def mount_knowledge_query_policy_routes(
    app: FastAPI,
    *,
    query_policy_service: WorkspaceQueryPolicyService,
    configuration_service: WorkspaceKnowledgeConfigurationService,
    prefix: str = "/v1/local_workspace",
) -> None:
    router = APIRouter(prefix=prefix, tags=["knowledge-query-policy"])

    @router.put(
        "/workspaces/{workspace_id}/query-policy",
        response_model=WorkspaceQueryPolicyResponseV1,
    )
    async def update_query_policy(
        request: Request,
        workspace_id: str,
        body: UpdateQueryPolicyRequestV1,
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
        mode = _parse_query_policy_mode(body.mode)
        live_result_retention = _parse_live_result_retention(body.live_result_retention)
        try:
            result = query_policy_service.update_query_policy(
                UpdateWorkspaceQueryPolicyCommand(
                    tenant_id=tenant_id,
                    workspace_id=workspace_id,
                    mode=mode,
                    allowed_connection_refs=body.allowed_connection_refs,
                    allowed_capability_ids=body.allowed_capability_ids,
                    max_live_calls=body.max_live_calls,
                    max_total_duration_ms=body.max_total_duration_ms,
                    max_result_items=body.max_result_items,
                    max_result_bytes=body.max_result_bytes,
                    live_result_retention=live_result_retention,
                    expected_revision=expected_revision,
                    idempotency_key_hash=idem_hash,
                )
            )
        except WorkspaceQueryPolicyError as exc:
            raise _map_query_policy_error(exc) from exc
        except WorkspaceKnowledgeConfigurationMutationError as exc:
            raise map_knowledge_configuration_mutation_error(exc) from exc

        payload = _query_policy_response(
            workspace_id=workspace_id,
            policy=result.policy,
            configuration_revision=result.configuration_revision,
        )
        return Response(
            content=payload.model_dump_json(),
            media_type="application/json",
            status_code=status.HTTP_200_OK,
        )

    @router.get(
        "/workspaces/{workspace_id}/knowledge-configuration",
        response_model=WorkspaceKnowledgeConfigurationResponseV1,
    )
    async def get_knowledge_configuration(
        request: Request,
        workspace_id: str,
        x_tenant_id: str | None = Header(default=None, alias="X-Tenant-Id"),
    ) -> Response:
        tenant_id = resolve_knowledge_configuration_tenant_id(
            request,
            header_tenant_id=x_tenant_id,
        )
        try:
            configuration = configuration_service.get_configuration(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
            )
        except WorkspaceKnowledgeConfigurationServiceError as exc:
            raise _map_configuration_service_error(exc) from exc
        except (ValidationError, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="configuration_projection_invalid",
            ) from exc
        if configuration is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="workspace_not_found",
            )
        try:
            payload = _configuration_response(configuration)
        except (ValidationError, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="configuration_projection_invalid",
            ) from exc
        return Response(
            content=payload.model_dump_json(),
            media_type="application/json",
            status_code=status.HTTP_200_OK,
        )

    app.include_router(router)
