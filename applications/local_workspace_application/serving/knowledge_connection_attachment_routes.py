# © Artur Czarnecki. All rights reserved.

"""HTTP routes for workspace connection attachments."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime

from fastapi import APIRouter, FastAPI, Header, HTTPException, Request, Response, status
from pydantic import BaseModel, ConfigDict, Field

from intergrax.fastapi_core.context import get_request_context
from local_workspace_application.workspaces.knowledge_configuration_mutation_engine import (
    WorkspaceKnowledgeConfigurationMutationError,
    WorkspaceKnowledgeMutationExecutionDispositionV1,
)
from local_workspace_application.workspaces.knowledge_connection_attachment_service import (
    AttachWorkspaceConnectionCommand,
    WorkspaceConnectionAttachmentError,
    WorkspaceConnectionAttachmentService,
)
from local_workspace_application.workspaces.knowledge_connection_detachment_service import (
    DetachWorkspaceConnectionCommand,
    WorkspaceConnectionDetachmentService,
)

_IF_MATCH_RE = re.compile(r"^WKC/(\d+)$")
_IDEMPOTENCY_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")


class AttachConnectionRequestV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    safe_display_label: str | None = Field(
        default=None,
        min_length=1,
        max_length=256,
    )


class WorkspaceConnectionAttachmentResponseV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    attachment_id: str
    workspace_id: str
    connection_ref: str
    safe_display_label: str
    status: str
    effective_revision: int
    configuration_revision: int
    created_at: datetime
    updated_at: datetime


def _resolve_tenant_id(request: Request, *, header_tenant_id: str | None) -> str:
    try:
        context = get_request_context(request)
    except RuntimeError:
        context = None
    if context is not None and context.tenant_id:
        return str(context.tenant_id)
    if header_tenant_id and header_tenant_id.strip():
        return header_tenant_id.strip()
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="tenant_id_required",
    )


def _parse_if_match(value: str | None) -> int:
    if value is None or not value.strip():
        raise HTTPException(
            status_code=status.HTTP_428_PRECONDITION_REQUIRED,
            detail="knowledge_configuration_if_match_required",
        )
    match = _IF_MATCH_RE.fullmatch(value.strip())
    if match is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="knowledge_configuration_if_match_invalid",
        )
    return int(match.group(1))


def _require_idempotency_key(value: str | None) -> str:
    if value is None or not value.strip():
        raise HTTPException(
            status_code=status.HTTP_428_PRECONDITION_REQUIRED,
            detail="knowledge_configuration_idempotency_key_required",
        )
    normalized = value.strip()
    if len(normalized) < 1 or len(normalized) > 256:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="knowledge_configuration_idempotency_key_invalid",
        )
    if _IDEMPOTENCY_CONTROL_RE.search(normalized) is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="knowledge_configuration_idempotency_key_invalid",
        )
    return normalized


def _idempotency_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _map_attachment_error(exc: WorkspaceConnectionAttachmentError) -> HTTPException:
    code = exc.error_code
    if code == "workspace_not_found":
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=code)
    if code == "connection_not_found":
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=code)
    if code == "connection_attachment_not_found":
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=code)
    if code in {"connection_unavailable", "connection_attachment_projection_incomplete"}:
        return HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=code)
    if code == "connection_detach_dependency_resolution_failed":
        return HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=code)
    if code == "safe_display_label_invalid":
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=code)
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=code)


def _map_mutation_error(exc: WorkspaceKnowledgeConfigurationMutationError) -> HTTPException:
    code = exc.error_code
    if code in {
        "configuration_revision_conflict",
        "configuration_idempotency_conflict",
    }:
        return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=code)
    if code in {
        "configuration_recovery_required",
        "configuration_conditional_store_required",
        "configuration_mutation_cleanup_failed",
    }:
        return HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=code)
    if code == "workspace_not_found":
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=code)
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=code)


def mount_knowledge_connection_attachment_routes(
    app: FastAPI,
    *,
    attachment_service: WorkspaceConnectionAttachmentService,
    detachment_service: WorkspaceConnectionDetachmentService | None = None,
    prefix: str = "/v1/local_workspace",
) -> None:
    router = APIRouter(prefix=prefix, tags=["connection-attachment"])

    @router.put(
        "/workspaces/{workspace_id}/connections/{connection_ref}",
        response_model=WorkspaceConnectionAttachmentResponseV1,
    )
    async def attach_connection(
        request: Request,
        workspace_id: str,
        connection_ref: str,
        body: AttachConnectionRequestV1 | None = None,
        x_tenant_id: str | None = Header(default=None, alias="X-Tenant-Id"),
        if_match: str | None = Header(default=None, alias="If-Match"),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> Response:
        tenant_id = _resolve_tenant_id(request, header_tenant_id=x_tenant_id)
        expected_revision = _parse_if_match(if_match)
        idem_hash = _idempotency_hash(_require_idempotency_key(idempotency_key))
        payload = body or AttachConnectionRequestV1()
        try:
            result = attachment_service.attach_connection(
                AttachWorkspaceConnectionCommand(
                    tenant_id=tenant_id,
                    workspace_id=workspace_id,
                    connection_ref=connection_ref,
                    expected_revision=expected_revision,
                    idempotency_key_hash=idem_hash,
                    requested_safe_display_label=payload.safe_display_label,
                )
            )
        except WorkspaceConnectionAttachmentError as exc:
            raise _map_attachment_error(exc) from exc
        except WorkspaceKnowledgeConfigurationMutationError as exc:
            raise _map_mutation_error(exc) from exc

        disposition = result.disposition
        status_code = status.HTTP_201_CREATED
        if disposition in {
            WorkspaceKnowledgeMutationExecutionDispositionV1.EXISTING_RESULT,
            WorkspaceKnowledgeMutationExecutionDispositionV1.COMMITTED_REPLAY,
        }:
            status_code = status.HTTP_200_OK

        attachment = result.attachment
        response_payload = WorkspaceConnectionAttachmentResponseV1(
            attachment_id=attachment.attachment_id,
            workspace_id=attachment.workspace_id,
            connection_ref=attachment.connection_ref,
            safe_display_label=attachment.safe_display_label,
            status=attachment.status.value,
            effective_revision=attachment.effective_revision,
            configuration_revision=result.configuration_revision,
            created_at=attachment.created_at,
            updated_at=attachment.updated_at,
        )
        return Response(
            content=response_payload.model_dump_json(),
            media_type="application/json",
            status_code=status_code,
        )

    if detachment_service is not None:

        @router.delete(
            "/workspaces/{workspace_id}/connections/{connection_ref}",
            response_model=WorkspaceConnectionAttachmentResponseV1,
        )
        async def detach_connection(
            request: Request,
            workspace_id: str,
            connection_ref: str,
            x_tenant_id: str | None = Header(default=None, alias="X-Tenant-Id"),
            if_match: str | None = Header(default=None, alias="If-Match"),
            idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        ) -> Response:
            tenant_id = _resolve_tenant_id(request, header_tenant_id=x_tenant_id)
            expected_revision = _parse_if_match(if_match)
            idem_hash = _idempotency_hash(_require_idempotency_key(idempotency_key))
            try:
                result = detachment_service.detach_connection(
                    DetachWorkspaceConnectionCommand(
                        tenant_id=tenant_id,
                        workspace_id=workspace_id,
                        connection_ref=connection_ref,
                        expected_revision=expected_revision,
                        idempotency_key_hash=idem_hash,
                    )
                )
            except WorkspaceConnectionAttachmentError as exc:
                raise _map_attachment_error(exc) from exc
            except WorkspaceKnowledgeConfigurationMutationError as exc:
                raise _map_mutation_error(exc) from exc

            attachment = result.attachment
            response_payload = WorkspaceConnectionAttachmentResponseV1(
                attachment_id=attachment.attachment_id,
                workspace_id=attachment.workspace_id,
                connection_ref=attachment.connection_ref,
                safe_display_label=attachment.safe_display_label,
                status=attachment.status.value,
                effective_revision=attachment.effective_revision,
                configuration_revision=result.configuration_revision,
                created_at=attachment.created_at,
                updated_at=attachment.updated_at,
            )
            return Response(
                content=response_payload.model_dump_json(),
                media_type="application/json",
                status_code=status.HTTP_200_OK,
            )

    app.include_router(router)
