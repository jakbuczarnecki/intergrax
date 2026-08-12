# © Artur Czarnecki. All rights reserved.

"""HTTP routes for tenant connection product orchestration (PRODUCT-5B)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping

from fastapi import APIRouter, FastAPI, Header, HTTPException, Query, Request, status
from pydantic import BaseModel, ConfigDict, Field

from intergrax.fastapi_core.context import get_request_context
from intergrax.runtime.vendor_knowledge.tenant_connections import TenantConnectionAdministrativeStatus
from local_workspace_application.workspaces.tenant_connection_product_errors import (
    TenantConnectionProductError,
)
from local_workspace_application.workspaces.tenant_connection_product_orchestration import (
    TenantConnectionProductOrchestrationFactory,
)


class SafeTenantConnectionResponseV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    connection_ref: str
    tenant_id: str
    provider_id: str
    integration_kind: str
    safe_display_name: str
    administrative_status: str
    configuration_version: int
    connected_principal_ref: str | None
    created_at: datetime
    updated_at: datetime


class ConnectionProviderResponseV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider_id: str
    integration_kind: str
    auth_mode: str
    safe_display_name: str
    supported_scopes_summary: str
    qualification: str


class BeginAuthorizationRequestV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider_id: str = Field(min_length=1, max_length=64)
    redirect_uri: str | None = Field(default=None, max_length=512)
    safe_display_name: str | None = Field(default=None, max_length=256)
    connection_ref: str | None = Field(default=None, max_length=128)


class BeginAuthorizationResponseV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    authorization_transaction_ref: str
    authorization_url: str | None
    expires_at: datetime
    required_user_action: str
    manual_instructions: str | None = None


class CompleteAuthorizationRequestV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    authorization_transaction_ref: str | None = Field(default=None, max_length=128)
    authorization_code: str | None = Field(default=None, max_length=4096)
    state: str | None = Field(default=None, max_length=512)
    credential_payload: Mapping[str, Any] | None = None


class CompleteAuthorizationResponseV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    connection: SafeTenantConnectionResponseV1
    disposition: str


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


def _map_product_error(exc: TenantConnectionProductError) -> HTTPException:
    code = exc.error_code
    if code in {
        "connection_provider_unsupported",
        "connection_provider_misconfigured",
        "authorization_transaction_not_found",
        "connection_not_found",
    }:
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=code)
    if code in {
        "authorization_transaction_expired",
        "authorization_exchange_outcome_unknown",
        "credential_binding_invalid",
        "connection_version_conflict",
        "connection_runtime_unavailable",
    }:
        return HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=code)
    if code in {"tenant_mismatch", "authorization_state_invalid", "authorization_callback_replay"}:
        return HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=code)
    if code == "connection_already_exists":
        return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=code)
    if code in {"authorization_redirect_not_allowed", "authorization_already_in_progress"}:
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=code)
    if code in {"connection_revoked", "connection_not_active"}:
        return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=code)
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=code)


def _to_safe_response(connection: object) -> SafeTenantConnectionResponseV1:
    return SafeTenantConnectionResponseV1(
        connection_ref=connection.connection_ref,
        tenant_id=connection.tenant_id,
        provider_id=connection.provider_id,
        integration_kind=connection.integration_kind.value,
        safe_display_name=connection.safe_display_name,
        administrative_status=connection.administrative_status.value,
        configuration_version=connection.configuration_version,
        connected_principal_ref=connection.connected_principal_ref,
        created_at=connection.created_at,
        updated_at=connection.updated_at,
    )


def mount_tenant_connection_routes(
    app: FastAPI,
    *,
    orchestration_factory: TenantConnectionProductOrchestrationFactory,
    prefix: str = "/v1/local_workspace",
) -> None:
    router = APIRouter(prefix=f"{prefix}/knowledge", tags=["knowledge-connections"])

    @router.get("/connection-providers")
    def list_connection_providers(
        request: Request,
        tenant_id: str | None = Header(default=None, alias="X-Tenant-Id"),
    ) -> list[ConnectionProviderResponseV1]:
        resolved_tenant = _resolve_tenant_id(request, header_tenant_id=tenant_id)
        service = orchestration_factory.for_tenant(resolved_tenant)
        return [
            ConnectionProviderResponseV1(
                provider_id=str(item["provider_id"]),
                integration_kind=str(item["integration_kind"]),
                auth_mode=str(item["auth_mode"]),
                safe_display_name=str(item["safe_display_name"]),
                supported_scopes_summary=str(item["supported_scopes_summary"]),
                qualification=str(item.get("qualification", "qualified")),
            )
            for item in service.list_supported_connection_providers()
        ]

    @router.post("/connections/authorize/begin")
    def begin_authorization(
        request: Request,
        body: BeginAuthorizationRequestV1,
        tenant_id: str | None = Header(default=None, alias="X-Tenant-Id"),
    ) -> BeginAuthorizationResponseV1:
        resolved_tenant = _resolve_tenant_id(request, header_tenant_id=tenant_id)
        service = orchestration_factory.for_tenant(resolved_tenant)
        try:
            result = service.begin_connection_authorization(
                provider_id=body.provider_id,
                redirect_uri=body.redirect_uri,
                safe_display_name=body.safe_display_name,
                connection_ref=body.connection_ref,
            )
        except TenantConnectionProductError as exc:
            raise _map_product_error(exc) from exc
        except ValueError as exc:
            if str(exc) == "connection_provider_misconfigured":
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="connection_provider_misconfigured",
                ) from exc
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        return BeginAuthorizationResponseV1(
            authorization_transaction_ref=result.authorization_transaction_ref,
            authorization_url=result.authorization_url,
            expires_at=result.expires_at,
            required_user_action=result.required_user_action,
            manual_instructions=result.manual_instructions,
        )

    @router.post("/connections/authorize/complete")
    def complete_authorization(
        request: Request,
        body: CompleteAuthorizationRequestV1,
        tenant_id: str | None = Header(default=None, alias="X-Tenant-Id"),
    ) -> CompleteAuthorizationResponseV1:
        resolved_tenant = _resolve_tenant_id(request, header_tenant_id=tenant_id)
        service = orchestration_factory.for_tenant(resolved_tenant)
        try:
            result = service.complete_connection_authorization(
                authorization_transaction_ref=body.authorization_transaction_ref,
                authorization_code=body.authorization_code,
                state=body.state,
                credential_payload=body.credential_payload,
            )
        except TenantConnectionProductError as exc:
            raise _map_product_error(exc) from exc
        return CompleteAuthorizationResponseV1(
            connection=_to_safe_response(result.connection),
            disposition=result.disposition,
        )

    @router.get("/connections")
    def list_connections(
        request: Request,
        tenant_id: str | None = Header(default=None, alias="X-Tenant-Id"),
        administrative_status: str | None = Query(default=None),
        limit: int = Query(default=100, ge=1, le=1000),
    ) -> list[SafeTenantConnectionResponseV1]:
        resolved_tenant = _resolve_tenant_id(request, header_tenant_id=tenant_id)
        service = orchestration_factory.for_tenant(resolved_tenant)
        status_filter: TenantConnectionAdministrativeStatus | None = None
        if administrative_status:
            status_filter = TenantConnectionAdministrativeStatus(administrative_status)
        return [
            _to_safe_response(connection)
            for connection in service.list_connections(
                administrative_status=status_filter,
                limit=limit,
            )
        ]

    @router.get("/connections/{connection_ref}")
    def get_connection(
        request: Request,
        connection_ref: str,
        tenant_id: str | None = Header(default=None, alias="X-Tenant-Id"),
    ) -> SafeTenantConnectionResponseV1:
        resolved_tenant = _resolve_tenant_id(request, header_tenant_id=tenant_id)
        service = orchestration_factory.for_tenant(resolved_tenant)
        try:
            connection = service.get_connection(connection_ref)
        except TenantConnectionProductError as exc:
            raise _map_product_error(exc) from exc
        return _to_safe_response(connection)

    @router.post("/connections/{connection_ref}/reconnect")
    def reconnect_connection(
        request: Request,
        connection_ref: str,
        body: BeginAuthorizationRequestV1,
        tenant_id: str | None = Header(default=None, alias="X-Tenant-Id"),
    ) -> BeginAuthorizationResponseV1:
        resolved_tenant = _resolve_tenant_id(request, header_tenant_id=tenant_id)
        service = orchestration_factory.for_tenant(resolved_tenant)
        redirect_uri = body.redirect_uri or ""
        try:
            result = service.reconnect_connection(
                connection_ref=connection_ref,
                redirect_uri=redirect_uri,
            )
        except TenantConnectionProductError as exc:
            raise _map_product_error(exc) from exc
        return BeginAuthorizationResponseV1(
            authorization_transaction_ref=result.authorization_transaction_ref,
            authorization_url=result.authorization_url,
            expires_at=result.expires_at,
            required_user_action=result.required_user_action,
            manual_instructions=result.manual_instructions,
        )

    @router.post("/connections/{connection_ref}/revoke")
    def revoke_connection(
        request: Request,
        connection_ref: str,
        tenant_id: str | None = Header(default=None, alias="X-Tenant-Id"),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> SafeTenantConnectionResponseV1:
        resolved_tenant = _resolve_tenant_id(request, header_tenant_id=tenant_id)
        service = orchestration_factory.for_tenant(resolved_tenant)
        try:
            connection = service.revoke_connection(
                connection_ref=connection_ref,
                idempotency_key=idempotency_key,
            )
        except TenantConnectionProductError as exc:
            raise _map_product_error(exc) from exc
        return _to_safe_response(connection)

    callback_router = APIRouter(tags=["knowledge-connections-oauth"])

    @callback_router.get("/oauth/callback/{provider_id}")
    def oauth_callback(
        provider_id: str,
        code: str | None = Query(default=None),
        state: str | None = Query(default=None),
        error: str | None = Query(default=None),
    ) -> CompleteAuthorizationResponseV1:
        if error:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=error)
        if not code or not state:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="authorization_state_invalid",
            )
        try:
            result = orchestration_factory.complete_oauth_callback(
                provider_id=provider_id,
                authorization_code=code,
                state=state,
            )
        except TenantConnectionProductError as exc:
            raise _map_product_error(exc) from exc
        return CompleteAuthorizationResponseV1(
            connection=_to_safe_response(result.connection),
            disposition=result.disposition,
        )

    app.include_router(router)
    app.include_router(callback_router)


__all__ = ["mount_tenant_connection_routes"]
