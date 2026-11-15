# © Artur Czarnecki. All rights reserved.

"""Request-scoped harness principal resolution for FastMCP ``run_agent`` (TLX-TL11 TRUST-A)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from fastapi import HTTPException, status
from fastmcp.exceptions import ToolError
from fastmcp.server.context import request_ctx

from intergrax.applications._shared.harness_auth import (
    HarnessAuthState,
    resolve_harness_auth_state_from_app,
    resolve_harness_authenticated_principal_for_state,
)
from intergrax.applications._shared.harness_principal import (
    HarnessAuthenticatedPrincipal,
    harness_principal_to_request_identity,
    reject_identity_assertion_conflicts,
)
from intergrax.applications._shared.mcp_identity_mode import (
    McpIdentityMode,
    coerce_mcp_identity_mode,
    is_loopback_client,
)
from intergrax.applications._shared.mcp_wire_errors import mcp_tool_error
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.mcp_server_definition import McpServerDefinitionErrorCode

if TYPE_CHECKING:
    from starlette.requests import Request


def resolve_harness_auth_for_mcp_request(request: Request | None) -> HarnessAuthState | None:
    """Resolve typed harness auth from the Starlette app serving the MCP HTTP request.

    A request-time resolver installed by coupling wins over the one-shot copy, so identity wired on the
    host after MCP coupling (factory ordering) is still enforced (MRC-1, G-ID-02 / G-ID-09).
    """
    if request is None:
        return None
    resolver = getattr(request.app.state, "harness_auth_resolver", None)
    if callable(resolver):
        live = resolver()
        if live is not None:
            return live
    return resolve_harness_auth_state_from_app(request.app)


@dataclass(frozen=True, slots=True)
class McpRunAgentIdentityResolution:
    tenant_id: str
    user_id: str
    principal: HarnessAuthenticatedPrincipal | None
    request_identity: RequestIdentity | None


def _starlette_request_from_fastmcp() -> Request | None:
    try:
        ctx = request_ctx.get()
    except LookupError:
        return None
    if ctx is None:
        return None
    request = getattr(ctx, "request", None)
    if request is None:
        return None
    return request


def _local_dev_harness_bypass(harness_auth: HarnessAuthState | None) -> bool:
    return (
        harness_auth is not None
        and not harness_auth.require_api_key
        and harness_auth.resolved_api_key is None
        and harness_auth.identity_provider is None
    )


def _harness_auth_requires_principal(harness_auth: HarnessAuthState | None) -> bool:
    if harness_auth is None:
        return False
    if harness_auth.identity_provider is not None:
        return True
    if harness_auth.resolved_api_key is not None:
        return True
    if harness_auth.require_api_key:
        return True
    return False


def _resolve_verified_principal(
    request: Request | None,
    harness_auth: HarnessAuthState | None,
) -> HarnessAuthenticatedPrincipal | None:
    if request is None:
        return None
    return resolve_harness_authenticated_principal_for_state(
        harness_auth=harness_auth,
        x_api_key=request.headers.get("X-Api-Key"),
        authorization=request.headers.get("Authorization"),
    )


def _enforce_local_dev_admission(request: Request | None) -> None:
    """LOCAL_DEV_EXPLICIT is admitted only for in-process calls and loopback clients."""
    if not is_loopback_client(request):
        raise mcp_tool_error(
            McpServerDefinitionErrorCode.MCP_PRINCIPAL_REQUIRED,
            "local development identity mode refuses non-loopback clients",
        )


def resolve_mcp_caller_principal(
    *,
    harness_auth: HarnessAuthState | None,
    identity_mode: McpIdentityMode | str | None = None,
) -> HarnessAuthenticatedPrincipal | None:
    """Verified principal of the current MCP call, or ``None`` only where the mode admits it."""
    mode = coerce_mcp_identity_mode(identity_mode)
    request = _starlette_request_from_fastmcp()
    effective_harness_auth = harness_auth
    if effective_harness_auth is None:
        effective_harness_auth = resolve_harness_auth_for_mcp_request(request)
    try:
        principal = _resolve_verified_principal(request, effective_harness_auth)
    except HTTPException as exc:
        raise _tool_error_from_http(exc) from exc
    if principal is not None:
        return principal
    if mode is McpIdentityMode.HARNESS_BOUND:
        detail = "authenticated harness principal required for MCP calls"
        if request is None:
            detail += " (missing HTTP request context)"
        raise mcp_tool_error(McpServerDefinitionErrorCode.MCP_PRINCIPAL_REQUIRED, detail)
    if _harness_auth_requires_principal(effective_harness_auth) and not _local_dev_harness_bypass(
        effective_harness_auth,
    ):
        raise mcp_tool_error(
            McpServerDefinitionErrorCode.MCP_PRINCIPAL_REQUIRED,
            "authenticated harness principal required for MCP calls",
        )
    _enforce_local_dev_admission(request)
    return None


def require_mcp_caller_principal(
    *,
    identity_mode: McpIdentityMode | str | None = None,
    harness_auth: HarnessAuthState | None = None,
) -> None:
    """Principal gate for discovery and other tools that carry no tenant assertion."""
    resolve_mcp_caller_principal(harness_auth=harness_auth, identity_mode=identity_mode)


def resolve_mcp_run_agent_identity(
    *,
    harness_auth: HarnessAuthState | None,
    asserted_tenant_id: str | None,
    asserted_user_id: str | None,
    default_tenant_id: str,
    default_user_id: str,
    identity_mode: McpIdentityMode | str | None = None,
) -> McpRunAgentIdentityResolution:
    """
    Bind MCP ``run_agent`` tenant/user from the verified harness principal.

    Client-supplied tenant/user are assertions only and can never establish identity in
    ``HARNESS_BOUND`` (default); conflicts fail before Execution. ``LOCAL_DEV_EXPLICIT`` (explicit code
    opt-in) additionally admits assertion-only identity for in-process and loopback callers.
    """
    principal = resolve_mcp_caller_principal(harness_auth=harness_auth, identity_mode=identity_mode)

    if principal is not None:
        identity = harness_principal_to_request_identity(principal)
        try:
            reject_identity_assertion_conflicts(
                canonical=identity,
                asserted_tenant_id=asserted_tenant_id,
                asserted_user_id=asserted_user_id,
            )
        except HTTPException as exc:
            raise _tool_error_from_http(exc) from exc
        user_id = identity.user_id or default_user_id
        return McpRunAgentIdentityResolution(
            tenant_id=identity.tenant_id,
            user_id=user_id,
            principal=principal,
            request_identity=identity,
        )

    tenant_id = (asserted_tenant_id or default_tenant_id).strip()
    user_id = (asserted_user_id or default_user_id).strip()
    if not tenant_id:
        raise ToolError("tenant_id is required for MCP run_agent")
    if not user_id:
        raise ToolError("user_id is required for MCP run_agent")
    return McpRunAgentIdentityResolution(
        tenant_id=tenant_id,
        user_id=user_id,
        principal=None,
        request_identity=None,
    )


def _tool_error_from_http(exc: HTTPException) -> ToolError:
    detail = exc.detail
    if isinstance(detail, str):
        message = detail
    else:
        message = str(detail)
    if exc.status_code == status.HTTP_401_UNAUTHORIZED:
        return mcp_tool_error(McpServerDefinitionErrorCode.MCP_PRINCIPAL_REQUIRED, f"unauthorized: {message}")
    if exc.status_code == status.HTTP_403_FORBIDDEN:
        return mcp_tool_error(McpServerDefinitionErrorCode.MCP_PRINCIPAL_REQUIRED, f"forbidden: {message}")
    if exc.status_code == status.HTTP_400_BAD_REQUEST:
        return mcp_tool_error(
            McpServerDefinitionErrorCode.MCP_TENANT_ASSERTION_CONFLICT,
            f"identity_conflict: {message}",
        )
    return ToolError(message)


__all__ = [
    "McpRunAgentIdentityResolution",
    "require_mcp_caller_principal",
    "resolve_harness_auth_for_mcp_request",
    "resolve_mcp_caller_principal",
    "resolve_mcp_run_agent_identity",
]
