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
    resolve_harness_authenticated_principal_for_state,
)
from intergrax.applications._shared.harness_principal import (
    HarnessAuthenticatedPrincipal,
    harness_principal_to_request_identity,
    reject_identity_assertion_conflicts,
)
from intergrax.contracts.agent_run import RequestIdentity

if TYPE_CHECKING:
    from fastmcp import FastMCP
    from starlette.requests import Request

_MCP_HARNESS_AUTH: dict[int, HarnessAuthState | None] = {}


def register_nexus_mcp_harness_auth(
    mcp: FastMCP,
    harness_auth: HarnessAuthState | None,
) -> None:
    """Associate harness auth configuration with a Nexus MCP server instance."""
    _MCP_HARNESS_AUTH[id(mcp)] = harness_auth


def get_nexus_mcp_harness_auth(mcp: FastMCP) -> HarnessAuthState | None:
    return _MCP_HARNESS_AUTH.get(id(mcp))


def clear_nexus_mcp_harness_auth(mcp: FastMCP) -> None:
    _MCP_HARNESS_AUTH.pop(id(mcp), None)


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


def resolve_mcp_run_agent_identity(
    *,
    harness_auth: HarnessAuthState | None,
    asserted_tenant_id: str | None,
    asserted_user_id: str | None,
    default_tenant_id: str,
    default_user_id: str,
) -> McpRunAgentIdentityResolution:
    """
    Bind MCP ``run_agent`` tenant/user from verified harness principal when configured.

    Client-supplied tenant/user are assertions only; conflicts fail before Execution.
    """
    request = _starlette_request_from_fastmcp()
    try:
        principal = _resolve_verified_principal(request, harness_auth)
    except HTTPException as exc:
        raise _tool_error_from_http(exc) from exc

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

    if _harness_auth_requires_principal(harness_auth) and not _local_dev_harness_bypass(
        harness_auth,
    ):
        if request is None:
            raise ToolError(
                "authenticated harness principal required for MCP run_agent "
                "(missing HTTP request context)",
            )
        raise ToolError("authenticated harness principal required for MCP run_agent")

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
        return ToolError(f"unauthorized: {message}")
    if exc.status_code == status.HTTP_403_FORBIDDEN:
        return ToolError(f"forbidden: {message}")
    if exc.status_code == status.HTTP_400_BAD_REQUEST:
        return ToolError(f"identity_conflict: {message}")
    return ToolError(message)


__all__ = [
    "McpRunAgentIdentityResolution",
    "clear_nexus_mcp_harness_auth",
    "get_nexus_mcp_harness_auth",
    "register_nexus_mcp_harness_auth",
    "resolve_mcp_run_agent_identity",
]
