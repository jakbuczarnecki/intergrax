# © Artur Czarnecki. All rights reserved.

"""HTTP caller authorization against strict serving session tenant binding (P3-A2)."""

from __future__ import annotations

from fastapi import HTTPException, status

from intergrax.applications._shared.strict_production_serving_execution_tenant_authority import (
    StrictProductionServingExecutionTenantBinding,
)
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.fastapi_core.auth.context import AuthContext


def authorize_strict_production_serving_http_caller(
    auth: AuthContext,
    *,
    binding: StrictProductionServingExecutionTenantBinding,
) -> None:
    """Fail closed when authenticated caller is not authorized for this serving session."""
    if not auth.is_authenticated:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )
    caller_tenant = (auth.tenant_id or "").strip()
    if not caller_tenant:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="caller_tenant_not_authorized_for_serving_session",
        )
    if caller_tenant != binding.authoritative_execution_tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="caller_tenant_not_authorized_for_serving_session",
        )


def request_identity_from_strict_production_http_auth(
    auth: AuthContext,
) -> RequestIdentity:
    """Map fastapi_core AuthContext to execution RequestIdentity (no body upgrade)."""
    if not auth.is_authenticated:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )
    tenant_id = (auth.tenant_id or "").strip()
    user_id = (auth.user_id or "").strip()
    if not tenant_id or not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                "tenant_id and user_id must be set on authenticated RequestContext "
                "(configure AuthProvider / API key or JWT on the app)."
            ),
        )
    auth_subject = user_id
    return RequestIdentity(
        tenant_id=tenant_id,
        user_id=user_id,
        principal_type=PrincipalType.USER,
        auth_subject=auth_subject,
    )


__all__ = [
    "authorize_strict_production_serving_http_caller",
    "request_identity_from_strict_production_http_auth",
]
