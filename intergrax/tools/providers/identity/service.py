# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.integrations.contracts.identity_provider import IdentityProviderBackend, IdentityTenant, IdentityUser
from intergrax.tools.providers.identity.contracts import (
    IdentityGetUserInput,
    IdentityGetUserOutput,
    IdentityListTenantsInput,
    IdentityListTenantsOutput,
    IdentityTenantOutput,
    IdentityUserOutput,
    IdentityVerifyTokenInput,
    IdentityVerifyTokenOutput,
)
from intergrax.tools.registry.wiring import ToolWiringContext

IDENTITY_VERIFY_TOKEN_TOOL_ID = "identity.verify_token"
IDENTITY_GET_USER_TOOL_ID = "identity.get_user"
IDENTITY_LIST_TENANTS_TOOL_ID = "identity.list_tenants"


def _require_identity(ctx: ToolWiringContext) -> IdentityProviderBackend:
    backend = ctx.identity_provider
    if backend is None:
        raise RuntimeError("identity_provider_not_configured")
    return backend


def _user_output(user: IdentityUser) -> IdentityUserOutput:
    return IdentityUserOutput(
        user_id=user.user_id,
        email=user.email,
        name=user.name,
        tenant_id=user.tenant_id,
        metadata=dict(user.metadata),
    )


def identity_verify_token(ctx: ToolWiringContext, params: IdentityVerifyTokenInput) -> IdentityVerifyTokenOutput:
    user = _require_identity(ctx).verify_token(params.token.strip())
    return IdentityVerifyTokenOutput(valid=True, user=_user_output(user))


def identity_get_user(ctx: ToolWiringContext, params: IdentityGetUserInput) -> IdentityGetUserOutput:
    user = _require_identity(ctx).userinfo(params.token.strip())
    return IdentityGetUserOutput(user=_user_output(user))


def identity_list_tenants(ctx: ToolWiringContext, params: IdentityListTenantsInput) -> IdentityListTenantsOutput:
    tenants = [
        IdentityTenantOutput(
            tenant_id=item.tenant_id,
            name=item.name,
            metadata=dict(item.metadata),
        )
        for item in _require_identity(ctx).list_tenants(limit=params.limit)
    ]
    return IdentityListTenantsOutput(tenants=tenants, total=len(tenants))
