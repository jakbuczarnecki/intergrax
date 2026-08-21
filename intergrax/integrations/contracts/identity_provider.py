# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Identity provider integration contract (Phase M.6 P6)."""

from __future__ import annotations

from typing import Optional, Protocol, Sequence, runtime_checkable

from pydantic import BaseModel, Field

AGENT_PLATFORM_ADMIN_ROLE = "agent_platform_admin"
AGENT_PLATFORM_ADMIN_SCOPE = "agent_platform:admin"


class IdentityUser(BaseModel):
    """Normalized authenticated user profile."""

    user_id: str
    email: str = ""
    name: str = ""
    tenant_id: str = ""
    metadata: dict[str, str] = Field(default_factory=dict)
    roles: tuple[str, ...] = ()
    scopes: tuple[str, ...] = ()


def identity_user_has_agent_platform_admin_authority(user: IdentityUser) -> bool:
    """Return whether the provider-normalized principal carries Agent Platform admin authority."""
    return (
        AGENT_PLATFORM_ADMIN_ROLE in user.roles
        or AGENT_PLATFORM_ADMIN_SCOPE in user.scopes
    )


class IdentityTenant(BaseModel):
    """Organization/tenant row for multi-tenant harness hosts."""

    tenant_id: str
    name: str = ""
    metadata: dict[str, str] = Field(default_factory=dict)


@runtime_checkable
class IdentityProviderBackend(Protocol):
    """OIDC/SSO verification facade for multi-tenant harness hosts."""

    def verify_token(self, token: str) -> IdentityUser:
        """Validate bearer token and return normalized user profile."""

    def userinfo(self, token: str) -> IdentityUser:
        """Fetch user profile for a valid access token."""

    def list_tenants(self, *, limit: int = 50) -> Sequence[IdentityTenant]:
        """Optional directory listing for enterprise SSO integrations."""
