# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Identity provider integration contract (Phase M.6 P6)."""

from __future__ import annotations

from typing import Optional, Protocol, Sequence, runtime_checkable

from pydantic import BaseModel, Field


class IdentityUser(BaseModel):
    """Normalized authenticated user profile."""

    user_id: str
    email: str = ""
    name: str = ""
    tenant_id: str = ""
    metadata: dict[str, str] = Field(default_factory=dict)


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
