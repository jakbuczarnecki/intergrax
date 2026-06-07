# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.integrations.contracts.identity_provider import IdentityTenant, IdentityUser
from intergrax.tools.providers.identity.contracts import (
    IdentityGetUserInput,
    IdentityListTenantsInput,
    IdentityVerifyTokenInput,
)
from intergrax.tools.providers.identity.service import identity_get_user, identity_list_tenants, identity_verify_token
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = pytest.mark.unit


class FakeIdentityProvider:
    def verify_token(self, token: str) -> IdentityUser:
        return IdentityUser(user_id="u-1", email="ops@example.com", name="Ops", tenant_id="t-1")

    def userinfo(self, token: str) -> IdentityUser:
        return IdentityUser(user_id="u-1", email="ops@example.com", name="Ops", tenant_id="t-1")

    def list_tenants(self, *, limit: int = 50) -> list[IdentityTenant]:
        return [IdentityTenant(tenant_id="t-1", name="Primary")]


def test_identity_verify_token_returns_user() -> None:
    ctx = ToolWiringContext(identity_provider=FakeIdentityProvider())
    out = identity_verify_token(ctx, IdentityVerifyTokenInput(token=" bearer-token "))
    assert out.valid is True
    assert out.user.user_id == "u-1"
    assert out.user.email == "ops@example.com"


def test_identity_get_user_returns_profile() -> None:
    ctx = ToolWiringContext(identity_provider=FakeIdentityProvider())
    out = identity_get_user(ctx, IdentityGetUserInput(token="token"))
    assert out.user.tenant_id == "t-1"


def test_identity_list_tenants_returns_directory() -> None:
    ctx = ToolWiringContext(identity_provider=FakeIdentityProvider())
    out = identity_list_tenants(ctx, IdentityListTenantsInput(limit=10))
    assert out.total == 1
    assert out.tenants[0].name == "Primary"


def test_identity_not_configured() -> None:
    with pytest.raises(RuntimeError, match="identity_provider_not_configured"):
        identity_verify_token(ToolWiringContext(), IdentityVerifyTokenInput(token="x"))
