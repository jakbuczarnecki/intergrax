# © Artur Czarnecki. All rights reserved.

"""Unit tests for strict production HTTP caller authorization (P3-A2)."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from intergrax.applications._shared.strict_production_serving_execution_tenant_authority import (
    StrictProductionServingExecutionTenantBinding,
)
from intergrax.applications._shared.strict_production_serving_http_caller_authorization import (
    authorize_strict_production_serving_http_caller,
    request_identity_from_strict_production_http_auth,
)
from intergrax.fastapi_core.auth.context import AuthContext

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _binding() -> StrictProductionServingExecutionTenantBinding:
    return StrictProductionServingExecutionTenantBinding(
        authoritative_execution_tenant_id="env-prod",
        application_id="app",
        application_environment_id="env-prod",
        control_plane_environment_tenant_id="tenant-b",
    )


def test_authorize_accepts_matching_caller_tenant() -> None:
    auth = AuthContext(
        is_authenticated=True,
        tenant_id="env-prod",
        user_id="operator",
        scopes=("*",),
    )
    authorize_strict_production_serving_http_caller(auth, binding=_binding())


def test_authorize_rejects_mismatched_caller_tenant() -> None:
    auth = AuthContext(
        is_authenticated=True,
        tenant_id="tenant-a",
        user_id="operator",
        scopes=("*",),
    )
    with pytest.raises(HTTPException) as exc:
        authorize_strict_production_serving_http_caller(auth, binding=_binding())
    assert exc.value.status_code == 403


def test_request_identity_requires_user_and_tenant() -> None:
    identity = request_identity_from_strict_production_http_auth(
        AuthContext(
            is_authenticated=True,
            tenant_id="env-prod",
            user_id="operator",
            scopes=("*",),
        ),
    )
    assert identity.tenant_id == "env-prod"
    assert identity.user_id == "operator"
