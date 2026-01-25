# FILE: tests/unit/api/fastapi_core/auth/test_scope_required.py

from __future__ import annotations

import pytest

from intergrax.fastapi_core.auth.authorization import ScopeRequired
from intergrax.fastapi_core.auth.context import AuthContext
from intergrax.fastapi_core.context import RequestContext
from intergrax.fastapi_core.errors.auth import MissingScopeError, NotAuthenticatedError


def _ctx(scopes: tuple[str, ...]) -> RequestContext:
    return RequestContext(
        request_id="req-1",
        path="/test",
        method="GET",
        tenant_id="t1",
        user_id="u1",
        auth=AuthContext(
            is_authenticated=True,
            tenant_id="t1",
            user_id="u1",
            scopes=scopes,
        ),
    )


def test_scope_required_allows_when_scope_present() -> None:
    guard = ScopeRequired("runs:write")
    guard(_ctx(("runs:write",)))


def test_scope_required_blocks_when_scope_missing() -> None:
    guard = ScopeRequired("runs:write")

    with pytest.raises(MissingScopeError):
        guard(_ctx(("runs:read",)))


def test_scope_required_blocks_when_not_authenticated() -> None:
    guard = ScopeRequired("runs:write")

    ctx = RequestContext(
        request_id="req-1",
        path="/test",
        method="GET",
        tenant_id=None,
        user_id=None,
        auth=None,
    )

    with pytest.raises(NotAuthenticatedError):
        guard(ctx)
