# FILE: tests/integration/api/fastapi_core/auth/test_scope_required_forbidden.py

from __future__ import annotations

from fastapi import Depends
from fastapi.testclient import TestClient

from intergrax.fastapi_core.app_factory import create_app
from intergrax.fastapi_core.auth.authorization import ScopeRequired
from intergrax.fastapi_core.auth.context import AuthContext
from intergrax.fastapi_core.auth.provider import AuthProvider
from intergrax.fastapi_core.config import ApiConfig
from intergrax.fastapi_core.auth.providers.no_auth import NoAuthProvider
from intergrax.fastapi_core.errors.error_types import ApiErrorType


class AuthenticatedNoScopeProvider(AuthProvider):
    def authenticate(self, request):
        return AuthContext(
            is_authenticated=True,
            tenant_id="t1",
            user_id="u1",
            scopes=(),
        )

class AuthenticatedWithScopeProvider(AuthProvider):
    def authenticate(self, request):
        return AuthContext(
            is_authenticated=True,
            tenant_id="t1",
            user_id="u1",
            scopes=("runs:write",),
        )


def test_scope_required_returns_403_when_scope_missing() -> None:
    app = create_app(
        ApiConfig(
            auth_provider=AuthenticatedNoScopeProvider(),
        )
    )

    @app.get("/protected")
    def protected(
        _: None = Depends(ScopeRequired("runs:write")),
    ) -> dict[str, str]:
        return {"ok": "true"}

    client = TestClient(app, raise_server_exceptions=False)

    response = client.get("/protected")

    assert response.status_code == 403
    body = response.json()

    assert body["error_type"] == ApiErrorType.FORBIDDEN.value


def test_scope_required_returns_401_when_not_authenticated() -> None:
    app = create_app(
        ApiConfig(
            auth_provider=NoAuthProvider(),
        )
    )

    @app.get("/protected")
    def protected(
        _: None = Depends(ScopeRequired("runs:write")),
    ) -> dict[str, str]:
        return {"ok": "true"}

    client = TestClient(app, raise_server_exceptions=False)

    response = client.get("/protected")

    assert response.status_code == 401
    body = response.json()

    assert body["error_type"] == ApiErrorType.UNAUTHORIZED.value


def test_scope_required_allows_request_when_scope_present() -> None:
    app = create_app(
        ApiConfig(
            auth_provider=AuthenticatedWithScopeProvider(),
        )
    )

    @app.get("/protected")
    def protected(
        _: None = Depends(ScopeRequired("runs:write")),
    ) -> dict[str, str]:
        return {"ok": "true"}

    client = TestClient(app, raise_server_exceptions=False)

    response = client.get("/protected")

    assert response.status_code == 200
    assert response.json() == {"ok": "true"}
