from fastapi import Request
from starlette.requests import Request as StarletteRequest

from intergrax.fastapi_core.auth.context import AuthContext
from intergrax.fastapi_core.auth.provider import AuthProvider
from intergrax.fastapi_core.auth.providers.compose.provider import CompositeAuthProvider


class DummyProvider(AuthProvider):
    def __init__(self, ctx: AuthContext | None) -> None:
        self._ctx = ctx

    def authenticate(self, request: Request):
        return self._ctx


def _request() -> Request:
    return StarletteRequest(
        {"type": "http", "method": "GET", "path": "/"}
    )


def test_composite_uses_first_non_none() -> None:
    ctx1 = AuthContext(
        is_authenticated=True,
        tenant_id="t1",
        user_id=None,
        scopes=(),
    )

    providers = [
        DummyProvider(None),
        DummyProvider(ctx1),
        DummyProvider(None),
    ]

    composite = CompositeAuthProvider(providers)

    result = composite.authenticate(_request())

    assert result is ctx1
