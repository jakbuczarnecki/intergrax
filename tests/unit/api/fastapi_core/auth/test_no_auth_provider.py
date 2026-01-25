from fastapi import Request
from starlette.requests import Request as StarletteRequest

from intergrax.fastapi_core.auth.providers.no_auth import NoAuthProvider


def test_no_auth_provider_returns_none() -> None:
    provider = NoAuthProvider()

    scope = {
        "type": "http",
        "headers": [],
        "method": "GET",
        "path": "/",
    }
    request = StarletteRequest(scope)

    ctx = provider.authenticate(request)

    assert ctx is None
