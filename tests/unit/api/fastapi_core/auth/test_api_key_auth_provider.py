from fastapi import Request
from starlette.requests import Request as StarletteRequest

from intergrax.fastapi_core.auth.api_key import ApiKeyConfig, ApiKeyIdentity
from intergrax.fastapi_core.auth.providers.api_key.provider import ApiKeyAuthProvider
from intergrax.fastapi_core.protocol import ApiHeaders


def _request_with_api_key(api_key: str) -> Request:
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": [
            (ApiHeaders.API_KEY.lower().encode(), api_key.encode()),
        ],
    }
    return Request(scope)


def test_api_key_auth_provider_success() -> None:
    identity = ApiKeyIdentity(
        tenant_id="tenant-1",
        user_id="user-1",
        scopes=("read", "write"),
    )

    config = ApiKeyConfig(
        keys={"valid-key": identity}
    )

    provider = ApiKeyAuthProvider(config)

    ctx = provider.authenticate(_request_with_api_key("valid-key"))

    assert ctx is not None
    assert ctx.is_authenticated is True
    assert ctx.tenant_id == "tenant-1"
    assert ctx.user_id == "user-1"
    assert ctx.scopes == ("read", "write")


def test_api_key_auth_provider_invalid_key() -> None:
    config = ApiKeyConfig(keys={})
    provider = ApiKeyAuthProvider(config)

    ctx = provider.authenticate(_request_with_api_key("invalid"))

    assert ctx is None
