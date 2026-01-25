from fastapi import APIRouter, Depends
from fastapi.testclient import TestClient

from intergrax.fastapi_core.app_factory import create_app
from intergrax.fastapi_core.auth.dependency import get_auth_context
from intergrax.fastapi_core.config import ApiConfig
from intergrax.fastapi_core.auth.api_key import ApiKeyConfig, ApiKeyIdentity
from intergrax.fastapi_core.protocol import ApiHeaders


def test_auth_context_available_in_request() -> None:
    identity = ApiKeyIdentity(
        tenant_id="tenant-x",
        user_id="user-x",
        scopes=("read",),
    )

    config = ApiConfig(
        api_key_config=ApiKeyConfig(
            keys={"k1": identity}
        )
    )

    app = create_app(config)

    router = APIRouter()

    @router.get("/whoami")
    def whoami(ctx=Depends(get_auth_context)):
        return {
            "tenant": ctx.tenant_id,
            "user": ctx.user_id,
        }

    app.include_router(router)

    client = TestClient(app)

    response = client.get(
        "/whoami",
        headers={ApiHeaders.API_KEY: "k1"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "tenant": "tenant-x",
        "user": "user-x",
    }
