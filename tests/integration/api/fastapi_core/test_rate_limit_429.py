# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

# FILE: tests/integration/api/fastapi_core/rate_limit/test_rate_limit_429.py

from __future__ import annotations

from fastapi import Depends
from fastapi.testclient import TestClient

from intergrax.fastapi_core.app_factory import create_app
from intergrax.fastapi_core.config import ApiConfig
from intergrax.fastapi_core.context import RequestContext
from intergrax.fastapi_core.errors.error_types import ApiErrorType
from intergrax.fastapi_core.rate_limit.dependency import RateLimitRequired
from intergrax.fastapi_core.rate_limit.policy import RateLimitPolicy



class DenyAllPolicy(RateLimitPolicy):
    def allow(self, context: RequestContext) -> bool:
        return False


def test_rate_limit_returns_429_when_exceeded() -> None:
    app = create_app(
        ApiConfig(
            rate_limit_policy=DenyAllPolicy(),
        )
    )

    @app.get("/limited")
    def limited(
        _: None = Depends(RateLimitRequired()),
    ) -> dict[str, str]:
        return {"ok": "true"}

    client = TestClient(app, raise_server_exceptions=False)

    response = client.get("/limited")

    assert response.status_code == 429
    body = response.json()

    assert body["error_type"] == ApiErrorType.RATE_LIMITED.value
