# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from intergrax.fastapi_core.config import ApiEnvironment
from intergrax.runtime.nexus.responses.response_schema import (
    RouteInfo,
    RuntimeAnswer,
    RuntimeStats,
    StopReason,
)
from legal_agent.host.factory import create_legal_backend_app
from legal_agent.host.settings import LegalBackendSettings

pytestmark = pytest.mark.unit


def test_settings_prod_requires_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LEGAL_BACKEND_ENV", "prod")
    monkeypatch.delenv("LEGAL_BACKEND_BOOTSTRAP_API_KEY", raising=False)
    monkeypatch.delenv("LEGAL_BACKEND_API_KEYS_JSON", raising=False)
    monkeypatch.delenv("LEGAL_BACKEND_ALLOW_UNAUTHENTICATED", raising=False)
    with pytest.raises(ValueError, match="Production Legal backend requires"):
        LegalBackendSettings.from_env()


@pytest.fixture
def dev_settings() -> LegalBackendSettings:
    return LegalBackendSettings(
        environment=ApiEnvironment.DEV,
        legal_product_profile="strict_legal",
        legal_llm_provider="ollama",
        legal_default_agent_id="legal-default",
        legal_route_prefix="/v1/legal",
        identity_source="body_or_context",
        cors_allow_origins=frozenset(),
        allowed_hosts=frozenset(),
        openapi_enabled_override=True,
        session_sqlite_path=None,
        api_keys_map={},
    )


def test_legal_backend_exposes_health_and_openapi(dev_settings: LegalBackendSettings) -> None:
    app = create_legal_backend_app(settings=dev_settings)
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code in {200, 204}, r.text
    o = client.get("/openapi.json")
    assert o.status_code == 200
    assert "Legal" in app.title or "legal" in app.title.lower()


def test_legal_backend_chat_with_mocked_engine(dev_settings: LegalBackendSettings) -> None:
    app = create_legal_backend_app(settings=dev_settings)
    client = TestClient(app)
    answer = RuntimeAnswer(
        answer="host ok",
        stop_reason=StopReason.COMPLETED,
        run_id="run-host",
        route=RouteInfo(strategy="legal_agent"),
        stats=RuntimeStats(),
    )
    with patch(
        "intergrax.agents.agent_engine.AgentEngine.run",
        new_callable=AsyncMock,
        return_value=answer,
    ):
        r = client.post(
            "/v1/legal/chat",
            json={
                "message": "ping",
                "session_id": "s-host",
                "tenant_id": "ten-host",
                "user_id": "user-host",
            },
        )
    assert r.status_code == 200, r.text
    assert r.json()["answer"] == "host ok"
