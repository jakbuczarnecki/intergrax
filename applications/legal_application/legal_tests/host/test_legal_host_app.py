# © Artur Czarnecki. All rights reserved.

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
from intergrax.runtime.task.task import TaskResult, TaskState
from legal_application.host.factory import create_legal_backend_app
from legal_application.host.settings import LegalBackendSettings

pytestmark = pytest.mark.unit


def test_settings_prod_requires_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LEGAL_BACKEND_ENV", "prod")
    monkeypatch.delenv("LEGAL_BACKEND_BOOTSTRAP_API_KEY", raising=False)
    monkeypatch.delenv("LEGAL_BACKEND_API_KEYS_JSON", raising=False)
    monkeypatch.delenv("LEGAL_BACKEND_ALLOW_UNAUTHENTICATED", raising=False)
    with pytest.raises(ValueError, match="Production Legal backend requires"):
        LegalBackendSettings.from_env()


def test_settings_default_nexus_loop_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LEGAL_USE_NEXUS_LOOP", raising=False)
    monkeypatch.delenv("LEGAL_USE_LEGACY_AGENT_ENGINE", raising=False)
    settings = LegalBackendSettings.from_env()
    assert settings.use_nexus_loop is True
    assert settings.use_legacy_agent_engine is False


def test_settings_legacy_opt_out(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LEGAL_USE_LEGACY_AGENT_ENGINE", "true")
    settings = LegalBackendSettings.from_env()
    assert settings.use_nexus_loop is False
    assert settings.use_legacy_agent_engine is True


@pytest.fixture
def dev_settings_legacy() -> LegalBackendSettings:
    return LegalBackendSettings(
        environment=ApiEnvironment.DEV,
        legal_product_profile="strict_legal",
        legal_llm_provider="ollama",
        legal_default_agent_id="legal-default",
        legal_route_prefix="/v1/legal",
        identity_source="body_or_context",
        use_nexus_loop=False,
        use_legacy_agent_engine=True,
        cors_allow_origins=frozenset(),
        allowed_hosts=frozenset(),
        openapi_enabled_override=True,
        session_sqlite_path=None,
        api_keys_map={},
    )


@pytest.fixture
def dev_settings_nexus() -> LegalBackendSettings:
    return LegalBackendSettings(
        environment=ApiEnvironment.DEV,
        legal_product_profile="strict_legal",
        legal_llm_provider="ollama",
        legal_default_agent_id="legal-default",
        legal_route_prefix="/v1/legal",
        identity_source="body_or_context",
        use_nexus_loop=True,
        use_legacy_agent_engine=False,
        cors_allow_origins=frozenset(),
        allowed_hosts=frozenset(),
        openapi_enabled_override=True,
        session_sqlite_path=None,
        api_keys_map={},
    )


def test_legal_backend_exposes_health_and_openapi(dev_settings_nexus: LegalBackendSettings) -> None:
    app = create_legal_backend_app(settings=dev_settings_nexus)
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code in {200, 204}, r.text
    o = client.get("/openapi.json")
    assert o.status_code == 200
    assert "Legal" in app.title or "legal" in app.title.lower()


@pytest.mark.gate
def test_legal_backend_chat_with_nexus_loop_default(dev_settings_nexus: LegalBackendSettings) -> None:
    app = create_legal_backend_app(settings=dev_settings_nexus)
    client = TestClient(app)
    task_result = TaskResult(
        task_id="run-nexus-host",
        run_id="run-nexus-host",
        state=TaskState.COMPLETED,
        answer="nexus host ok",
    )
    with patch(
        "intergrax.runtime.task.unified_task_runner.UnifiedTaskRunner.run_runtime_request",
        new_callable=AsyncMock,
        return_value=task_result,
    ) as run_mock:
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
    assert r.json()["answer"] == "nexus host ok"
    run_mock.assert_awaited_once()


def test_legal_backend_chat_with_legacy_engine_opt_out(dev_settings_legacy: LegalBackendSettings) -> None:
    app = create_legal_backend_app(settings=dev_settings_legacy)
    client = TestClient(app)
    answer = RuntimeAnswer(
        answer="legacy host ok",
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
    assert r.json()["answer"] == "legacy host ok"
