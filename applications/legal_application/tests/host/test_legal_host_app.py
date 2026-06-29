# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from intergrax.fastapi_core.config import ApiEnvironment
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


@pytest.fixture
def dev_settings() -> LegalBackendSettings:
    return LegalBackendSettings(
        environment=ApiEnvironment.DEV,
        legal_product_profile="strict_legal",
        legal_llm_provider="ollama",
        legal_default_agent_id="legal-default",
        route_prefix="/v1/legal",
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


@pytest.mark.gate
def test_legal_backend_chat_with_unified_task_runner(
    dev_settings: LegalBackendSettings,
    product_harness_api_key: str,
    harness_auth_headers: dict[str, str],
) -> None:
    app = create_legal_backend_app(settings=dev_settings)
    client = TestClient(app, headers=harness_auth_headers)
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
