# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from intergrax.runtime.task.task import TaskResult, TaskState
from research_application.host.factory import create_research_backend_app
from research_application.host.settings import ResearchBackendSettings

pytestmark = [pytest.mark.integration, pytest.mark.gate]


@pytest.fixture
def research_client() -> TestClient:
    settings = ResearchBackendSettings(use_nexus_loop=True)
    app = create_research_backend_app(settings=settings)
    return TestClient(app)


def test_research_settings_default_nexus_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RESEARCH_USE_NEXUS_LOOP", raising=False)
    monkeypatch.delenv("RESEARCH_USE_LEGACY_AGENT_ENGINE", raising=False)
    settings = ResearchBackendSettings.from_env()
    assert settings.use_nexus_loop is True


def test_research_run_uses_unified_task_runner(research_client: TestClient) -> None:
    task_result = TaskResult(
        task_id="run_research_j1",
        run_id="run_research_j1",
        state=TaskState.COMPLETED,
        answer="research summary",
        metadata={"graph_id": "g1", "agent_ids": ["research_a", "research_b"]},
    )
    with patch(
        "intergrax.runtime.task.unified_task_runner.UnifiedTaskRunner.run_task",
        new_callable=AsyncMock,
        return_value=task_result,
    ) as run_mock:
        response = research_client.post(
            "/v1/research/run",
            json={
                "tenant_id": "t1",
                "user_id": "u1",
                "session_id": "s1",
                "message": "compare vendors",
            },
        )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["answer"] == "research summary"
    assert payload["graph_id"] == "g1"
    assert payload["agent_ids"] == ["research_a", "research_b"]
    run_mock.assert_awaited_once()
