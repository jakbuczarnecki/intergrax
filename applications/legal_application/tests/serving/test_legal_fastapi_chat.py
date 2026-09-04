# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from intergrax.agents.agent_contract import Agent
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.fastapi_core.app_factory import create_app
from intergrax.fastapi_core.config import ApiConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.applications._shared.host_task_execution_wiring import build_host_task_execution
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.task.task import TaskResult, TaskState
from legal_application.serving.fastapi_router import mount_legal_agent_routes
from testing_support.agent_registry_bootstrap import bootstrap_agent_registry_from_agents

pytestmark = pytest.mark.unit


@pytest.fixture
def client() -> TestClient:
    app = create_app(ApiConfig())

    class _DummyAgent(Agent):
        def get_contract(self) -> AgentContract:
            return AgentContract(
                id="legal-test",
                name="Dummy",
                description="test",
                capabilities=["legal.contract_review"],
            )

        def build_context(self, request: RuntimeRequest) -> RuntimeContext:
            raise RuntimeError("not used when HostTaskExecution.execute is mocked")

    agent = _DummyAgent()
    registry = bootstrap_agent_registry_from_agents({"legal-test": agent})
    nexus = NexusLoop(registry)
    host_execution = build_host_task_execution(nexus, orchestration_triggers=frozenset())
    mount_legal_agent_routes(
        app,
        registry=registry,
        default_agent_id="legal-test",
        host_execution=host_execution,
    )
    return TestClient(app)


def test_legal_chat_returns_mapped_answer(client: TestClient) -> None:
    task_result = TaskResult(
        task_id="run-api-1",
        run_id="run-api-1",
        state=TaskState.COMPLETED,
        answer="API OK",
    )
    with patch(
        "intergrax.runtime.execution.host_task.HostTaskExecution.execute",
        new_callable=AsyncMock,
        return_value=task_result,
    ):
        r = client.post(
            "/v1/legal/chat",
            json={
                "message": "ping",
                "session_id": "s-api",
                "tenant_id": "ten-api",
                "user_id": "user-api",
            },
        )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["answer"] == "API OK"
    assert data["run_id"] == "run-api-1"
    assert data["request_id"]
    assert data["api_version"] == "1"


def test_legal_chat_400_without_tenant(client: TestClient) -> None:
    with patch(
        "intergrax.runtime.execution.host_task.HostTaskExecution.execute",
        new_callable=AsyncMock,
    ):
        r = client.post(
            "/v1/legal/chat",
            json={
                "message": "ping",
                "session_id": "s-api",
                "user_id": "user-api",
            },
        )
    assert r.status_code == 400
