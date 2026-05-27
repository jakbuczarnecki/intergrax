# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from intergrax.agents.agent_contract import Agent
from intergrax.contracts.agent_contract_meta import AgentContract
from legal_application.serving.fastapi_router import (
    mount_legal_agent_routes,
)
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.fastapi_core.app_factory import create_app
from intergrax.fastapi_core.config import ApiConfig
from intergrax.runtime.nexus.responses.response_schema import (
    RouteInfo,
    RuntimeAnswer,
    RuntimeStats,
    StopReason,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def client() -> TestClient:
    app = create_app(ApiConfig())

    class _DummyAgent(Agent):
        def get_contract(self) -> AgentContract:
            return AgentContract(id="legal-test", name="Dummy", description="test")

        def build_context(self, request: RuntimeRequest) -> RuntimeContext:
            raise RuntimeError("not used when AgentEngine.run is mocked")

    mount_legal_agent_routes(
        app,
        agents={"legal-test": _DummyAgent()},
        default_agent_id="legal-test",
    )
    return TestClient(app)


def test_legal_chat_returns_mapped_answer(client: TestClient) -> None:
    answer = RuntimeAnswer(
        answer="API OK",
        stop_reason=StopReason.COMPLETED,
        run_id="run-api-1",
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
    with patch("intergrax.agents.agent_engine.AgentEngine.run", new_callable=AsyncMock):
        r = client.post(
            "/v1/legal/chat",
            json={
                "message": "ping",
                "session_id": "s-api",
                "user_id": "user-api",
            },
        )
    assert r.status_code == 400
