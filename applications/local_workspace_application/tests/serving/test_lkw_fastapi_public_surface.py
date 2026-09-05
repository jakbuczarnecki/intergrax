# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.runtime.task.task import TaskResult, TaskState
from local_workspace_application.host.task_executor import LocalWorkspaceTaskExecutor
from local_workspace_application.serving.fastapi_router import mount_local_workspace_routes

pytestmark = [pytest.mark.unit]


class _FakeRegistry:
    def get(self, agent_id: str) -> object:
        raise KeyError(agent_id)

    def get_contract(self, agent_id: str) -> AgentContract:
        raise KeyError(agent_id)

    def has(self, agent_id: str) -> bool:
        return agent_id in {contract.id for contract in self.list_contracts()}

    def list_agent_ids(self) -> list[str]:
        return [contract.id for contract in self.list_contracts()]

    def list_contracts(self) -> list[AgentContract]:
        return [
            AgentContract(
                id="local_search",
                name="Local Search",
                description="search",
                capabilities=["local.workspace.search"],
            )
        ]

    def is_routable(self, agent_id: str, *, production_mode: bool = False) -> bool:
        return self.has(agent_id)

    def list_routable_agent_ids(self, *, production_mode: bool = False) -> list[str]:
        return self.list_agent_ids()

    def find_by_capability(
        self,
        capability: str,
        *,
        production_mode: bool = False,
    ) -> list[object]:
        return []

    def find_best_match(
        self,
        task_context: object,
        *,
        production_mode: bool = False,
    ) -> object | None:
        return None


def test_get_agents_uses_registry_without_nexus() -> None:
    app = FastAPI()
    registry = _FakeRegistry()
    executor = AsyncMock(spec=LocalWorkspaceTaskExecutor)
    mount_local_workspace_routes(
        app,
        task_executor=executor,
        registry=registry,
    )
    client = TestClient(app)

    response = client.get("/v1/local_workspace/agents")

    assert response.status_code == 200
    payload = response.json()
    assert payload["agents"] == [
        {
            "agent_id": "local_search",
            "name": "Local Search",
            "capabilities": ["local.workspace.search"],
        }
    ]


def test_post_run_reaches_task_executor() -> None:
    app = FastAPI()
    executor = AsyncMock(spec=LocalWorkspaceTaskExecutor)
    executor.execute = AsyncMock(
        return_value=TaskResult(
            task_id="task-http",
            run_id="run-http",
            state=TaskState.COMPLETED,
            answer="ok",
            agent_id="local_search",
            metadata={},
        )
    )
    mount_local_workspace_routes(
        app,
        task_executor=executor,
        registry=_FakeRegistry(),
    )
    client = TestClient(app)

    response = client.post(
        "/v1/local_workspace/run",
        json={"message": "hello", "capability": "local.workspace.search"},
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "ok"
    executor.execute.assert_awaited_once()


def test_fastapi_router_has_no_nexus_registry_dependency() -> None:
    router_source = (
        Path(__file__).resolve().parents[2] / "serving" / "fastapi_router.py"
    ).read_text(encoding="utf-8")
    forbidden = ("NexusLoop", "nexus_loop", "resolve_harness_host_nexus_loop_legacy")
    for token in forbidden:
        assert token not in router_source


def test_runtime_event_metadata_has_no_nexus_dependency() -> None:
    metadata_source = (
        Path(__file__).resolve().parents[2] / "serving" / "runtime_event_metadata.py"
    ).read_text(encoding="utf-8")
    forbidden = ("NexusLoop", "nexus_loop", "event_bus")
    for token in forbidden:
        assert token not in metadata_source
