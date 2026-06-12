# © Artur Czarnecki. All rights reserved.

"""CE-3.8: graph + UAEP emit CONTEXT_ASSEMBLED with engine_id=default."""

from __future__ import annotations

import pytest

from intergrax.agents.agent_engine import AgentEngine
from intergrax.applications._shared.context_wiring import resolve_context_manager_from_environment
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.context.bootstrap import reset_context_catalog_bootstrap_for_tests
from intergrax.contracts.agent_execution_result import AgentExecutionStatus
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.nexus.execution.execution_graph import ExecutionGraph, ExecutionNode
from intergrax.runtime.nexus.execution.graph_executor import GraphExecutor
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext
from echo.echo_agent import EchoAgent
from testing_support.uaep_gate_stubs import UaepPipelineStubAgent

pytestmark = [pytest.mark.asyncio, pytest.mark.integration, pytest.mark.gate]


class _GateWindowAdapter(LLMAdapter):
    provider = "fake"
    model = "fake-gate"

    def __init__(self, window: int = 4096) -> None:
        super().__init__()
        self._window = window

    @property
    def context_window_tokens(self) -> int:
        return self._window

    def generate_messages(self, messages, **kwargs) -> LLMAdapterResponse:
        _ = messages, kwargs
        return LLMAdapterResponse(content="ok")


@pytest.fixture(autouse=True)
def _reset_context_catalog() -> None:
    reset_context_catalog_bootstrap_for_tests()
    yield
    reset_context_catalog_bootstrap_for_tests()


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.gate
async def test_graph_path_emits_context_assembled_with_engine_id() -> None:
    agent = UaepPipelineStubAgent(
        agent_id="worker",
        capability="cap.worker",
        prefix="W",
        answer_separator=":",
        description="ce graph stub",
    )
    registry = AgentRegistry()
    registry.register(agent)

    bus = RuntimeEventBus(record_history=True)
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="ce.paths.graph")
    adapter = _GateWindowAdapter()
    context_manager = resolve_context_manager_from_environment(
        env,
        event_bus=bus,
        llm_adapter=adapter,
    )
    executor = GraphExecutor(registry, event_bus=bus, context_manager=context_manager)
    task = Task(tenant_id="t1", user_id="u1", message="assemble me", context=TaskContext(capability="cap.worker"))
    graph = ExecutionGraph(
        graph_id="ce_graph",
        task_id=task.task_id,
        nodes=[ExecutionNode(node_id="n1", agent_id="worker", capability="cap.worker")],
    )

    await executor.execute(graph, task)

    assembled = [e for e in bus.history if e.event_type == RuntimeEventType.CONTEXT_ASSEMBLED]
    assert assembled
    assert assembled[-1].payload.get("engine_id") == "default"


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.gate
async def test_uaep_path_emits_context_assembled_with_engine_id() -> None:
    bus = RuntimeEventBus(record_history=True)
    engine = AgentEngine({"echo": EchoAgent()}, event_bus=bus)
    request = RuntimeRequest(
        tenant_id="t1",
        user_id="u1",
        session_id="s1",
        agent_id="echo",
        message="uaep assemble",
        metadata={"run_id": "run_ce_paths", "task_id": "task_ce_paths"},
    )

    result = await engine.run_with_result(request)

    assert result.status == AgentExecutionStatus.COMPLETED
    assembled = [e for e in bus.history if e.event_type == RuntimeEventType.CONTEXT_ASSEMBLED]
    assert assembled
    assert assembled[0].payload.get("engine_id") == "default"
