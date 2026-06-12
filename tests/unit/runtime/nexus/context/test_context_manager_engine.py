# © Artur Czarnecki. All rights reserved.

"""CE-3.4, CE-3.7: ContextManager engine injection on graph path."""

from __future__ import annotations

import pytest

from intergrax.context.bootstrap import bootstrap_context_catalog, reset_context_catalog_bootstrap_for_tests
from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.nexus.context.context_engine import DefaultNexusContextEngine
from intergrax.runtime.nexus.context.context_manager import ContextManager
from intergrax.runtime.nexus.execution.execution_graph import ExecutionNode
from intergrax.runtime.task.task import Task, TaskContext

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.asyncio]


class _SmallWindowAdapter(LLMAdapter):
    provider = "fake"
    model = "fake-small"

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
def _catalog() -> None:
    reset_context_catalog_bootstrap_for_tests()
    bootstrap_context_catalog(register_shipped=True, discover_entry_points=False)
    yield
    reset_context_catalog_bootstrap_for_tests()


@pytest.mark.asyncio
async def test_build_agent_context_async_records_engine_id() -> None:
    bus = RuntimeEventBus(record_history=True)
    adapter = _SmallWindowAdapter()
    engine = DefaultNexusContextEngine(engine_id="default")
    manager = ContextManager(
        event_bus=bus,
        context_engine=engine,
        llm_adapter=adapter,
    )
    task = Task(tenant_id="t1", user_id="u1", message="hello graph", context=TaskContext())
    node = ExecutionNode(node_id="n1", agent_id="a1", capability="cap.test")
    prior: dict[str, AgentExecutionResult] = {}

    bundle = await manager.build_agent_context_async(task, node, prior)

    assert bundle.message
    assert bundle.metadata.get("engine_id") == "default"
    assembled = [e for e in bus.history if e.event_type == RuntimeEventType.CONTEXT_ASSEMBLED]
    assert len(assembled) == 1
    assert assembled[0].payload.get("engine_id") == "default"
