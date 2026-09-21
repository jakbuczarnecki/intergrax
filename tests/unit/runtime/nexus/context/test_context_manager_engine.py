# © Artur Czarnecki. All rights reserved.

"""CE-3.4, CE-3.7: ContextManager engine injection on graph path."""

from __future__ import annotations

import pytest

from intergrax.context.bootstrap import bootstrap_context_catalog, reset_context_catalog_bootstrap_for_tests
from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.base.base_llm_adapter import BaseLLMAdapter
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.context.bootstrap import materialize_context_plugin_registry
from intergrax.context.orchestrator import ContextOrchestrator
from intergrax.runtime.nexus.context.codebase_engine import CodebaseContextEngine
from intergrax.runtime.nexus.context.context_engine import DefaultNexusContextEngine
from intergrax.runtime.nexus.context.context_manager import ContextManager
from intergrax.runtime.nexus.context.provider_handles import WORKSPACE_FILES_METADATA_KEY
from intergrax.runtime.nexus.execution.execution_graph import ExecutionNode
from intergrax.runtime.task.task import Task, TaskContext
from testing_support.builder import build_task_for_tests, canonical_execution_identity_scope

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.asyncio]

_ENGINE_SEED = "context-manager-engine"


class _SmallWindowAdapter(BaseLLMAdapter):
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
    task = build_task_for_tests(
        seed=_ENGINE_SEED,
        tenant_id="t1",
        user_id="u1",
        message="hello graph",
    ).model_copy(update={"context": TaskContext()})
    node = ExecutionNode(node_id="n1", agent_id="a1", capability="cap.test")
    prior: dict[str, AgentExecutionResult] = {}

    with canonical_execution_identity_scope(_ENGINE_SEED):
        bundle = await manager.build_agent_context_async(task, node, prior)

    assert bundle.message
    assert bundle.metadata.get("engine_id") == "default"
    assembled = [e for e in bus.history if e.event_type == RuntimeEventType.CONTEXT_ASSEMBLED]
    assert len(assembled) == 1
    assert assembled[0].payload.get("engine_id") == "default"


@pytest.mark.asyncio
async def test_codebase_orchestrator_path_injects_workspace_files() -> None:
    bus = RuntimeEventBus(record_history=True)
    adapter = _SmallWindowAdapter()
    registry = materialize_context_plugin_registry(["intergrax.builtin"])
    engine = CodebaseContextEngine(registry=registry)
    orchestrator = ContextOrchestrator(engine)
    manager = ContextManager(
        event_bus=bus,
        context_engine=engine,
        context_orchestrator=orchestrator,
        llm_adapter=adapter,
    )
    task = build_task_for_tests(
        seed=f"{_ENGINE_SEED}-codebase",
        tenant_id="t1",
        user_id="u1",
        message="fix handler",
    ).model_copy(
        update={
            "context": TaskContext(),
            "metadata": {WORKSPACE_FILES_METADATA_KEY: {"main.py": "print('ok')\n"}},
        }
    )
    node = ExecutionNode(node_id="n1", agent_id="a1", capability="cap.code")
    with canonical_execution_identity_scope(f"{_ENGINE_SEED}-codebase"):
        bundle = await manager.build_agent_context_async(task, node, {})
    assert "main.py" in bundle.message or "print" in bundle.message
    assert bundle.metadata.get("engine_id") == "codebase"
