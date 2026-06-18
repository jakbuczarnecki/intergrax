# © Artur Czarnecki. All rights reserved.

"""CE-PROV-INT: graph engine assemble includes RAG + graph_prior fragments."""

from __future__ import annotations

import pytest

from intergrax.context.bootstrap import bootstrap_context_catalog, reset_context_catalog_bootstrap_for_tests
from intergrax.context.bootstrap import materialize_context_plugin_registry
from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.nexus.context.context_manager import ContextManager
from intergrax.runtime.nexus.context.context_engine import DefaultNexusContextEngine
from intergrax.runtime.nexus.context.provider_handles import RAG_CHUNKS_METADATA_KEY
from intergrax.runtime.nexus.execution.execution_graph import ExecutionNode
from intergrax.runtime.task.task import Task, TaskContext

pytestmark = [pytest.mark.integration, pytest.mark.gate]


class _WindowAdapter(LLMAdapter):
    provider = "fake"
    model = "fake"

    @property
    def context_window_tokens(self) -> int:
        return 8192

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
async def test_graph_assemble_includes_rag_and_graph_prior_fragments() -> None:
    bus = RuntimeEventBus(record_history=True)
    adapter = _WindowAdapter()
    registry = materialize_context_plugin_registry(["intergrax.builtin"])
    engine = DefaultNexusContextEngine(engine_id="default", registry=registry)
    manager = ContextManager(event_bus=bus, context_engine=engine, llm_adapter=adapter)

    prior = {
        "dep-1": AgentExecutionResult(
            agent_id="agent-a",
            run_id="run-dep",
            status=AgentExecutionStatus.COMPLETED,
            summary="dependency summary for rag task",
        )
    }
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="answer using retrieved docs",
        context=TaskContext(),
        metadata={
            RAG_CHUNKS_METADATA_KEY: [
                {
                    "text": "Retrieved policy paragraph.",
                    "metadata": {"doc_id": "policy-1", "source": "handbook.pdf"},
                }
            ],
        },
    )
    node = ExecutionNode(
        node_id="n1",
        agent_id="worker",
        capability="cap.test",
        depends_on=["dep-1"],
    )

    bundle = await manager.build_agent_context_async(task, node, prior)

    assert "[context:rag:" in bundle.message
    assert "Retrieved policy paragraph." in bundle.message
    assert "[context:graph_prior:dep-1]" in bundle.message
    assert "dependency summary for rag task" in bundle.message
    assert bundle.metadata.get("engine_id") == "default"
