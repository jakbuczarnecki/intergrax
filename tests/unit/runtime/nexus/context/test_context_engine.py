# © Artur Czarnecki. All rights reserved.

"""CE-3.1, CE-3.9, CE-3.10: DefaultNexusContextEngine."""

from __future__ import annotations

import pytest

from intergrax.context.contracts import (
    ContextAssemblyRequest,
    ContextBudgetSnapshot,
    ContextDecisionSnapshot,
    ContextFragment,
    ContextFragmentSource,
    ContextProviderContext,
)
from intergrax.context.registry import ContextPluginRegistry
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.contracts.context_assembly import TaskContextAssemblyOptions
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.runtime.nexus.context.context_engine import DefaultNexusContextEngine

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _SmallWindowAdapter(LLMAdapter):
    provider = "fake"
    model = "fake-small"

    def __init__(self, window: int = 512) -> None:
        super().__init__()
        self._window = window

    @property
    def context_window_tokens(self) -> int:
        return self._window

    def generate_messages(self, messages, **kwargs) -> LLMAdapterResponse:
        _ = messages, kwargs
        return LLMAdapterResponse(content="ok")


@pytest.mark.asyncio
async def test_default_engine_assembles_with_compiler_and_preflight() -> None:
    adapter = _SmallWindowAdapter(window=512)
    config = RuntimeConfig(llm_adapter=adapter, production_mode=False)
    engine = DefaultNexusContextEngine(engine_id="default")
    request = ContextAssemblyRequest(
        trace_id="t1",
        run_id="r1",
        task_id="task1",
        tenant_id="tenant1",
        assembly_scope="acp_step",
        objective="test",
        decision_profile=ContextDecisionSnapshot(),
        budget_policy=ContextBudgetSnapshot(max_tokens_estimate=200),
        assembly_options=TaskContextAssemblyOptions(),
    )
    messages = [ChatMessage(role="user", content="short prompt")]
    provider_ctx = ContextProviderContext(
        engine_id="default",
        handles={
            "runtime_config": config,
            "messages": messages,
            "max_output_tokens": 64,
        },
    )

    assembled = await engine.assemble(request, provider_ctx=provider_ctx)

    assert assembled.total_tokens <= assembled.budget_tokens
    assert len(assembled.fragments_included) >= 1
    assert assembled.messages


class _CollectOnceProvider:
    provider_id = "test.collect"

    @property
    def supported_sources(self) -> frozenset[ContextFragmentSource]:
        return frozenset({ContextFragmentSource.CUSTOM})

    async def collect(
        self,
        request: ContextAssemblyRequest,
        ctx: ContextProviderContext,
    ) -> list[ContextFragment]:
        _ = request
        return [
            ContextFragment(
                fragment_id="frag-a",
                source=ContextFragmentSource.CUSTOM,
                source_id="src-a",
                content="duplicate body",
                token_estimate=4,
                relevance_score=0.5,
                freshness_score=0.5,
                confidence_score=0.5,
                mandatory=False,
                content_hash="same-hash",
            ),
            ContextFragment(
                fragment_id="frag-b",
                source=ContextFragmentSource.CUSTOM,
                source_id="src-b",
                content="duplicate body",
                token_estimate=4,
                relevance_score=0.5,
                freshness_score=0.5,
                confidence_score=0.5,
                mandatory=False,
                content_hash="same-hash",
            ),
        ]


@pytest.mark.asyncio
async def test_engine_emits_candidate_bus_events() -> None:
    adapter = _SmallWindowAdapter(window=512)
    config = RuntimeConfig(llm_adapter=adapter, production_mode=False)
    registry = ContextPluginRegistry()
    registry.add_provider(_CollectOnceProvider())
    engine = DefaultNexusContextEngine(engine_id="default", registry=registry)
    bus = RuntimeEventBus(record_history=True)
    request = ContextAssemblyRequest(
        trace_id="t1",
        run_id="r1",
        task_id="task1",
        tenant_id="tenant1",
        assembly_scope="graph_node",
        objective="test",
        decision_profile=ContextDecisionSnapshot(),
        budget_policy=ContextBudgetSnapshot(max_tokens_estimate=200),
        assembly_options=TaskContextAssemblyOptions(),
        graph_node_id="node-1",
    )
    provider_ctx = ContextProviderContext(
        engine_id="default",
        handles={
            "runtime_config": config,
            "messages": [ChatMessage(role="user", content="short prompt")],
            "max_output_tokens": 64,
            "event_bus": bus,
            "node_id": "node-1",
            "agent_id": "agent-1",
        },
    )

    await engine.assemble(request, provider_ctx=provider_ctx)

    types = [event.event_type for event in bus.history]
    assert RuntimeEventType.CONTEXT_CANDIDATE_COLLECTED in types
    assert RuntimeEventType.CONTEXT_CANDIDATE_DROPPED in types
