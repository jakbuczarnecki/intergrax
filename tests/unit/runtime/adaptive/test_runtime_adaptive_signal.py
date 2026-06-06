# © Artur Czarnecki. All rights reserved.

"""W-ADAPT-1.11: RuntimeEngine adaptive signal emission unit test."""

from __future__ import annotations

import pytest

from intergrax.runtime.adaptive.signal_collector import SignalCollector
from intergrax.runtime.adaptive.signal_store import InMemorySignalStore
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime import RuntimeEngine
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest, RuntimeAnswer
from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@pytest.mark.asyncio
async def test_runtime_engine_records_adaptive_outcome_signal() -> None:
    store = InMemorySignalStore()
    collector = SignalCollector(store, application_id="runtime.lab")
    config = RuntimeConfig(
        llm_adapter=FakeLLMAdapter(),
        enable_rag=False,
        production_mode=False,
        signal_collector=collector,
    )
    context = RuntimeContext.build(
        config=config,
        session_manager=build_in_memory_session_manager(),
    )
    engine = RuntimeEngine(context=context)
    request = RuntimeRequest(
        message="hello",
        tenant_id="t-runtime",
        agent_id="echo",
        user_id="u1",
        session_id="s1",
    )
    state = RuntimeState(context=context, request=request, run_id="run_engine_1")
    answer = RuntimeAnswer(run_id="run_engine_1", answer="hello response")

    engine._maybe_record_adaptive_outcome_signal(
        request=request,
        state=state,
        runtime_answer=answer,
        start_perf=0.0,
    )

    signals = store.list_signals(tenant_id="t-runtime")
    assert len(signals) == 1
    assert signals[0].run_id == "run_engine_1"
    assert signals[0].validation_passed is True
