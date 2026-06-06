# © Artur Czarnecki. All rights reserved.

"""W-ADAPT-1.10: Nexus adaptive signal emission integration test."""

from __future__ import annotations

import pytest

from echo.echo_agent import EchoAgent
from intergrax.runtime.adaptive.signal_collector import SignalCollector
from intergrax.runtime.adaptive.signal_store import InMemorySignalStore
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext, TaskState

pytestmark = [pytest.mark.asyncio, pytest.mark.integration, pytest.mark.gate]


async def test_nexus_loop_emits_adaptive_outcome_signal_on_completion() -> None:
    registry = AgentRegistry()
    registry.register(EchoAgent())
    store = InMemorySignalStore()
    collector = SignalCollector(store, application_id="integration.lab")
    loop = NexusLoop(registry, signal_collector=collector)

    task = Task(
        tenant_id="t-adapt",
        user_id="u1",
        message="adaptive signal check",
        context=TaskContext(capability="echo.basic"),
    )
    result = await loop.handle_task(task)

    assert result.state == TaskState.COMPLETED
    signals = store.list_signals(tenant_id="t-adapt")
    assert len(signals) == 1
    signal = signals[0]
    assert signal.run_id == result.run_id
    assert signal.agent_id == "echo"
    assert signal.application_id == "integration.lab"
    assert signal.utility is not None
