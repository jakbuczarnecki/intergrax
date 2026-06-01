# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.rag.tracking.metrics import get_rag_metrics_collector, record_retrieval, set_rag_metrics_enabled
from intergrax.rag.tracking.observability_bridge import make_rag_metrics_runtime_plugin
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.hooks.hook_registry import HookRegistry

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_rag_metrics_plugin_handles_task_completed() -> None:
    set_rag_metrics_enabled(True)
    record_retrieval(
        tenant_id="t1",
        route_tier="standard",
        retriever_id="hybrid",
        retrieval_latency_ms=12.0,
        hits=2,
    )

    bus = RuntimeEventBus(record_history=False)
    plugin = make_rag_metrics_runtime_plugin()
    plugin.register(bus, HookRegistry(), MagicMock())

    event = RuntimeEvent(
        task_id="task-1",
        run_id="run-1",
        tenant_id="t1",
        agent_id="agent-1",
        event_type=RuntimeEventType.TASK_COMPLETED,
        phase=ExecutionPhase.COMPLETION,
    )
    await bus.publish(event)
    set_rag_metrics_enabled(False)
