# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from intergrax.llm_adapters.tracking.metrics import get_llm_metrics_collector, record_llm_call, set_metrics_enabled
from intergrax.llm_adapters.tracking.observability_bridge import make_llm_metrics_runtime_plugin
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.hooks.hook_registry import HookRegistry

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_llm_metrics_plugin_handles_task_completed() -> None:
    set_metrics_enabled(True)
    get_llm_metrics_collector().reset()
    record_llm_call(
        provider="groq",
        model="m",
        run_id="r",
        input_tokens=3,
        output_tokens=2,
        duration_ms=10,
        success=True,
    )

    bus = RuntimeEventBus(record_history=False)
    plugin = make_llm_metrics_runtime_plugin()
    plugin.register(bus, HookRegistry(), MagicMock())

    event = RuntimeEvent(
        task_id="task-1",
        run_id="run-1",
        tenant_id="_platform",
        agent_id="agent-1",
        event_type=RuntimeEventType.TASK_COMPLETED,
        phase=ExecutionPhase.COMPLETION,
    )
    await bus.publish(event)
    set_metrics_enabled(False)
