# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.runtime.events.emit_context import EmitContext
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.event_kind_registry import clear_event_kind_registry
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.observability.extension_sdk import agent_signal_event_kind
from external_contractor_adapter.signals.emit import emit_milestone_reached
from external_contractor_adapter.signals.registry import register_signal_schemas

pytestmark = pytest.mark.gate


@pytest.fixture(autouse=True)
def _register_agent_signal_kinds() -> None:
    clear_event_kind_registry()
    register_signal_schemas()
    yield
    clear_event_kind_registry()


def test_agent_signal_emits_domain_signal() -> None:
    bus = RuntimeEventBus(record_history=True)
    ctx = EmitContext(task_id="task-1", run_id="run-1", tenant_id="tenant-a", bus=bus)
    event = emit_milestone_reached(ctx, milestone="scaffold", detail="smoke")
    kind = agent_signal_event_kind("external_contractor_adapter", "milestone_reached")
    assert event.event_type == RuntimeEventType.DOMAIN_SIGNAL
    assert event.event_kind == kind
    assert bus.history[-1].event_id == event.event_id
