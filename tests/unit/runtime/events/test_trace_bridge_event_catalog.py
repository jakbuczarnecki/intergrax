# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.runtime.events.phase_coverage import phase_for_event
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.events.trace_bridge import _TASK_STATE_TO_EVENT, _TOOL_STEP_TO_EVENT

pytestmark = pytest.mark.gate


def test_trace_bridge_mappings_have_phase_coverage() -> None:
    errors: list[str] = []
    for step, event_type in _TOOL_STEP_TO_EVENT.items():
        if phase_for_event(event_type) is None:
            errors.append(f"tool step {step!r} -> {event_type.value}")
    for _state, event_type in _TASK_STATE_TO_EVENT.items():
        if phase_for_event(event_type) is None:
            errors.append(f"task state mapping -> {event_type.value}")
    for name in ("RETRY_STARTED", "LLM_CALL", "STEP_STARTED", "STEP_COMPLETED"):
        if phase_for_event(RuntimeEventType[name]) is None:
            errors.append(f"special case {name}")
    assert not errors, "missing phase_for_event mappings: " + "; ".join(errors)
