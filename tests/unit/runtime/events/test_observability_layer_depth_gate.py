# © Artur Czarnecki. All rights reserved.

"""OBS L2→L3 depth gate — unified journal, trace bridge catalog, live bus emit."""

from __future__ import annotations

import inspect

import pytest

from intergrax.runtime.events.phase_coverage import phase_for_event
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.events.trace_bridge import (
    _TASK_STATE_TO_EVENT,
    _TOOL_STEP_TO_EVENT,
    trace_event_to_runtime_event,
)
from intergrax.runtime.events.unified_run_journal import build_unified_run_journal
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState

pytestmark = pytest.mark.gate


def test_observability_depth_unified_journal_and_canonical_event_types() -> None:
    assert callable(build_unified_run_journal)
    for name in ("LLM_CALL", "POLICY_DECISION", "TOOL_REQUESTED", "TASK_COMPLETED"):
        event_type = RuntimeEventType[name]
        assert phase_for_event(event_type) is not None


def test_observability_depth_trace_bridge_catalog_complete() -> None:
    errors: list[str] = []
    for step, event_type in _TOOL_STEP_TO_EVENT.items():
        if phase_for_event(event_type) is None:
            errors.append(f"tool step {step!r} -> {event_type.value}")
    for _state, event_type in _TASK_STATE_TO_EVENT.items():
        if phase_for_event(event_type) is None:
            errors.append(f"task state mapping -> {event_type.value}")
    assert not errors


def test_observability_depth_runtime_state_live_emit_wired() -> None:
    source = inspect.getsource(RuntimeState.trace_event)
    assert "runtime_event_bus" in source
    assert "trace_event_to_runtime_event" in source or "record(" in source
