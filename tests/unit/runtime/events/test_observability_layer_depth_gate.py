# © Artur Czarnecki. All rights reserved.

"""OBS L3→L4 depth gate — unified journal, spine wiring, payload registry, export."""

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
from intergrax.runtime.events.payload_registry import list_registered_payload_schema_ids
from intergrax.runtime.events.unified_run_journal import build_unified_run_journal
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.orchestration.task_events import NexusRuntimeEventPublisher
from intergrax.runtime.observability.emitter import ObservabilityEmitter
from intergrax.runtime.observability.journal_export import JOURNAL_EXPORT_SCHEMA_VERSION

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
    trace_source = inspect.getsource(RuntimeState.trace_event)
    wiring_source = inspect.getsource(RuntimeState._get_observability_emitter)
    bridge_source = inspect.getsource(ObservabilityEmitter._bridge_trace_event)
    assert "emit_diagnostic" in trace_source
    assert "runtime_event_bus" in wiring_source
    assert "trace_event_to_runtime_event" in bridge_source
    assert "record(" in bridge_source


def test_observability_l4_payload_registry_covers_canonical_families() -> None:
    registered = set(list_registered_payload_schema_ids())
    for schema_id in (
        "decision.v1",
        "tool.v1",
        "validation.v1",
        "agent_selection.v1",
        "graph_node.v1",
    ):
        assert schema_id in registered


def test_observability_l4_journal_export_and_terminal_ref_wired() -> None:
    terminal_source = inspect.getsource(NexusRuntimeEventPublisher._terminal_payload_for_task)
    assert "journal_ref" in terminal_source
    assert JOURNAL_EXPORT_SCHEMA_VERSION


def test_observability_l4_platform_bootstrap_registers_journal_export() -> None:
    from pathlib import Path

    source_path = (
        Path(__file__).resolve().parents[4]
        / "intergrax"
        / "applications"
        / "_shared"
        / "platform_wiring.py"
    )
    source = source_path.read_text(encoding="utf-8")
    assert "register_journal_export_plugin" in source
