# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.contracts.event_severity import EventSeverity
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.event_catalog import EventCategory
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType

pytestmark = pytest.mark.gate


def test_runtime_event_auto_fills_kind_category_and_ops_hint() -> None:
    event = RuntimeEvent(
        task_id="task-1",
        run_id="run-1",
        event_type=RuntimeEventType.TOOL_COMPLETED,
        phase=ExecutionPhase.STEP_EXECUTION,
    )
    assert event.event_kind == "tool_completed"
    assert event.event_category == EventCategory.TOOL
    assert event.ops_hint == "ops:tool_audit"


def test_runtime_event_preserves_explicit_event_kind() -> None:
    event = RuntimeEvent(
        task_id="task-1",
        run_id="run-1",
        event_type=RuntimeEventType.DOMAIN_SIGNAL,
        event_kind="agents.legal.clause_flagged",
        event_category=EventCategory.AGENT,
        ops_hint="ops:domain_signal",
        phase=ExecutionPhase.STEP_EXECUTION,
    )
    assert event.event_kind == "agents.legal.clause_flagged"
    assert event.event_category == EventCategory.AGENT


def test_domain_signal_catalog_entry_exists() -> None:
    event = RuntimeEvent(
        task_id="task-1",
        run_id="run-1",
        event_type=RuntimeEventType.DOMAIN_SIGNAL,
        event_kind="platform.adaptive.custom",
        phase=ExecutionPhase.STEP_EXECUTION,
        severity=EventSeverity.INFO,
    )
    assert event.ops_hint == "ops:domain_signal"
    assert event.event_category == EventCategory.PLATFORM
