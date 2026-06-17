# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.emit_context import EmitContext
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.event_catalog import EventCategory
from intergrax.runtime.events.journal_query import query_journal
from intergrax.runtime.events.payloads.base import RuntimeEventPayload
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.events.signals import emit_domain_signal
from intergrax.runtime.observability.extension_sdk import register_extension_runtime_payload

pytestmark = pytest.mark.gate


class _AgentFlagV1(RuntimeEventPayload):
    schema_id = "agents.legal.flag.v1"
    ok: bool = True


class _ResearchNoteV1(RuntimeEventPayload):
    schema_id = "agents.research.note_added.v1"
    note_id: str = "n1"


@pytest.fixture(autouse=True)
def _register_kind() -> None:
    from intergrax.runtime.events.event_kind_registry import clear_event_kind_registry

    clear_event_kind_registry()
    register_extension_runtime_payload(
        _AgentFlagV1,
        event_kind="agents.legal.clause_flagged",
    )
    register_extension_runtime_payload(
        _ResearchNoteV1,
        event_kind="agents.research.note_added",
    )
    yield
    clear_event_kind_registry()


def test_subscribe_by_kind_prefix_on_record() -> None:
    bus = RuntimeEventBus(record_history=True)
    seen: list[str] = []
    bus.subscribe(lambda e: seen.append(e.event_kind), kind_prefix="agents.legal.")
    ctx = EmitContext(task_id="t1", run_id="r1", bus=bus)
    emit_domain_signal(ctx, kind="agents.legal.clause_flagged", payload=_AgentFlagV1())
    emit_domain_signal(ctx, kind="agents.research.note_added", payload=_ResearchNoteV1())
    assert seen == ["agents.legal.clause_flagged"]


def test_subscribe_by_category() -> None:
    bus = RuntimeEventBus(record_history=True)
    seen: list[EventCategory] = []
    bus.subscribe(lambda e: seen.append(e.event_category), categories={EventCategory.TOOL})
    ctx = EmitContext(task_id="t1", run_id="r1", bus=bus)
    from intergrax.runtime.events.payloads.canonical import ToolPayloadV1
    from intergrax.runtime.events.signals import emit_platform_event

    emit_platform_event(
        ctx,
        event_type=RuntimeEventType.TOOL_COMPLETED,
        payload=ToolPayloadV1(tool_name="x", status="ok"),
    )
    emit_domain_signal(ctx, kind="agents.legal.clause_flagged", payload=_AgentFlagV1())
    assert seen == [EventCategory.TOOL]


def test_query_journal_filters_kind_prefix() -> None:
    bus = RuntimeEventBus(record_history=True)
    ctx = EmitContext(task_id="t1", run_id="r1", bus=bus)
    emit_domain_signal(ctx, kind="agents.legal.clause_flagged", payload=_AgentFlagV1())
    emit_domain_signal(ctx, kind="agents.research.note_added", payload=_ResearchNoteV1())
    legal = query_journal(bus.history, kind_prefix="agents.legal.")
    assert len(legal) == 1
    assert legal[0].event_kind == "agents.legal.clause_flagged"


def test_legacy_event_types_subscribe_unchanged() -> None:
    bus = RuntimeEventBus(record_history=True)
    seen: list[RuntimeEventType] = []
    bus.subscribe(
        lambda e: seen.append(e.event_type),
        event_types={RuntimeEventType.TASK_CREATED},
    )
    bus.record(
        RuntimeEvent(
            task_id="t1",
            run_id="r1",
            event_type=RuntimeEventType.TASK_CREATED,
            phase=ExecutionPhase.INTAKE,
        )
    )
    bus.record(
        RuntimeEvent(
            task_id="t1",
            run_id="r1",
            event_type=RuntimeEventType.TASK_FAILED,
            phase=ExecutionPhase.COMPLETION,
        )
    )
    assert seen == [RuntimeEventType.TASK_CREATED]
