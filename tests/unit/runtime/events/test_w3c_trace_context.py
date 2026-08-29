# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest
from pydantic import ValidationError

from intergrax.contracts.execution_identity import mint_run_id, mint_task_id
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.event_kind_registry import clear_event_kind_registry
from intergrax.runtime.events.payloads.base import RuntimeEventPayload
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from testing_support.runtime_events import emit_context_test_identity, runtime_event_test_identity
from intergrax.runtime.events.signals import emit_domain_signal
from intergrax.runtime.events.w3c_trace_context import (
    child_traceparent,
    format_traceparent,
    generate_trace_id,
    generate_span_id,
    inject_w3c_trace_on_event,
    is_valid_traceparent,
    parse_traceparent,
)
from intergrax.runtime.observability.extension_sdk import register_extension_runtime_payload
from intergrax.runtime.observability.journal_export import render_journal_otlp_json
from intergrax.runtime.task.task import Task, TaskContext

pytestmark = pytest.mark.gate


class _FlagV1(RuntimeEventPayload):
    schema_id = "agents.trace.flag.v1"
    ok: bool = True

    def redact(self) -> _FlagV1:
        return self


@pytest.fixture(autouse=True)
def _register_kind() -> None:
    clear_event_kind_registry()
    register_extension_runtime_payload(_FlagV1, event_kind="agents.trace.clause_flagged")
    yield
    clear_event_kind_registry()


def test_format_and_parse_traceparent_round_trip() -> None:
    trace_id = generate_trace_id()
    parent_id = generate_span_id()
    value = format_traceparent(trace_id=trace_id, parent_id=parent_id, sampled=True)
    parsed = parse_traceparent(value)
    assert parsed.trace_id == trace_id
    assert parsed.parent_id == parent_id
    assert parsed.sampled is True
    assert is_valid_traceparent(value)


def test_child_traceparent_preserves_trace_id() -> None:
    root = format_traceparent(trace_id=generate_trace_id(), parent_id=generate_span_id())
    child = child_traceparent(root)
    assert parse_traceparent(root).trace_id == parse_traceparent(child).trace_id
    assert parse_traceparent(root).parent_id != parse_traceparent(child).parent_id


def test_runtime_event_rejects_invalid_traceparent() -> None:
    with pytest.raises(ValidationError):
        RuntimeEvent(
            event_type=RuntimeEventType.TASK_CREATED,
            phase=ExecutionPhase.INTAKE,
            traceparent="not-a-traceparent",
            **runtime_event_test_identity(),
        )


def test_emit_domain_signal_propagates_trace_context() -> None:
    root = format_traceparent(trace_id=generate_trace_id(), parent_id=generate_span_id())
    ctx = emit_context_test_identity(
        traceparent=root,
        tracestate="vendor1=opaque",
    )
    event = emit_domain_signal(
        ctx,
        kind="agents.trace.clause_flagged",
        payload=_FlagV1(),
    )
    assert event.traceparent == root
    assert event.tracestate == "vendor1=opaque"


def test_inject_w3c_trace_on_event_uses_inbound_metadata() -> None:
    inbound = format_traceparent(trace_id=generate_trace_id(), parent_id=generate_span_id())
    task_id = mint_task_id()
    run_id = mint_run_id()
    task = Task(
        task_id=task_id,
        tenant_id="tenant-a",
        user_id="user-1",
        message="hello",
        context=TaskContext(),
        metadata={"traceparent": inbound, "tracestate": "vendor1=opaque"},
    )
    identity = runtime_event_test_identity(task_id=task_id, run_id=run_id)
    event = RuntimeEvent(
        event_type=RuntimeEventType.TASK_CREATED,
        phase=ExecutionPhase.INTAKE,
        **identity,
    )
    wired = inject_w3c_trace_on_event(event, task)
    assert wired.traceparent is not None
    assert parse_traceparent(wired.traceparent).trace_id == parse_traceparent(inbound).trace_id
    assert task.metadata["w3c_trace_id"] == parse_traceparent(inbound).trace_id


def test_otlp_export_uses_w3c_trace_ids_when_present() -> None:
    trace_id = generate_trace_id()
    span_id = generate_span_id()
    tp = format_traceparent(trace_id=trace_id, parent_id=span_id)
    otlp = render_journal_otlp_json(
        {
            "run_id": "run-1",
            "tenant_id": "tenant-a",
            "events": [
                {
                    "event_id": "evt-1",
                    "event_type": "task_created",
                    "task_id": "task-1",
                    "phase": "intake",
                    "timestamp": "2026-06-17T12:00:00+00:00",
                    "traceparent": tp,
                }
            ],
        }
    )
    span = otlp["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
    assert span["traceId"] == trace_id
    assert span["spanId"] == span_id


def test_with_parent_creates_child_traceparent() -> None:
    root = format_traceparent(trace_id=generate_trace_id(), parent_id=generate_span_id())
    identity = runtime_event_test_identity()
    parent = RuntimeEvent(
        event_type=RuntimeEventType.TASK_CREATED,
        phase=ExecutionPhase.INTAKE,
        traceparent=root,
        tracestate="vendor1=opaque",
        **identity,
    )
    child = RuntimeEvent(
        event_type=RuntimeEventType.STEP_STARTED,
        phase=ExecutionPhase.STEP_EXECUTION,
        **identity,
    ).with_parent(parent)
    assert child.parent_event_id == parent.event_id
    assert parse_traceparent(child.traceparent or "").trace_id == parse_traceparent(root).trace_id
    assert child.tracestate == "vendor1=opaque"
