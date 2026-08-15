# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import re

import pytest

from intergrax.contracts.execution_identity import (
    ActiveExecutionIdentity,
    mint_attempt_id,
    mint_event_id,
    mint_run_id,
    mint_task_id,
    validate_event_id,
)
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.runtime.events.emit_context import EmitContext
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.events.signals import emit_domain_signal, emit_platform_event
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.payloads.canonical import ToolPayloadV1
from intergrax.runtime.nexus.retry.coordinator import RetryCoordinator
from intergrax.runtime.nexus.retry.retry_engine import RetryRecord
from intergrax.runtime.task.task import Task, TaskContext

_CANONICAL_ID = re.compile(r"^(task|run|attempt|evt)_[0-9a-f]{32}$")


@pytest.mark.unit
@pytest.mark.gate
def test_validate_event_id_accepts_valid() -> None:
    value = mint_event_id()
    assert validate_event_id(value) == value
    assert _CANONICAL_ID.fullmatch(value)


@pytest.mark.unit
@pytest.mark.gate
def test_mint_event_id_format() -> None:
    assert mint_event_id().startswith("evt_")


@pytest.mark.unit
@pytest.mark.gate
def test_validate_event_id_rejects_wrong_prefix() -> None:
    with pytest.raises(ValueError, match="EventId must start with"):
        validate_event_id("task_0123456789abcdef0123456789abcdef")


@pytest.mark.unit
@pytest.mark.gate
def test_validate_event_id_rejects_uppercase_hex() -> None:
    event_id = mint_event_id()
    upper = "evt_" + event_id.split("_", 1)[1].upper()
    with pytest.raises(ValueError, match="suffix"):
        validate_event_id(upper)


@pytest.mark.unit
@pytest.mark.gate
def test_validate_event_id_rejects_malformed_suffix() -> None:
    with pytest.raises(ValueError, match="suffix"):
        validate_event_id("evt_short")


@pytest.mark.unit
@pytest.mark.gate
def test_validate_event_id_rejects_non_string() -> None:
    with pytest.raises(TypeError):
        validate_event_id(123)


@pytest.mark.unit
@pytest.mark.gate
def test_runtime_event_requires_all_identity_fields() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    event = RuntimeEvent(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        event_type=RuntimeEventType.STEP_STARTED,
        phase=ExecutionPhase.STEP_EXECUTION,
    )
    assert event.task_id == task_id
    assert event.run_id == run_id
    assert event.attempt_id == attempt_id
    assert _CANONICAL_ID.fullmatch(event.event_id)
    dumped = event.model_dump(mode="json")
    assert dumped["task_id"] == task_id
    assert dumped["run_id"] == run_id
    assert dumped["attempt_id"] == attempt_id
    assert dumped["event_id"] == event.event_id


@pytest.mark.unit
@pytest.mark.gate
def test_runtime_event_rejects_missing_attempt_id() -> None:
    with pytest.raises(Exception):
        RuntimeEvent(
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            event_type=RuntimeEventType.STEP_STARTED,
            phase=ExecutionPhase.STEP_EXECUTION,
        )


@pytest.mark.unit
@pytest.mark.gate
def test_runtime_event_rejects_malformed_attempt_id() -> None:
    with pytest.raises(ValueError):
        RuntimeEvent(
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            attempt_id="attempt_BAD",
            event_type=RuntimeEventType.STEP_STARTED,
            phase=ExecutionPhase.STEP_EXECUTION,
        )


@pytest.mark.unit
@pytest.mark.gate
def test_emit_context_propagates_to_platform_event() -> None:
    bus = RuntimeEventBus(record_history=True)
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    ctx = EmitContext(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        bus=bus,
    )
    event = emit_platform_event(
        ctx,
        event_type=RuntimeEventType.TASK_CREATED,
        payload=ToolPayloadV1(tool_name="x", status="requested"),
        phase=ExecutionPhase.INTAKE,
    )
    assert event.task_id == task_id
    assert event.run_id == run_id
    assert event.attempt_id == attempt_id
    assert bus.history[-1].attempt_id == attempt_id


@pytest.mark.unit
@pytest.mark.gate
def test_emit_context_propagates_to_domain_signal() -> None:
    from intergrax.runtime.events.event_kind_registry import clear_event_kind_registry
    from intergrax.runtime.observability.extension_sdk import register_extension_runtime_payload
    from pydantic import Field
    from intergrax.runtime.events.payloads.base import RuntimeEventPayload

    class _Payload(RuntimeEventPayload):
        schema_id = "agents.test.trace_1b.v1"
        value: str = Field(default="ok")

        def redact(self) -> _Payload:
            return self

    clear_event_kind_registry()
    register_extension_runtime_payload(_Payload, event_kind="agents.test.trace_1b")
    bus = RuntimeEventBus(record_history=True)
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    ctx = EmitContext(task_id=task_id, run_id=run_id, attempt_id=attempt_id, bus=bus)
    event = emit_domain_signal(
        ctx,
        kind="agents.test.trace_1b",
        payload=_Payload(),
    )
    assert event.attempt_id == attempt_id
    clear_event_kind_registry()


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_runtime_execution_context_tool_events_carry_attempt_id() -> None:
    from intergrax.contracts.tool_request import ToolRequest, ToolResponse, ToolResponseStatus

    class _Gateway:
        async def invoke(self, request: ToolRequest) -> ToolResponse:
            return ToolResponse(request_id=request.request_id, status=ToolResponseStatus.SUCCESS)

    class _Collector:
        def __init__(self) -> None:
            self.events: list[RuntimeEvent] = []

        async def emit(self, event: RuntimeEvent) -> None:
            self.events.append(event)

    collector = _Collector()
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    ctx = RuntimeExecutionContext(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        agent_id="agent-1",
        tool_gateway=_Gateway(),
        event_emitter=collector,
    )
    await ctx.invoke_tool(
        ToolRequest(tool_name="rag.ingest_document", agent_id="agent-1", input={"x": 1})
    )
    assert collector.events
    for event in collector.events:
        assert event.task_id == task_id
        assert event.run_id == run_id
        assert event.attempt_id == attempt_id


@pytest.mark.unit
@pytest.mark.gate
def test_retry_coordinator_scheduled_and_started_attempt_ids() -> None:
    task = Task(tenant_id="t", user_id="u", context=TaskContext())
    run_id = mint_run_id()
    attempt_a1 = mint_attempt_id()
    attempt_a2 = mint_attempt_id()
    coordinator = RetryCoordinator(max_run_retries=1, retry_run_on=frozenset())
    scheduled = coordinator.scheduled_event_for_agent_retry(
        task,
        run_id=run_id,
        attempt_id=attempt_a1,
        record=RetryRecord(
            attempt=1,
            agent_id="a1",
            reason="validation_failed",
            alternate_agent_id="a2",
        ),
    )
    started = RetryCoordinator.build_started_event(
        task,
        run_id=run_id,
        attempt_id=attempt_a2,
        scope="agent",
        retry_ordinal=1,
        reason="validation_failed",
    )
    assert scheduled.event_type == RuntimeEventType.RETRY_SCHEDULED
    assert scheduled.attempt_id == attempt_a1
    assert started.event_type == RuntimeEventType.RETRY_STARTED
    assert started.attempt_id == attempt_a2


@pytest.mark.unit
@pytest.mark.gate
def test_active_execution_identity_transition_retry_mints_new_attempt() -> None:
    identity = ActiveExecutionIdentity()
    run_id = mint_run_id()
    attempt_a1 = mint_attempt_id()
    identity.bind(run_id=run_id, attempt_id=attempt_a1)
    attempt_a2 = identity.transition_retry()
    assert attempt_a2 != attempt_a1
    assert identity.run_id == run_id
    assert identity.attempt_id == attempt_a2
