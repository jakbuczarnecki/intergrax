# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.emit_context import EmitContext
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.payload_registry import register_payload_schema
from intergrax.runtime.events.event_kind_registry import register_event_kind
from intergrax.runtime.events.payloads.base import RuntimeEventPayload
from intergrax.runtime.events.signals import emit_domain_signal

pytestmark = pytest.mark.unit


class _PhasePayload(RuntimeEventPayload):
    schema_id = "intergrax.test.emit_phase.v1"

    marker: str = "x"

    def redact(self) -> _PhasePayload:
        return self


@pytest.fixture(autouse=True)
def _register_phase_payload() -> None:
    register_payload_schema(_PhasePayload, extension=True)
    register_event_kind("applications.test.emit_phase", _PhasePayload.schema_id)


def test_emit_domain_signal_without_phase_uses_step_execution() -> None:
    bus = RuntimeEventBus()
    ctx = EmitContext(task_id="task-1", run_id="run-1", bus=bus)
    event = emit_domain_signal(ctx, kind="applications.test.emit_phase", payload=_PhasePayload())
    assert event.phase is ExecutionPhase.STEP_EXECUTION


def test_emit_domain_signal_with_application_hosting_phase() -> None:
    bus = RuntimeEventBus()
    ctx = EmitContext(task_id="task-1", run_id="run-1", bus=bus)
    event = emit_domain_signal(
        ctx,
        kind="applications.test.emit_phase",
        payload=_PhasePayload(),
        phase=ExecutionPhase.APPLICATION_HOSTING,
    )
    assert event.phase is ExecutionPhase.APPLICATION_HOSTING
