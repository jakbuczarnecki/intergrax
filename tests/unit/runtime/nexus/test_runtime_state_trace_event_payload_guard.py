# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import pytest

from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.tracing.session.session_consolidation_diag import SessionConsolidationDiagV1
from intergrax.runtime.nexus.tracing.trace_models import DiagnosticPayload, TraceComponent, TraceLevel

from testing_support.builder import (
    FakeLLMAdapter,
    build_runtime_state_for_tests,
    canonical_execution_identity_scope,
    canonical_run_id_for_tests,
)


pytestmark = pytest.mark.unit


@dataclass(frozen=True)
class _DummyDiag(DiagnosticPayload):
    value: int

    def redact(self) -> SessionConsolidationDiagV1:        
        return self

    @classmethod
    def schema_id(cls) -> str:
        return "intergrax.tests.dummy"

    def to_dict(self) -> Dict[str, Any]:
        return {"value": int(self.value)}


_TRACE_GUARD_SEED = "trace-guard-test"


@pytest.fixture
def runtime_state() -> RuntimeState:
    state = build_runtime_state_for_tests(run_id=_TRACE_GUARD_SEED)
    state.context.config.llm_adapter = FakeLLMAdapter(fixed_text="OK")
    with canonical_execution_identity_scope(_TRACE_GUARD_SEED):
        yield state


def test_trace_event_rejects_non_diagnostic_payload(runtime_state: RuntimeState) -> None:
    with pytest.raises(TypeError) as exc:
        runtime_state.trace_event(
            component=TraceComponent.ENGINE,
            step="guard",
            message="should fail",
            level=TraceLevel.DEBUG,
            payload={"x": 1},  # type: ignore[arg-type]
        )

    assert "DiagnosticPayload" in str(exc.value)


def test_trace_event_accepts_diagnostic_payload(runtime_state: RuntimeState) -> None:
    runtime_state.trace_event(
        component=TraceComponent.ENGINE,
        step="guard",
        message="should pass",
        level=TraceLevel.DEBUG,
        payload=_DummyDiag(value=1),
    )

def test_trace_event_appends_event_and_increments_seq(runtime_state: RuntimeState) -> None:
    """
    trace_event() must append events to state.trace_events and increment seq deterministically.
    """
    assert runtime_state.trace_events == []

    runtime_state.trace_event(
        component=TraceComponent.ENGINE,
        step="s1",
        message="m1",
        level=TraceLevel.INFO,
        payload=None,
    )
    runtime_state.trace_event(
        component=TraceComponent.ENGINE,
        step="s2",
        message="m2",
        level=TraceLevel.DEBUG,
        payload=None,
    )

    assert len(runtime_state.trace_events) == 2

    e1 = runtime_state.trace_events[0]
    e2 = runtime_state.trace_events[1]

    assert e1.seq == 1
    assert e2.seq == 2

    assert e1.event_id != e2.event_id


def test_trace_event_sets_run_id_component_level_and_message(runtime_state: RuntimeState) -> None:
    """
    trace_event() must propagate run_id and store the provided fields exactly.
    """
    runtime_state.trace_event(
        component=TraceComponent.ENGINE,
        step="guard",
        message="hello",
        level=TraceLevel.WARNING,
        payload=None,
    )

    ev = runtime_state.trace_events[-1]
    assert ev.run_id == str(canonical_run_id_for_tests(_TRACE_GUARD_SEED))
    assert ev.component == TraceComponent.ENGINE
    assert ev.level == TraceLevel.WARNING
    assert ev.step == "guard"
    assert ev.message == "hello"


def test_trace_event_ts_utc_is_timezone_aware_iso_utc(runtime_state: RuntimeState) -> None:
    """
    ts_utc must be an ISO-8601 timestamp and timezone-aware (UTC).
    """
    from datetime import datetime, timezone

    runtime_state.trace_event(
        component=TraceComponent.ENGINE,
        step="ts",
        message="check",
        level=TraceLevel.INFO,
        payload=None,
    )

    ts = runtime_state.trace_events[-1].ts_utc
    dt = datetime.fromisoformat(ts)

    assert dt.tzinfo is not None
    assert dt.tzinfo.utcoffset(dt) == timezone.utc.utcoffset(dt)


def test_trace_event_stores_payload_instance(runtime_state: RuntimeState) -> None:
    """
    Payload must be stored as the DiagnosticPayload instance (no conversion at this layer).
    """
    payload = _DummyDiag(value=42)

    runtime_state.trace_event(
        component=TraceComponent.ENGINE,
        step="payload",
        message="check",
        level=TraceLevel.INFO,
        payload=payload,
    )

    ev = runtime_state.trace_events[-1]
    assert ev.payload is payload
