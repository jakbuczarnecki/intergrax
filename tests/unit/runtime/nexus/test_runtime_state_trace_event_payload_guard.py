# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import pytest

from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.session.in_memory_session_storage import InMemorySessionStorage
from intergrax.runtime.nexus.session.session_manager import SessionManager
from intergrax.runtime.nexus.tracing.trace_models import DiagnosticPayload, TraceComponent, TraceLevel

from tests._support.builder import FakeLLMAdapter


pytestmark = pytest.mark.unit


@dataclass(frozen=True)
class _DummyDiag(DiagnosticPayload):
    value: int

    @classmethod
    def schema_id(cls) -> str:
        return "intergrax.tests.dummy"

    def to_dict(self) -> Dict[str, Any]:
        return {"value": int(self.value)}


@pytest.fixture
def runtime_state() -> RuntimeState:
    config = RuntimeConfig(
        llm_adapter=FakeLLMAdapter(fixed_text="OK"),
        enable_rag=False,
        enable_websearch=False,
        tools_mode="off",
    )

    session_manager = SessionManager(storage=InMemorySessionStorage())

    ctx = RuntimeContext.build(
        config=config,
        session_manager=session_manager,
    )

    request = RuntimeRequest(
        session_id="trace-guard-test-session",
        user_id="trace-guard-test-user",
        message="test",
        attachments=[],
    )

    return RuntimeState(
        context=ctx,
        request=request,
        run_id="trace-guard-test-run-001",
    )


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
