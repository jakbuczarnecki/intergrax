# © Artur Czarnecki. All rights reserved.

"""TOOL-ENG-7 — post-tool verify hook tests."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from intergrax.runtime.nexus.engine.runtime_state import ToolCallTrace
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.nexus.tools.tool_verify_hooks import emit_high_risk_tool_verify_signal
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent
from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from testing_support.builder import build_runtime_state_for_tests
from tests.unit.runtime.nexus.tools.conftest import FakeRegistry

pytestmark = pytest.mark.unit


class _In(BaseModel):
    value: int = 1


class _Out(BaseModel):
    result: int = 0


def test_emit_high_risk_verify_signal() -> None:
    contract = ToolContract(
        tool_id="danger.tool",
        name="danger",
        description="dangerous",
        input_schema=_In,
        output_schema=_Out,
        side_effects=True,
        error_mapping={},
        risk_level=ToolRiskLevel.HIGH,
    )
    state = build_runtime_state_for_tests(run_id="run-verify")
    invoker = RuntimeToolInvoker(registry=FakeRegistry(contract), executor=object())  # type: ignore[arg-type]
    trace = ToolCallTrace(
        tool_name="danger.tool",
        arguments={},
        output_preview="{}",
        success=True,
        error_message=None,
        raw_trace={},
    )
    assert emit_high_risk_tool_verify_signal(state=state, invoker=invoker, trace=trace)
    event = next(e for e in state.trace_events if e.step == "tool_verify_required")
    assert event.component == TraceComponent.TOOLS
