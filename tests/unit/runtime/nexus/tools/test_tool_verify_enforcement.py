# © Artur Czarnecki. All rights reserved.

"""TOOL-ENG-7 — high-risk tool verify enforcement."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from intergrax.runtime.nexus.engine.runtime_state import ToolCallTrace
from intergrax.runtime.nexus.errors.tool_verification_required_error import (
    ToolVerificationRequiredError,
)
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.nexus.tools.tool_verify_hooks import run_post_tool_verify
from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from testing_support.builder import build_runtime_state_for_tests
from tests.unit.runtime.nexus.tools.conftest import FakeRegistry

pytestmark = pytest.mark.unit


class _In(BaseModel):
    value: int = 1


class _Out(BaseModel):
    result: int = 0


def _high_risk_contract() -> ToolContract:
    return ToolContract(
        tool_id="danger.tool",
        name="danger",
        description="dangerous",
        input_schema=_In,
        output_schema=_Out,
        side_effects=True,
        error_mapping={},
        risk_level=ToolRiskLevel.HIGH,
    )


def test_high_risk_tool_blocked_when_enforced() -> None:
    state = build_runtime_state_for_tests(run_id="run-block")
    state.context.config.enforce_high_risk_tool_verify = True
    invoker = RuntimeToolInvoker(registry=FakeRegistry(_high_risk_contract()), executor=object())  # type: ignore[arg-type]
    trace = ToolCallTrace(
        tool_name="danger.tool",
        arguments={},
        output_preview="{}",
        success=True,
        error_message=None,
        raw_trace={},
    )
    with pytest.raises(ToolVerificationRequiredError):
        run_post_tool_verify(state=state, invoker=invoker, trace=trace)


def test_high_risk_tool_allowed_with_explicit_approval() -> None:
    state = build_runtime_state_for_tests(run_id="run-approve")
    state.context.config.enforce_high_risk_tool_verify = True
    state.high_risk_tool_approvals = frozenset({"danger.tool"})
    invoker = RuntimeToolInvoker(registry=FakeRegistry(_high_risk_contract()), executor=object())  # type: ignore[arg-type]
    trace = ToolCallTrace(
        tool_name="danger.tool",
        arguments={},
        output_preview="{}",
        success=True,
        error_message=None,
        raw_trace={},
    )
    run_post_tool_verify(state=state, invoker=invoker, trace=trace)
