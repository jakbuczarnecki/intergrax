# © Artur Czarnecki. All rights reserved.

"""Neutral CodeCraft tool error mapping (HARNESS-01-R5-W3-CLOSE-B2-R1)."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from intergrax.contracts.codecraft.bound_capability_execution import (
    CodeCraftBoundCapabilityExecutionOutcome,
)
from intergrax.runtime.codecraft.wiring_bound_capability_execution import (
    _map_tool_execution_result,
)
from intergrax.tools.execution_models import ToolExecutionError, ToolExecutionResult
from intergrax.tools.providers.sandbox.contracts import SandboxExecOutput

pytestmark = pytest.mark.unit


def _fail(code: str, message: str) -> ToolExecutionResult[BaseModel]:
    return ToolExecutionResult(
        success=False,
        output=None,
        error=ToolExecutionError(error_code=code, error_message=message),
    )


def test_permission_error_with_message() -> None:
    result = _map_tool_execution_result(_fail("permission_error", "denied"))
    assert result.outcome is CodeCraftBoundCapabilityExecutionOutcome.REJECTED
    assert result.reason_detail == "denied"


def test_permission_error_empty_message_no_name_error() -> None:
    result = _map_tool_execution_result(_fail("permission_error", ""))
    assert result.outcome is CodeCraftBoundCapabilityExecutionOutcome.REJECTED
    assert result.reason_detail == "permission_error"


def test_policy_error_empty_message() -> None:
    result = _map_tool_execution_result(_fail("policy_error", ""))
    assert result.outcome is CodeCraftBoundCapabilityExecutionOutcome.REJECTED
    assert result.reason_detail == "policy_error"


def test_validation_error_empty_message() -> None:
    result = _map_tool_execution_result(_fail("validation_error", ""))
    assert result.outcome is CodeCraftBoundCapabilityExecutionOutcome.FAILED
    assert result.reason_detail == "validation_error"


def test_unknown_error_empty_message() -> None:
    result = _map_tool_execution_result(_fail("vendor_specific_failure", ""))
    assert result.outcome is CodeCraftBoundCapabilityExecutionOutcome.FAILED
    assert result.reason_detail == "vendor_specific_failure"


def test_unknown_error_message_overrides_code() -> None:
    result = _map_tool_execution_result(
        _fail("vendor_specific_failure", "provider failed"),
    )
    assert result.outcome is CodeCraftBoundCapabilityExecutionOutcome.FAILED
    assert result.reason_detail == "provider failed"


def test_sandbox_success_path() -> None:
    output = SandboxExecOutput(success=True, output={"stdout": "ok"})
    tool_result = ToolExecutionResult.ok(output)
    result = _map_tool_execution_result(tool_result)
    assert result.outcome is CodeCraftBoundCapabilityExecutionOutcome.SUCCEEDED


def test_sandbox_unsuccessful_output() -> None:
    output = SandboxExecOutput(success=False, output={}, error="boom")
    tool_result = ToolExecutionResult.ok(output)
    result = _map_tool_execution_result(tool_result)
    assert result.outcome is CodeCraftBoundCapabilityExecutionOutcome.FAILED
