# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

import time

import pytest
from pydantic import BaseModel

from intergrax.runtime.nexus.errors.error_codes import RuntimeErrorCode
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.tools.core.contracts import ToolContract, ToolRetryPolicy, ToolRiskLevel
from intergrax.tools.execution_models import ToolExecutionRequest, ToolExecutionResult
from testing_support.builder import (
    build_runtime_state_for_tests,
    canonical_execution_identity_scope,
    canonical_run_id_for_tests,
)

pytestmark = pytest.mark.unit


class InputModel(BaseModel):
    value: int


class OutputModel(BaseModel):
    result: int


from tests.unit.runtime.nexus.tools.conftest import FakeRegistry


class AllowAllScopePolicy:
    def is_allowed(self, *args, **kwargs) -> bool:
        return True


def test_runtime_tool_invoker_trace_includes_contract_metadata():
    contract = ToolContract(
        tool_id="meta_tool",
        name="meta_tool",
        description="meta",
        input_schema=InputModel,
        output_schema=OutputModel,
        side_effects=False,
        error_mapping={},
        risk_level=ToolRiskLevel.HIGH,
        injects_context=True,
        category="retrieval",
        timeout_ms=5_000,
    )

    class OkExecutor:
        def execute(self, request: ToolExecutionRequest[BaseModel]) -> BaseModel:
            return OutputModel(result=request.input.value)

    invoker = RuntimeToolInvoker(
        registry=FakeRegistry(contract),
        executor=OkExecutor(),
        scope_policy=AllowAllScopePolicy(),
    )
    run_id = canonical_run_id_for_tests("run_meta")
    state = build_runtime_state_for_tests(run_id=run_id)
    request = ToolExecutionRequest(
        run_id=run_id,
        tool_id="meta_tool",
        step_id="1",
        input=InputModel(value=3),
    )

    with canonical_execution_identity_scope(run_id):
        result = invoker.invoke(state=state, agent_id="agent", request=request)

    assert result.success is True
    start = next(e for e in state.trace_events if e.step == "tool_invocation_start")
    assert start.payload.risk_level == "high"
    assert start.payload.injects_context is True
    assert start.payload.category == "retrieval"
    assert start.payload.timeout_ms == 5_000


def test_runtime_tool_invoker_timeout():
    contract = ToolContract(
        tool_id="slow_tool",
        name="slow_tool",
        description="slow",
        input_schema=InputModel,
        output_schema=OutputModel,
        side_effects=False,
        error_mapping={},
        timeout_ms=50,
    )

    class SlowExecutor:
        def execute(self, request: ToolExecutionRequest[BaseModel]) -> BaseModel:
            time.sleep(0.2)
            return OutputModel(result=1)

    invoker = RuntimeToolInvoker(
        registry=FakeRegistry(contract),
        executor=SlowExecutor(),
        scope_policy=AllowAllScopePolicy(),
    )
    run_id = canonical_run_id_for_tests("run_timeout")
    state = build_runtime_state_for_tests(run_id=run_id)
    request = ToolExecutionRequest(
        run_id=run_id,
        tool_id="slow_tool",
        step_id="1",
        input=InputModel(value=1),
    )

    with canonical_execution_identity_scope(run_id):
        result = invoker.invoke(state=state, agent_id="agent", request=request)

    assert result.success is False
    assert result.error is not None
    assert result.error.error_code == RuntimeErrorCode.TIMEOUT


def test_runtime_tool_invoker_retries_then_succeeds():
    contract = ToolContract(
        tool_id="flaky_tool",
        name="flaky_tool",
        description="flaky",
        input_schema=InputModel,
        output_schema=OutputModel,
        side_effects=False,
        error_mapping={},
        retry_policy=ToolRetryPolicy(max_attempts=3, backoff_ms=0),
    )

    calls = {"n": 0}

    class FlakyExecutor:
        def execute(self, request: ToolExecutionRequest[BaseModel]) -> BaseModel:
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("transient")
            return OutputModel(result=42)

    invoker = RuntimeToolInvoker(
        registry=FakeRegistry(contract),
        executor=FlakyExecutor(),
        scope_policy=AllowAllScopePolicy(),
    )
    run_id = canonical_run_id_for_tests("run_retry")
    state = build_runtime_state_for_tests(run_id=run_id)
    request = ToolExecutionRequest(
        run_id=run_id,
        tool_id="flaky_tool",
        step_id="1",
        input=InputModel(value=1),
    )

    with canonical_execution_identity_scope(run_id):
        result = invoker.invoke(state=state, agent_id="agent", request=request)

    assert isinstance(result, ToolExecutionResult)
    assert result.success is True
    assert result.output is not None
    assert result.output.result == 42
    assert calls["n"] == 3
    retry_events = [e for e in state.trace_events if e.step == "tool_invocation_retry"]
    assert len(retry_events) == 2
