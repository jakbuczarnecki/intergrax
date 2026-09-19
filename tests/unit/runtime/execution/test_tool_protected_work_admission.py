# © Artur Czarnecki. All rights reserved.

"""Tool path protected-work admission (HARNESS-02 B2 / Q08 / Q09 / Q16)."""

from __future__ import annotations

import ast
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import BaseModel

from intergrax.runtime.execution.deadline_scope import (
    bind_active_execution_deadline_scope,
    reset_active_execution_deadline_scope,
)
from intergrax.contracts.execution_deadline.projection import ExecutionDeadlineProjection
from intergrax.runtime.cancellation.coordinator import (
    CANCELLATION_REQUESTED_KEY,
)
from intergrax.runtime.execution.protected_work_admission import (
    CanonicalHardProtectedWorkAdmission,
    StaticCancellationView,
)
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    reset_active_execution_identity,
    validate_run_id,
)
from intergrax.tools.execution_models import ToolExecutionRequest
from testing_support.builder import build_runtime_state_for_tests, canonical_run_id_for_tests
from tests.unit.runtime.nexus.tools.conftest import FakeRegistry
from tests.unit.runtime.nexus.tools.test_gr10_r8_orchestration_inner_guard import (
    _CountingExecutor,
    _In,
    _contract,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_INVOKER = _REPO_ROOT / "intergrax" / "runtime" / "nexus" / "tools" / "invoker.py"


def _expired_projection() -> ExecutionDeadlineProjection:
    return ExecutionDeadlineProjection(
        deadline_at_utc=datetime(2020, 1, 1, tzinfo=timezone.utc),
        remaining_seconds=0.0,
        is_expired=True,
        global_deadline_monotonic=1.0,
    )


class _FakeMonotonicClock:
    def __init__(self, value: float = 1.0) -> None:
        self._value = value

    def monotonic(self) -> float:
        return self._value

    def advance(self, seconds: float) -> None:
        self._value += seconds


def test_q09_expired_tool_blocked_before_executor() -> None:
    executor = _CountingExecutor()
    invoker = RuntimeToolInvoker(registry=FakeRegistry(_contract()), executor=executor)
    projection = _expired_projection()
    monotonic = _FakeMonotonicClock(1.0)
    tokens = bind_active_execution_deadline_scope(
        projection=projection,
        admission=CanonicalHardProtectedWorkAdmission(
            projection=projection,
            cancellation_view=StaticCancellationView(cancelled=False),
            monotonic_clock=monotonic,
        ),
        monotonic_clock=monotonic,
    )
    state = build_runtime_state_for_tests(
        run_id=canonical_run_id_for_tests("h02-expired-tool"),
    )
    run_id = canonical_run_id_for_tests("h02-expired-tool")
    request = ToolExecutionRequest(
        run_id=validate_run_id(run_id),
        tool_id="probe.tool",
        step_id="s1",
        input=_In(value=1),
    )
    identity_token = bind_active_execution_identity(
        run_id=validate_run_id(run_id),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )
    try:
        result = invoker.invoke(state=state, agent_id="agent-1", request=request)
    finally:
        reset_active_execution_identity(identity_token)
        reset_active_execution_deadline_scope(*tokens)
    assert executor.calls == 0
    assert result.success is False
    assert result.error is not None
    assert "deadline_exceeded" in result.error.error_message


def test_q08_cancelled_first_attempt_blocked() -> None:
    executor = _CountingExecutor()
    invoker = RuntimeToolInvoker(registry=FakeRegistry(_contract()), executor=executor)
    state = build_runtime_state_for_tests(
        run_id=canonical_run_id_for_tests("h02-cancel-tool"),
    )
    state.request.metadata[CANCELLATION_REQUESTED_KEY] = True
    run_id = canonical_run_id_for_tests("h02-cancel-tool")
    identity_token = bind_active_execution_identity(
        run_id=validate_run_id(run_id),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )
    try:
        result = invoker.invoke(
            state=state,
            agent_id="agent-1",
            request=ToolExecutionRequest(
                run_id=validate_run_id(run_id),
                tool_id="probe.tool",
                step_id="step-1",
                input=_In(value=1),
            ),
        )
    finally:
        reset_active_execution_identity(identity_token)
    assert executor.calls == 0
    assert result.success is False


def test_q16_admission_before_idempotency_claim_structural() -> None:
    source = _INVOKER.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(_INVOKER))
    invoke_methods = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "invoke"
    ]
    assert invoke_methods
    body = ast.get_source_segment(source, invoke_methods[0]) or ""
    admission_pos = body.find("_protected_work_admission_tool_denial")
    idempotency_pos = body.find("_requires_idempotency_coordination")
    assert admission_pos != -1 and idempotency_pos != -1
    assert admission_pos < idempotency_pos
