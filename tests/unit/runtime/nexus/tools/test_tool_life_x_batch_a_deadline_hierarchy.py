# © Artur Czarnecki. All rights reserved.

"""TOOL-LIFE-X Batch A — canonical deadline hierarchy negative qualification."""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import BaseModel

from intergrax.contracts.dependency_concurrency_admission import (
    DependencyConcurrencyIdentity,
    DependencyConcurrencyKind,
    DependencyConcurrencyOverloadMode,
    DependencyConcurrencyPolicy,
)
from intergrax.contracts.execution_deadline.projection import ExecutionDeadlineProjection
from intergrax.contracts.execution_identity import validate_run_id
from intergrax.runtime.cancellation.coordinator import CANCELLATION_REQUESTED_KEY
from intergrax.runtime.execution.deadline_scope import (
    bind_active_execution_deadline_scope,
    reset_active_execution_deadline_scope,
)
from intergrax.runtime.execution.protected_work_admission import (
    CanonicalHardProtectedWorkAdmission,
    StaticCancellationView,
)
from intergrax.runtime.nexus.errors.error_codes import RuntimeErrorCode
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.resilience.dependency_attempt_execution_boundary import (
    DependencyAttemptExecutionBoundary,
)
from intergrax.runtime.resilience.local_dependency_concurrency_admission import (
    LocalDependencyConcurrencyAdmission,
)
from intergrax.tools.core.contracts import ToolContract, ToolRetryPolicy, ToolRiskLevel
from intergrax.tools.execution_models import ToolEffectCertainty, ToolExecutionRequest
from intergrax.tools.registry import ToolRegistry
from testing_support.builder import (
    build_runtime_state_for_tests,
    canonical_governed_execution_scope,
    canonical_run_id_for_tests,
)
from tests.unit.runtime.nexus.tools.conftest import FakeRegistry

pytestmark = pytest.mark.unit


class _In(BaseModel):
    value: int


class _Out(BaseModel):
    result: int


class _CountingExecutor:
    def __init__(self, *, delay_s: float = 0.0) -> None:
        self.calls = 0
        self._delay_s = delay_s
        self._gate: threading.Event | None = None

    def set_gate(self, gate: threading.Event) -> None:
        self._gate = gate

    def execute(self, request: ToolExecutionRequest[BaseModel]) -> BaseModel:
        self.calls += 1
        if self._gate is not None:
            self._gate.wait(timeout=10.0)
        if self._delay_s > 0:
            time.sleep(self._delay_s)
        return _Out(result=request.input.value)


class _SystemMonotonicClock:
    def monotonic(self) -> float:
        return time.monotonic()


def _contract(
    *,
    tool_id: str = "probe.tool",
    timeout_ms: int = 30_000,
    retry_policy: ToolRetryPolicy | None = None,
) -> ToolContract:
    return ToolContract(
        tool_id=tool_id,
        name=tool_id,
        description=tool_id,
        input_schema=_In,
        output_schema=_Out,
        side_effects=False,
        error_mapping={ConnectionError: RuntimeErrorCode.TOOL_ERROR},
        risk_level=ToolRiskLevel.LOW,
        timeout_ms=timeout_ms,
        retry_policy=retry_policy or ToolRetryPolicy(),
    )


def _deadline_scope(
    *,
    remaining_s: float,
    monotonic: _SystemMonotonicClock,
) -> tuple[object, ...]:
    bound = monotonic.monotonic() + remaining_s
    deadline_at = datetime.now(timezone.utc) + timedelta(seconds=remaining_s)
    projection = ExecutionDeadlineProjection(
        deadline_at_utc=deadline_at,
        remaining_seconds=remaining_s,
        is_expired=remaining_s <= 0,
        global_deadline_monotonic=bound,
    )
    return bind_active_execution_deadline_scope(
        projection=projection,
        admission=CanonicalHardProtectedWorkAdmission(
            projection=projection,
            cancellation_view=StaticCancellationView(cancelled=False),
            monotonic_clock=monotonic,
        ),
        monotonic_clock=monotonic,
    )


def _governance_scope(run_id: str):
    return canonical_governed_execution_scope(
        run_id,
        governance_tenant_id="test-tenant",
        governance_workspace_id="test-workspace",
        governance_principal_id="test-principal",
    )


def _invoke(
    invoker: RuntimeToolInvoker,
    *,
    run_id: str,
    tool_id: str = "probe.tool",
    deadline_tokens: tuple[object, ...] | None = None,
) -> object:
    state = build_runtime_state_for_tests(run_id=run_id)
    with _governance_scope(run_id):
        try:
            return invoker.invoke(
                state=state,
                agent_id="agent-1",
                request=ToolExecutionRequest(
                    run_id=validate_run_id(run_id),
                    tool_id=tool_id,
                    step_id="s1",
                    input=_In(value=1),
                ),
            )
        finally:
            if deadline_tokens is not None:
                reset_active_execution_deadline_scope(*deadline_tokens)


def test_parent_expired_before_tool_invocation() -> None:
    executor = _CountingExecutor()
    invoker = RuntimeToolInvoker(registry=FakeRegistry(_contract()), executor=executor)
    monotonic = _SystemMonotonicClock()
    tokens = _deadline_scope(remaining_s=0.0, monotonic=monotonic)
    result = _invoke(
        invoker,
        run_id=canonical_run_id_for_tests("tlx-a-expired-before"),
        deadline_tokens=tokens,
    )
    assert executor.calls == 0
    assert result.success is False
    assert result.error is not None
    assert "deadline_exceeded" in result.error.error_message


def test_parent_budget_shorter_than_tool_timeout() -> None:
    executor = _CountingExecutor(delay_s=0.5)
    contract = _contract(timeout_ms=60_000)
    invoker = RuntimeToolInvoker(registry=FakeRegistry(contract), executor=executor)
    monotonic = _SystemMonotonicClock()
    tokens = _deadline_scope(remaining_s=0.12, monotonic=monotonic)
    started = time.monotonic()
    result = _invoke(
        invoker,
        run_id=canonical_run_id_for_tests("tlx-a-parent-shorter"),
        deadline_tokens=tokens,
    )
    elapsed = time.monotonic() - started
    assert executor.calls == 1
    assert result.success is False
    assert result.error is not None
    assert result.error.error_code == RuntimeErrorCode.TIMEOUT
    assert elapsed < 0.45


def test_tool_timeout_shorter_than_parent_budget() -> None:
    executor = _CountingExecutor(delay_s=0.25)
    contract = _contract(timeout_ms=80)
    invoker = RuntimeToolInvoker(registry=FakeRegistry(contract), executor=executor)
    monotonic = _SystemMonotonicClock()
    tokens = _deadline_scope(remaining_s=5.0, monotonic=monotonic)
    started = time.monotonic()
    result = _invoke(
        invoker,
        run_id=canonical_run_id_for_tests("tlx-a-tool-shorter"),
        deadline_tokens=tokens,
    )
    elapsed = time.monotonic() - started
    assert result.error is not None
    assert result.error.error_code == RuntimeErrorCode.TIMEOUT
    assert elapsed < 0.45


def test_retry_backoff_bounded_by_remaining_parent_budget() -> None:
    class _FlakyExecutor:
        def __init__(self) -> None:
            self.calls = 0

        def execute(self, request: ToolExecutionRequest[BaseModel]) -> BaseModel:
            self.calls += 1
            if self.calls == 1:
                raise ConnectionError("transient")
            return _Out(result=1)

    contract = _contract(
        retry_policy=ToolRetryPolicy(max_attempts=2, backoff_ms=5_000),
    )
    invoker = RuntimeToolInvoker(registry=FakeRegistry(contract), executor=_FlakyExecutor())
    monotonic = _SystemMonotonicClock()
    tokens = _deadline_scope(remaining_s=0.06, monotonic=monotonic)
    started = time.monotonic()
    result = _invoke(
        invoker,
        run_id=canonical_run_id_for_tests("tlx-a-backoff-bounded"),
        deadline_tokens=tokens,
    )
    elapsed = time.monotonic() - started
    assert result.success is False
    assert result.error is not None
    assert "deadline_exceeded" in result.error.error_message
    assert elapsed < 0.35


def test_queue_wait_bounded_by_remaining_parent_budget() -> None:
    registry = ToolRegistry()
    tool_id = "slow.tool"
    registry.register(_contract(tool_id=tool_id, timeout_ms=5_000), _CountingExecutor())
    executor = _CountingExecutor()
    gate = threading.Event()
    executor.set_gate(gate)
    policies = {
        DependencyConcurrencyIdentity(
            kind=DependencyConcurrencyKind.TOOL,
            value=tool_id,
        ): DependencyConcurrencyPolicy(
            max_concurrent_calls=1,
            overload_mode=DependencyConcurrencyOverloadMode.WAIT_WITH_TIMEOUT,
            wait_timeout_seconds=30.0,
        ),
    }
    admission = LocalDependencyConcurrencyAdmission(policies)
    boundary = DependencyAttemptExecutionBoundary(admission)
    invoker = RuntimeToolInvoker(
        registry=registry,
        executor=executor,
        dependency_attempt_boundary=boundary,
    )
    run_id = canonical_run_id_for_tests("tlx-a-queue-wait")
    state = build_runtime_state_for_tests(run_id=run_id)
    holder_state = build_runtime_state_for_tests(run_id=run_id)
    monotonic = _SystemMonotonicClock()
    tokens = _deadline_scope(remaining_s=0.15, monotonic=monotonic)
    holder_started = threading.Event()
    holder_done = threading.Event()

    def _hold_slot() -> None:
        with _governance_scope(run_id):
            holder_started.set()
            invoker.invoke(
                state=holder_state,
                agent_id="agent-1",
                request=ToolExecutionRequest(
                    run_id=validate_run_id(run_id),
                    tool_id=tool_id,
                    step_id="hold",
                    input=_In(value=1),
                ),
            )
            holder_done.set()

    holder = threading.Thread(target=_hold_slot, daemon=True)
    holder.start()
    assert holder_started.wait(timeout=2.0)
    time.sleep(0.05)
    with _governance_scope(run_id):
        result = invoker.invoke(
            state=state,
            agent_id="agent-1",
            request=ToolExecutionRequest(
                run_id=validate_run_id(run_id),
                tool_id=tool_id,
                step_id="waiter",
                input=_In(value=2),
            ),
        )
    gate.set()
    holder.join(timeout=5.0)
    reset_active_execution_deadline_scope(*tokens)
    invoker.close()
    assert result.success is False
    assert result.error is not None
    assert result.error.error_code == RuntimeErrorCode.DEPENDENCY_ERROR
    assert result.effect_certainty == ToolEffectCertainty.NOT_STARTED


def test_cancelled_during_queue_wait_path() -> None:
    executor = _CountingExecutor()
    invoker = RuntimeToolInvoker(registry=FakeRegistry(_contract()), executor=executor)
    state = build_runtime_state_for_tests(
        run_id=canonical_run_id_for_tests("tlx-a-cancel-queue"),
    )
    state.request.metadata[CANCELLATION_REQUESTED_KEY] = True
    run_id = canonical_run_id_for_tests("tlx-a-cancel-queue")
    with _governance_scope(run_id):
        result = invoker.invoke(
            state=state,
            agent_id="agent-1",
            request=ToolExecutionRequest(
                run_id=validate_run_id(run_id),
                tool_id="probe.tool",
                step_id="s1",
                input=_In(value=1),
            ),
        )
    assert executor.calls == 0
    assert result.success is False


def test_without_deadline_scope_uses_contract_timeout() -> None:
    executor = _CountingExecutor(delay_s=0.05)
    contract = _contract(timeout_ms=2_000)
    invoker = RuntimeToolInvoker(registry=FakeRegistry(contract), executor=executor)
    result = _invoke(invoker, run_id=canonical_run_id_for_tests("tlx-a-no-scope"))
    assert result.success is True
    assert executor.calls == 1


def test_queue_wait_then_physical_timeout_uses_remaining_not_contract() -> None:
    """BA-F01: post-admission timeout must not reset to full ToolContract timeout."""
    registry = ToolRegistry()
    tool_id = "contended.tool"
    gate = threading.Event()

    class _QueueThenPhysicalExecutor(_CountingExecutor):
        def execute(self, request: ToolExecutionRequest[BaseModel]) -> BaseModel:
            self.calls += 1
            if self._gate is not None and self.calls == 1:
                self._gate.wait(timeout=10.0)
            if str(request.step_id) == "waiter":
                time.sleep(0.4)
            return _Out(result=request.input.value)

    executor = _QueueThenPhysicalExecutor()
    executor.set_gate(gate)
    registry.register(_contract(tool_id=tool_id, timeout_ms=60_000), _CountingExecutor())
    policies = {
        DependencyConcurrencyIdentity(
            kind=DependencyConcurrencyKind.TOOL,
            value=tool_id,
        ): DependencyConcurrencyPolicy(
            max_concurrent_calls=1,
            overload_mode=DependencyConcurrencyOverloadMode.WAIT_WITH_TIMEOUT,
            wait_timeout_seconds=30.0,
        ),
    }
    boundary = DependencyAttemptExecutionBoundary(
        LocalDependencyConcurrencyAdmission(policies),
    )
    invoker = RuntimeToolInvoker(
        registry=registry,
        executor=executor,
        dependency_attempt_boundary=boundary,
    )
    run_id = canonical_run_id_for_tests("tlx-a-queue-then-physical")
    state = build_runtime_state_for_tests(run_id=run_id)
    holder_state = build_runtime_state_for_tests(run_id=run_id)
    monotonic = _ManualMonotonicClock(100.0)
    tokens = _deadline_scope(remaining_s=0.45, monotonic=monotonic)
    holder_started = threading.Event()
    holder_done = threading.Event()

    def _hold() -> None:
        with _governance_scope(run_id):
            holder_started.set()
            invoker.invoke(
                state=holder_state,
                agent_id="agent-1",
                request=ToolExecutionRequest(
                    run_id=validate_run_id(run_id),
                    tool_id=tool_id,
                    step_id="hold",
                    input=_In(value=1),
                ),
            )
            holder_done.set()

    holder = threading.Thread(target=_hold, daemon=True)
    holder.start()
    assert holder_started.wait(timeout=2.0)
    monotonic.advance(0.33)

    def _open_gate_for_waiter() -> None:
        time.sleep(0.05)
        gate.set()

    threading.Thread(target=_open_gate_for_waiter, daemon=True).start()
    started = time.monotonic()
    with _governance_scope(run_id):
        result = invoker.invoke(
            state=state,
            agent_id="agent-1",
            request=ToolExecutionRequest(
                run_id=validate_run_id(run_id),
                tool_id=tool_id,
                step_id="waiter",
                input=_In(value=2),
            ),
        )
    elapsed = time.monotonic() - started
    holder_done.wait(timeout=5.0)
    reset_active_execution_deadline_scope(*tokens)
    invoker.close()
    assert executor.calls == 2
    assert result.success is False
    assert result.error is not None
    assert result.error.error_code == RuntimeErrorCode.TIMEOUT
    assert elapsed < 0.35


class _ManualMonotonicClock:
    def __init__(self, start: float) -> None:
        self._t = start

    def monotonic(self) -> float:
        return self._t

    def advance(self, delta: float) -> None:
        self._t += delta


def test_deadline_exhausted_after_permit_before_physical_submit() -> None:
    """BA-F01: no ToolExecutor submit when parent budget is gone after admission."""
    registry = ToolRegistry()
    tool_id = "gated.tool"
    registry.register(_contract(tool_id=tool_id, timeout_ms=30_000), _CountingExecutor())
    executor = _CountingExecutor()
    gate = threading.Event()
    executor.set_gate(gate)
    policies = {
        DependencyConcurrencyIdentity(
            kind=DependencyConcurrencyKind.TOOL,
            value=tool_id,
        ): DependencyConcurrencyPolicy(
            max_concurrent_calls=1,
            overload_mode=DependencyConcurrencyOverloadMode.WAIT_WITH_TIMEOUT,
            wait_timeout_seconds=30.0,
        ),
    }
    admission = LocalDependencyConcurrencyAdmission(policies)
    boundary = DependencyAttemptExecutionBoundary(admission)
    invoker = RuntimeToolInvoker(
        registry=registry,
        executor=executor,
        dependency_attempt_boundary=boundary,
    )
    run_id = canonical_run_id_for_tests("tlx-a-post-acquire-deadline")
    state = build_runtime_state_for_tests(run_id=run_id)
    holder_state = build_runtime_state_for_tests(run_id=run_id)
    clock = _ManualMonotonicClock(100.0)
    tokens = _deadline_scope(remaining_s=0.5, monotonic=clock)
    release_holder = threading.Event()
    holder_done = threading.Event()

    def _hold() -> None:
        with _governance_scope(run_id):
            invoker.invoke(
                state=holder_state,
                agent_id="agent-1",
                request=ToolExecutionRequest(
                    run_id=validate_run_id(run_id),
                    tool_id=tool_id,
                    step_id="hold",
                    input=_In(value=1),
                ),
            )
            release_holder.wait(timeout=5.0)
            holder_done.set()

    holder = threading.Thread(target=_hold, daemon=True)
    holder.start()
    time.sleep(0.05)
    clock.advance(0.55)
    gate.set()
    release_holder.set()
    holder_done.wait(timeout=5.0)
    with _governance_scope(run_id):
        result = invoker.invoke(
            state=state,
            agent_id="agent-1",
            request=ToolExecutionRequest(
                run_id=validate_run_id(run_id),
                tool_id=tool_id,
                step_id="waiter",
                input=_In(value=2),
            ),
        )
    reset_active_execution_deadline_scope(*tokens)
    invoker.close()
    assert executor.calls == 1
    assert result.success is False
    assert result.error is not None
    assert "deadline_exceeded" in result.error.error_message
    assert result.effect_certainty == ToolEffectCertainty.NOT_STARTED
