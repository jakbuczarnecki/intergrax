# © Artur Czarnecki. All rights reserved.

"""HARNESS-02-R1B live deadline and dynamic cancellation proofs (Q23–Q27)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pytest

from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_deadline.admission import (
    ExecutionProtectedWorkAdmissionResult,
)
from intergrax.contracts.execution_deadline.projection import ExecutionDeadlineProjection
from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    reset_active_execution_identity,
)
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.base.base_llm_adapter import BaseLLMAdapter
from intergrax.runtime.execution.active_execution_budget import (
    ActiveExecutionBudgetState,
    bind_active_execution_budget,
    reset_active_execution_budget,
)
from intergrax.runtime.execution.boundary import ExecutionBoundary, ExecutionIdentityBinding
from intergrax.runtime.execution.budget.ledger import create_execution_budget_ledger
from intergrax.runtime.execution.budget.models import ExecutionBudgetAllocationMode
from intergrax.runtime.execution.child import ChildExecutionRunner
from intergrax.runtime.execution.deadline_provider_guard import (
    ExecutionProtectedWorkDeniedError,
    resolve_active_provider_timeout_seconds,
)
from intergrax.runtime.execution.deadline_scope import (
    bind_active_execution_deadline_scope,
    peek_active_execution_deadline_projection,
    reset_active_execution_deadline_scope,
)
from intergrax.runtime.execution.deadline_scope import (
    peek_active_execution_protected_work_admission,
)
from intergrax.runtime.execution.protected_work_admission import (
    CanonicalHardProtectedWorkAdmission,
    StaticCancellationView,
    narrow_protected_work_admission_for_child,
)
from intergrax.runtime.governance.active_execution_authority import (
    bind_active_execution_authority,
    reset_active_execution_authority,
)
from intergrax.runtime.nexus.budget.budget_models import RunBudget
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.contracts.execution_identity import validate_run_id
from testing_support.builder import build_runtime_state_for_tests, canonical_run_id_for_tests
from tests.unit.runtime.nexus.tools.conftest import FakeRegistry
from tests.unit.runtime.nexus.tools.test_gr10_r8_orchestration_inner_guard import (
    _CountingExecutor,
    _In,
    _contract,
)

pytestmark = pytest.mark.unit


class _FakeMonotonicClock:
    def __init__(self, value: float = 100.0) -> None:
        self._value = value

    def monotonic(self) -> float:
        return self._value

    def advance(self, seconds: float) -> None:
        self._value += seconds


class _FakeUtcClock:
    def __init__(self, now: datetime) -> None:
        self._now = now

    def now_utc(self) -> datetime:
        return self._now


@dataclass
class _MutableCancellationView:
    cancelled: bool = False
    reason: str | None = None

    def is_cancelled(self) -> bool:
        return self.cancelled

    def cancellation_reason(self) -> str | None:
        return self.reason


def _bind_live_scope(
    *,
    monotonic: _FakeMonotonicClock,
    deadline_monotonic: float,
    deadline_at_utc: datetime | None = None,
    cancellation_view: _MutableCancellationView | StaticCancellationView | None = None,
) -> tuple:
    remaining = max(0.0, deadline_monotonic - monotonic.monotonic())
    projection = ExecutionDeadlineProjection(
        deadline_at_utc=deadline_at_utc,
        remaining_seconds=remaining,
        is_expired=remaining <= 0,
        global_deadline_monotonic=deadline_monotonic,
    )
    cancel = cancellation_view or StaticCancellationView(cancelled=False)
    admission = CanonicalHardProtectedWorkAdmission(
        projection=projection,
        cancellation_view=cancel,
        monotonic_clock=monotonic,
    )
    return bind_active_execution_deadline_scope(
        projection=projection,
        admission=admission,
        monotonic_clock=monotonic,
    )


def test_q23_deadline_crossed_after_bind_blocks_admission() -> None:
    monotonic = _FakeMonotonicClock(0.0)
    tokens = _bind_live_scope(monotonic=monotonic, deadline_monotonic=5.0)
    admission = CanonicalHardProtectedWorkAdmission(
        projection=peek_active_execution_deadline_projection(),
        cancellation_view=StaticCancellationView(cancelled=False),
        monotonic_clock=monotonic,
    )
    assert (
        admission.assert_can_start_protected_work()
        is ExecutionProtectedWorkAdmissionResult.AVAILABLE
    )
    monotonic.advance(6.0)
    assert (
        admission.assert_can_start_protected_work()
        is ExecutionProtectedWorkAdmissionResult.EXPIRED
    )
    reset_active_execution_deadline_scope(*tokens)


def test_q24_provider_timeout_uses_live_remaining_not_bind_snapshot() -> None:
    monotonic = _FakeMonotonicClock(100.0)
    tokens = _bind_live_scope(
        monotonic=monotonic,
        deadline_monotonic=110.0,
        deadline_at_utc=datetime(2026, 1, 1, 0, 0, 10, tzinfo=timezone.utc),
    )
    assert resolve_active_provider_timeout_seconds(30.0) == 10.0
    monotonic.advance(7.0)
    assert resolve_active_provider_timeout_seconds(30.0) == 3.0
    reset_active_execution_deadline_scope(*tokens)


@pytest.mark.asyncio
async def test_q25_bounded_child_under_unbounded_parent_gets_effective_deadline() -> None:
    @dataclass(frozen=True)
    class Ping:
        pass

    @dataclass(frozen=True)
    class Pong:
        deadline: datetime | None

    now = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
    monotonic = _FakeMonotonicClock(500.0)
    utc = _FakeUtcClock(now)
    unbounded_parent = ExecutionDeadlineProjection(
        deadline_at_utc=None,
        remaining_seconds=float("inf"),
        is_expired=False,
        global_deadline_monotonic=None,
    )
    tokens = bind_active_execution_deadline_scope(
        projection=unbounded_parent,
        admission=CanonicalHardProtectedWorkAdmission(
            projection=unbounded_parent,
            cancellation_view=StaticCancellationView(cancelled=False),
            monotonic_clock=monotonic,
        ),
        monotonic_clock=monotonic,
    )
    ledger = create_execution_budget_ledger(RunBudget())
    child_runner = ChildExecutionRunner[Ping, Pong](
        ledger=ledger,
        utc_clock=utc,
        monotonic_clock=monotonic,
    )
    observed: list[ExecutionDeadlineProjection | None] = []

    class ChildDelegate:
        async def execute(self, request: Ping) -> Pong:
            proj = peek_active_execution_deadline_projection()
            observed.append(proj)
            return Pong(deadline=proj.deadline_at_utc if proj else None)

    root = ExecutionIdentityBinding(
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )

    class RootDelegate:
        async def execute(self, request: Ping) -> Pong:
            budget_token = bind_active_execution_budget(
                ActiveExecutionBudgetState(
                    execution_id=root.execution_id,
                    mode=ExecutionBudgetAllocationMode.SHARED,
                    ledger=ledger,
                ),
            )
            try:
                return await child_runner.execute(
                    request=request,
                    delegate=ChildDelegate(),
                    requested_budget=RunBudget(max_wall_time_seconds=10.0),
                )
            finally:
                reset_active_execution_budget(budget_token)

    authority_token = bind_active_execution_authority(
        ParentExecutionAuthority.unrestricted_root(),
    )
    try:
        boundary = ExecutionBoundary[Ping, Pong](
            RootDelegate(),
            identity=root,
            authority=ParentExecutionAuthority.unrestricted_root(),
        )
        await boundary.execute(Ping())
        assert observed[0] is not None
        assert observed[0].deadline_at_utc is not None
        assert 9.0 <= observed[0].remaining_seconds <= 10.5
    finally:
        reset_active_execution_authority(authority_token)
        reset_active_execution_deadline_scope(*tokens)


def test_q26_cancellation_after_start_blocks_llm_sync() -> None:
    monotonic = _FakeMonotonicClock(0.0)
    cancel = _MutableCancellationView(cancelled=False)
    tokens = _bind_live_scope(
        monotonic=monotonic,
        deadline_monotonic=100.0,
        cancellation_view=cancel,
    )
    physical_calls = 0

    class _ProbeAdapter(BaseLLMAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.provider = "openai"

        @property
        def context_window_tokens(self) -> int:
            return 8192

        def generate_messages(self, messages):  # type: ignore[no-untyped-def]
            del messages
            return self._execute(lambda: "ok")

    adapter = _ProbeAdapter()
    cancel.cancelled = True
    def _physical() -> str:
        nonlocal physical_calls
        physical_calls += 1
        return "ok"

    try:
        with pytest.raises(ExecutionProtectedWorkDeniedError):
            adapter._execute(_physical)
        assert physical_calls == 0
    finally:
        reset_active_execution_deadline_scope(*tokens)


def test_q27_child_narrowing_preserves_live_parent_cancellation() -> None:
    monotonic = _FakeMonotonicClock(0.0)
    parent_cancel = _MutableCancellationView(cancelled=False)
    parent_projection = ExecutionDeadlineProjection(
        deadline_at_utc=None,
        remaining_seconds=float("inf"),
        is_expired=False,
        global_deadline_monotonic=None,
    )
    parent_admission = CanonicalHardProtectedWorkAdmission(
        projection=parent_projection,
        cancellation_view=parent_cancel,
        monotonic_clock=monotonic,
    )
    child_projection = ExecutionDeadlineProjection(
        deadline_at_utc=datetime(2026, 1, 1, 1, 0, tzinfo=timezone.utc),
        remaining_seconds=60.0,
        is_expired=False,
        global_deadline_monotonic=60.0,
    )
    narrowed = narrow_protected_work_admission_for_child(
        child_projection,
        parent_admission,
        monotonic_clock=monotonic,
    )
    assert (
        narrowed.assert_can_start_protected_work()
        is ExecutionProtectedWorkAdmissionResult.AVAILABLE
    )
    parent_cancel.cancelled = True
    assert (
        narrowed.assert_can_start_protected_work()
        is ExecutionProtectedWorkAdmissionResult.CANCELLED
    )


def test_tool_blocked_after_deadline_crossed_post_bind() -> None:
    monotonic = _FakeMonotonicClock(0.0)
    tokens = _bind_live_scope(monotonic=monotonic, deadline_monotonic=5.0)
    monotonic.advance(6.0)
    executor = _CountingExecutor()
    invoker = RuntimeToolInvoker(registry=FakeRegistry(_contract()), executor=executor)
    run_id = canonical_run_id_for_tests("h02-live-expired-tool")
    state = build_runtime_state_for_tests(run_id=run_id)
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
                step_id="s1",
                input=_In(value=1),
            ),
        )
    finally:
        reset_active_execution_identity(identity_token)
        reset_active_execution_deadline_scope(*tokens)
    assert executor.calls == 0
    assert result.success is False


def test_q18_streaming_deadline_crossed_after_bind() -> None:
    monotonic = _FakeMonotonicClock(0.0)
    tokens = _bind_live_scope(monotonic=monotonic, deadline_monotonic=5.0)
    monotonic.advance(6.0)
    factory_calls = 0

    class _ProbeAdapter(BaseLLMAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.provider = "openai"

        @property
        def context_window_tokens(self) -> int:
            return 8192

        def generate_messages(self, messages):  # type: ignore[no-untyped-def]
            del messages
            raise NotImplementedError

    adapter = _ProbeAdapter()

    def _factory():
        nonlocal factory_calls
        factory_calls += 1
        return iter(())

    try:
        with pytest.raises(ExecutionProtectedWorkDeniedError):
            list(adapter._execute_streaming(_factory))
        assert factory_calls == 0
    finally:
        reset_active_execution_deadline_scope(*tokens)


@pytest.mark.asyncio
async def test_child_live_expiry_under_unbounded_parent() -> None:
    @dataclass(frozen=True)
    class Ping:
        pass

    @dataclass(frozen=True)
    class Pong:
        ok: bool

    now = datetime(2026, 6, 1, tzinfo=timezone.utc)
    monotonic = _FakeMonotonicClock(0.0)
    utc = _FakeUtcClock(now)
    unbounded_parent = ExecutionDeadlineProjection(
        deadline_at_utc=None,
        remaining_seconds=float("inf"),
        is_expired=False,
        global_deadline_monotonic=None,
    )
    tokens = bind_active_execution_deadline_scope(
        projection=unbounded_parent,
        admission=CanonicalHardProtectedWorkAdmission(
            projection=unbounded_parent,
            cancellation_view=StaticCancellationView(cancelled=False),
            monotonic_clock=monotonic,
        ),
        monotonic_clock=monotonic,
    )
    ledger = create_execution_budget_ledger(RunBudget())
    child_runner = ChildExecutionRunner[Ping, Pong](
        ledger=ledger,
        utc_clock=utc,
        monotonic_clock=monotonic,
    )

    class ChildDelegate:
        async def execute(self, request: Ping) -> Pong:
            monotonic.advance(6.0)
            port = peek_active_execution_protected_work_admission()
            assert port is not None
            assert (
                port.assert_can_start_protected_work()
                is ExecutionProtectedWorkAdmissionResult.EXPIRED
            )
            return Pong(ok=True)

    root = ExecutionIdentityBinding(
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )

    class RootDelegate:
        async def execute(self, request: Ping) -> Pong:
            budget_token = bind_active_execution_budget(
                ActiveExecutionBudgetState(
                    execution_id=root.execution_id,
                    mode=ExecutionBudgetAllocationMode.SHARED,
                    ledger=ledger,
                ),
            )
            try:
                return await child_runner.execute(
                    request=request,
                    delegate=ChildDelegate(),
                    requested_budget=RunBudget(max_wall_time_seconds=5.0),
                )
            finally:
                reset_active_execution_budget(budget_token)

    authority_token = bind_active_execution_authority(
        ParentExecutionAuthority.unrestricted_root(),
    )
    try:
        boundary = ExecutionBoundary[Ping, Pong](
            RootDelegate(),
            identity=root,
            authority=ParentExecutionAuthority.unrestricted_root(),
        )
        await boundary.execute(Ping())
    finally:
        reset_active_execution_authority(authority_token)
        reset_active_execution_deadline_scope(*tokens)
