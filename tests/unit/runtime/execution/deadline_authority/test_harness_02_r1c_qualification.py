# © Artur Czarnecki. All rights reserved.

"""HARNESS-02-R1C child admission contributor preservation (Q28–Q29)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import pytest

from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_deadline.admission import (
    ExecutionProtectedWorkAdmissionPort,
    ExecutionProtectedWorkAdmissionResult,
)
from intergrax.contracts.execution_deadline.projection import ExecutionDeadlineProjection
from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    reset_active_execution_identity,
    validate_run_id,
)
from intergrax.runtime.execution.active_execution_budget import (
    ActiveExecutionBudgetState,
    bind_active_execution_budget,
    reset_active_execution_budget,
)
from intergrax.runtime.execution.boundary import ExecutionBoundary, ExecutionIdentityBinding
from intergrax.runtime.execution.budget.ledger import create_execution_budget_ledger
from intergrax.runtime.execution.budget.models import ExecutionBudgetAllocationMode
from intergrax.runtime.execution.child import ChildExecutionRunner
from intergrax.runtime.execution.deadline_scope import (
    bind_active_execution_deadline_scope,
    peek_active_execution_protected_work_admission,
    reset_active_execution_deadline_scope,
)
from intergrax.runtime.execution.protected_work_admission import (
    CanonicalHardProtectedWorkAdmission,
    ComposedProtectedWorkAdmission,
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


@dataclass
class MutableAdmissionContributor:
    """Contract-only contributor with mutable allow/deny (tests only)."""

    allow: bool = True
    call_count: int = 0
    order_log: list[str] = field(default_factory=list)
    order_label: str = ""

    def assert_can_start_protected_work(self) -> ExecutionProtectedWorkAdmissionResult:
        self.call_count += 1
        if self.order_label:
            self.order_log.append(self.order_label)
        if not self.allow:
            return ExecutionProtectedWorkAdmissionResult.CANCELLED
        return ExecutionProtectedWorkAdmissionResult.AVAILABLE


class _FakeUtcClock:
    def __init__(self, now: datetime) -> None:
        self._now = now

    def now_utc(self) -> datetime:
        return self._now


def _unbounded_root_projection() -> ExecutionDeadlineProjection:
    return ExecutionDeadlineProjection(
        deadline_at_utc=None,
        remaining_seconds=float("inf"),
        is_expired=False,
        global_deadline_monotonic=None,
    )


def _composed_root_admission(
    contributor: ExecutionProtectedWorkAdmissionPort,
    *,
    monotonic: _FakeMonotonicClock,
) -> ComposedProtectedWorkAdmission:
    projection = _unbounded_root_projection()
    return ComposedProtectedWorkAdmission(
        canonical=CanonicalHardProtectedWorkAdmission(
            projection=projection,
            cancellation_view=StaticCancellationView(cancelled=False),
            monotonic_clock=monotonic,
        ),
        contributors=(contributor,),
    )


def test_narrow_composed_parent_preserves_contributor_instances() -> None:
    monotonic = _FakeMonotonicClock(10.0)
    contributor = MutableAdmissionContributor()
    parent = _composed_root_admission(contributor, monotonic=monotonic)
    child_projection = ExecutionDeadlineProjection(
        deadline_at_utc=datetime(2026, 6, 1, 12, 30, tzinfo=timezone.utc),
        remaining_seconds=30.0,
        is_expired=False,
        global_deadline_monotonic=40.0,
    )
    narrowed = narrow_protected_work_admission_for_child(
        child_projection,
        parent,
        monotonic_clock=monotonic,
    )
    assert isinstance(narrowed, ComposedProtectedWorkAdmission)
    assert narrowed.contributors == (contributor,)
    assert narrowed.contributors[0] is contributor
    assert (
        narrowed.assert_can_start_protected_work()
        is ExecutionProtectedWorkAdmissionResult.AVAILABLE
    )


def test_grandchild_narrowing_reuses_contributors_without_duplication() -> None:
    monotonic = _FakeMonotonicClock(1.0)
    c1 = MutableAdmissionContributor(order_label="1")
    c2 = MutableAdmissionContributor(order_label="2")
    root = ComposedProtectedWorkAdmission(
        canonical=CanonicalHardProtectedWorkAdmission(
            projection=_unbounded_root_projection(),
            cancellation_view=StaticCancellationView(cancelled=False),
            monotonic_clock=monotonic,
        ),
        contributors=(c1, c2),
    )
    child_projection = ExecutionDeadlineProjection(
        deadline_at_utc=datetime(2026, 6, 1, 12, 10, tzinfo=timezone.utc),
        remaining_seconds=60.0,
        is_expired=False,
        global_deadline_monotonic=61.0,
    )
    child_admission = narrow_protected_work_admission_for_child(
        child_projection,
        root,
        monotonic_clock=monotonic,
    )
    grandchild_projection = ExecutionDeadlineProjection(
        deadline_at_utc=datetime(2026, 6, 1, 12, 11, tzinfo=timezone.utc),
        remaining_seconds=15.0,
        is_expired=False,
        global_deadline_monotonic=16.0,
    )
    grandchild_admission = narrow_protected_work_admission_for_child(
        grandchild_projection,
        child_admission,
        monotonic_clock=monotonic,
    )
    assert isinstance(child_admission, ComposedProtectedWorkAdmission)
    assert isinstance(grandchild_admission, ComposedProtectedWorkAdmission)
    assert child_admission.contributors == (c1, c2)
    assert grandchild_admission.contributors is child_admission.contributors
    assert len(grandchild_admission.contributors) == 2
    assert (
        grandchild_admission.assert_can_start_protected_work()
        is ExecutionProtectedWorkAdmissionResult.AVAILABLE
    )
    assert c1.call_count == 1
    assert c2.call_count == 1
    assert c1.order_log == ["1"]
    assert c2.order_log == ["2"]


def test_canonical_expired_short_circuits_contributors() -> None:
    monotonic = _FakeMonotonicClock(5.0)
    contributor = MutableAdmissionContributor()
    expired = ExecutionDeadlineProjection(
        deadline_at_utc=datetime(2020, 1, 1, tzinfo=timezone.utc),
        remaining_seconds=0.0,
        is_expired=True,
        global_deadline_monotonic=5.0,
    )
    composed = ComposedProtectedWorkAdmission(
        canonical=CanonicalHardProtectedWorkAdmission(
            projection=expired,
            cancellation_view=StaticCancellationView(cancelled=False),
            monotonic_clock=monotonic,
        ),
        contributors=(contributor,),
    )
    assert (
        composed.assert_can_start_protected_work()
        is ExecutionProtectedWorkAdmissionResult.EXPIRED
    )
    assert contributor.call_count == 0


def test_canonical_cancelled_short_circuits_contributors() -> None:
    monotonic = _FakeMonotonicClock()
    contributor = MutableAdmissionContributor()
    projection = ExecutionDeadlineProjection(
        deadline_at_utc=None,
        remaining_seconds=float("inf"),
        is_expired=False,
        global_deadline_monotonic=None,
    )
    composed = ComposedProtectedWorkAdmission(
        canonical=CanonicalHardProtectedWorkAdmission(
            projection=projection,
            cancellation_view=StaticCancellationView(cancelled=True),
            monotonic_clock=monotonic,
        ),
        contributors=(contributor,),
    )
    assert (
        composed.assert_can_start_protected_work()
        is ExecutionProtectedWorkAdmissionResult.CANCELLED
    )
    assert contributor.call_count == 0


@pytest.mark.asyncio
async def test_q28_root_contributor_preserved_in_child() -> None:
    @dataclass(frozen=True)
    class Ping:
        pass

    @dataclass(frozen=True)
    class Pong:
        blocked: bool

    monotonic = _FakeMonotonicClock(200.0)
    contributor = MutableAdmissionContributor(allow=True)
    admission = _composed_root_admission(contributor, monotonic=monotonic)
    now = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
    utc = _FakeUtcClock(now)
    tokens = bind_active_execution_deadline_scope(
        projection=_unbounded_root_projection(),
        admission=admission,
        monotonic_clock=monotonic,
    )
    ledger = create_execution_budget_ledger(RunBudget())
    child_runner = ChildExecutionRunner[Ping, Pong](
        ledger=ledger,
        utc_clock=utc,
        monotonic_clock=monotonic,
    )
    executor = _CountingExecutor()
    invoker = RuntimeToolInvoker(registry=FakeRegistry(_contract()), executor=executor)
    run_id = canonical_run_id_for_tests("h02-r1c-q28")
    state = build_runtime_state_for_tests(run_id=run_id)
    tool_calls_in_child = 0

    class ChildDelegate:
        async def execute(self, request: Ping) -> Pong:
            nonlocal tool_calls_in_child
            port = peek_active_execution_protected_work_admission()
            assert isinstance(port, ComposedProtectedWorkAdmission)
            assert port.contributors[0] is contributor
            assert (
                port.assert_can_start_protected_work()
                is ExecutionProtectedWorkAdmissionResult.AVAILABLE
            )
            contributor.allow = False
            assert (
                port.assert_can_start_protected_work()
                is ExecutionProtectedWorkAdmissionResult.CANCELLED
            )
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
                tool_calls_in_child = executor.calls
                return Pong(blocked=result.success is False)
            finally:
                reset_active_execution_identity(identity_token)

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
                    requested_budget=RunBudget(max_wall_time_seconds=30.0),
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
        pong = await boundary.execute(Ping())
        assert pong.blocked is True
        assert tool_calls_in_child == 0
        assert executor.calls == 0
    finally:
        reset_active_execution_authority(authority_token)
        reset_active_execution_deadline_scope(*tokens)


@pytest.mark.asyncio
async def test_q29_root_contributor_preserved_through_grandchild() -> None:
    @dataclass(frozen=True)
    class Ping:
        depth: int

    @dataclass(frozen=True)
    class Pong:
        denied: bool

    monotonic = _FakeMonotonicClock(50.0)
    contributor = MutableAdmissionContributor(allow=True)
    admission = _composed_root_admission(contributor, monotonic=monotonic)
    now = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
    utc = _FakeUtcClock(now)
    tokens = bind_active_execution_deadline_scope(
        projection=_unbounded_root_projection(),
        admission=admission,
        monotonic_clock=monotonic,
    )
    ledger = create_execution_budget_ledger(RunBudget())
    child_runner = ChildExecutionRunner[Ping, Pong](
        ledger=ledger,
        utc_clock=utc,
        monotonic_clock=monotonic,
    )

    class GrandchildDelegate:
        async def execute(self, request: Ping) -> Pong:
            contributor.allow = False
            port = peek_active_execution_protected_work_admission()
            assert isinstance(port, ComposedProtectedWorkAdmission)
            assert port.contributors[0] is contributor
            denied = (
                port.assert_can_start_protected_work()
                is not ExecutionProtectedWorkAdmissionResult.AVAILABLE
            )
            return Pong(denied=denied)

    class ChildDelegate:
        async def execute(self, request: Ping) -> Pong:
            port = peek_active_execution_protected_work_admission()
            assert isinstance(port, ComposedProtectedWorkAdmission)
            assert port.contributors[0] is contributor
            return await child_runner.execute(
                request=Ping(depth=2),
                delegate=GrandchildDelegate(),
                requested_budget=RunBudget(max_wall_time_seconds=10.0),
            )

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
                    request=Ping(depth=1),
                    delegate=ChildDelegate(),
                    requested_budget=RunBudget(max_wall_time_seconds=20.0),
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
        pong = await boundary.execute(Ping(depth=0))
        assert pong.denied is True
    finally:
        reset_active_execution_authority(authority_token)
        reset_active_execution_deadline_scope(*tokens)
