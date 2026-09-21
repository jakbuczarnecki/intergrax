# © Artur Czarnecki. All rights reserved.

"""HARNESS-02-R1A qualification proofs Q05–Q07, Q18–Q22."""

from __future__ import annotations

import ast
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

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
from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.runtime.execution.boundary import ExecutionBoundary, ExecutionIdentityBinding
from intergrax.runtime.execution.active_execution_budget import (
    ActiveExecutionBudgetState,
    bind_active_execution_budget,
    reset_active_execution_budget,
)
from intergrax.runtime.execution.budget.ledger import create_execution_budget_ledger
from intergrax.runtime.execution.budget.models import ExecutionBudgetAllocationMode
from intergrax.runtime.execution.child import ChildExecutionRunner
from intergrax.runtime.execution.deadline_authority import (
    ExecutionDeadlineAuthorityResolver,
    InMemoryExecutionDeadlinePersistence,
)
from intergrax.runtime.execution.deadline_scope import (
    bind_active_execution_deadline_scope,
    peek_active_execution_deadline_projection,
    reset_active_execution_deadline_scope,
)
from intergrax.runtime.execution.deadline_provider_guard import (
    ExecutionProtectedWorkDeniedError,
)
from intergrax.runtime.execution.durable_execution_wiring import (
    wire_durable_execution_runtime_dependencies,
)
from intergrax.runtime.execution.protected_work_admission import (
    CanonicalHardProtectedWorkAdmission,
    ExecutionProtectedWorkAdmissionDeniedError,
    StaticCancellationView,
)
from intergrax.runtime.governance.active_execution_authority import (
    bind_active_execution_authority,
    reset_active_execution_authority,
)
from intergrax.runtime.execution.runtime import ExecutionRuntime
from intergrax.runtime.nexus.budget.budget_models import RunBudget
from intergrax.runtime.task.nexus_worker_execution import NexusWorkerRuntime
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.base.base_llm_adapter import BaseLLMAdapter

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[5]
_CONTRACTS_DEADLINE = _REPO_ROOT / "intergrax" / "contracts" / "execution_deadline"


class _FakeUtcClock:
    def __init__(self, now: datetime) -> None:
        self._now = now

    def now_utc(self) -> datetime:
        return self._now


class _FakeMonotonicClock:
    def __init__(self, value: float = 100.0) -> None:
        self._value = value

    def monotonic(self) -> float:
        return self._value

    def advance(self, seconds: float) -> None:
        self._value += seconds


def _bind_parent_projection(
    deadline_at_utc: datetime,
    *,
    monotonic: _FakeMonotonicClock,
    now_utc: datetime | None = None,
) -> tuple:
    now = now_utc or datetime(2026, 1, 1, tzinfo=timezone.utc)
    remaining = max(0.0, (deadline_at_utc - now).total_seconds())
    projection = ExecutionDeadlineProjection(
        deadline_at_utc=deadline_at_utc,
        remaining_seconds=remaining,
        is_expired=deadline_at_utc <= now,
        global_deadline_monotonic=monotonic.monotonic() + remaining,
    )
    admission = CanonicalHardProtectedWorkAdmission(
        projection=projection,
        cancellation_view=StaticCancellationView(cancelled=False),
        monotonic_clock=monotonic,
    )
    return bind_active_execution_deadline_scope(
        projection=projection,
        admission=admission,
        monotonic_clock=monotonic,
    )


@pytest.mark.asyncio
async def test_q05_child_effective_deadline_capped_by_parent() -> None:
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Ping:
        pass

    @dataclass(frozen=True)
    class Pong:
        deadline: datetime | None

    now = datetime.now(timezone.utc)
    parent_deadline = now + timedelta(seconds=120.0)
    monotonic = _FakeMonotonicClock(200.0)
    tokens = _bind_parent_projection(
        parent_deadline,
        monotonic=monotonic,
        now_utc=now,
    )
    observed: list[ExecutionDeadlineProjection | None] = []
    ledger = create_execution_budget_ledger(RunBudget())
    child_runner = ChildExecutionRunner[Ping, Pong](ledger=ledger)

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
    parent_after_child: list[ExecutionDeadlineProjection | None] = []

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
                result = await child_runner.execute(
                    request=request,
                    delegate=ChildDelegate(),
                    requested_budget=RunBudget(max_wall_time_seconds=999.0),
                )
                parent_after_child.append(peek_active_execution_deadline_projection())
                return result
            finally:
                reset_active_execution_budget(budget_token)

    try:
        boundary = ExecutionBoundary[Ping, Pong](
            RootDelegate(),
            identity=root,
            authority=ParentExecutionAuthority.unrestricted_root(),
        )
        await boundary.execute(Ping())
        assert observed[0] is not None
        assert observed[0].deadline_at_utc is not None
        assert observed[0].deadline_at_utc <= parent_deadline
        assert parent_after_child[0] is not None
        assert parent_after_child[0].deadline_at_utc == parent_deadline
    finally:
        reset_active_execution_deadline_scope(*tokens)


@pytest.mark.asyncio
async def test_q06_grandchild_observes_narrowed_projection() -> None:
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Ping:
        depth: int

    @dataclass(frozen=True)
    class Pong:
        deadline: datetime | None

    now = datetime.now(timezone.utc)
    root_deadline = now + timedelta(seconds=120.0)
    monotonic = _FakeMonotonicClock(300.0)
    tokens = _bind_parent_projection(root_deadline, monotonic=monotonic, now_utc=now)
    parent_at_child: list[ExecutionDeadlineProjection | None] = []
    ledger = create_execution_budget_ledger(RunBudget())
    child_runner = ChildExecutionRunner[Ping, Pong](ledger=ledger)
    grandchild_seen: list[datetime | None] = []

    class GrandchildDelegate:
        async def execute(self, request: Ping) -> Pong:
            proj = peek_active_execution_deadline_projection()
            grandchild_seen.append(proj.deadline_at_utc if proj else None)
            return Pong(deadline=proj.deadline_at_utc if proj else None)

    class ChildDelegate:
        async def execute(self, request: Ping) -> Pong:
            parent_at_child.append(peek_active_execution_deadline_projection())
            return await child_runner.execute(
                request=Ping(depth=2),
                delegate=GrandchildDelegate(),
                requested_budget=RunBudget(max_wall_time_seconds=15.0),
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
                    requested_budget=RunBudget(max_wall_time_seconds=30.0),
                )
            finally:
                reset_active_execution_budget(budget_token)

    try:
        boundary = ExecutionBoundary[Ping, Pong](
            RootDelegate(),
            identity=root,
            authority=ParentExecutionAuthority.unrestricted_root(),
        )
        await boundary.execute(Ping(depth=0))
        assert len(grandchild_seen) == 1
        assert grandchild_seen[0] is not None
        assert parent_at_child[0] is not None
        assert grandchild_seen[0] <= parent_at_child[0].deadline_at_utc
        assert grandchild_seen[0] <= root_deadline
    finally:
        reset_active_execution_deadline_scope(*tokens)


@pytest.mark.asyncio
async def test_q07_expired_parent_blocks_child_before_delegate() -> None:
    monotonic = _FakeMonotonicClock(1.0)
    expired = ExecutionDeadlineProjection(
        deadline_at_utc=datetime(2020, 1, 1, tzinfo=timezone.utc),
        remaining_seconds=0.0,
        is_expired=True,
        global_deadline_monotonic=1.0,
    )
    tokens = bind_active_execution_deadline_scope(
        projection=expired,
        admission=CanonicalHardProtectedWorkAdmission(
            projection=expired,
            cancellation_view=StaticCancellationView(cancelled=False),
            monotonic_clock=monotonic,
        ),
        monotonic_clock=monotonic,
    )
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Ping:
        pass

    @dataclass(frozen=True)
    class Pong:
        ok: bool

    calls = 0
    ledger = create_execution_budget_ledger(RunBudget())

    class ChildDelegate:
        async def execute(self, request: Ping) -> Pong:
            nonlocal calls
            calls += 1
            return Pong(ok=True)

    child_runner = ChildExecutionRunner[Ping, Pong](ledger=ledger)
    root = ExecutionIdentityBinding(
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )
    identity_token = bind_active_execution_identity(
        run_id=root.run_id,
        attempt_id=root.attempt_id,
        execution_id=root.execution_id,
    )
    budget_token = bind_active_execution_budget(
        ActiveExecutionBudgetState(
            execution_id=root.execution_id,
            mode=ExecutionBudgetAllocationMode.SHARED,
            ledger=ledger,
        ),
    )
    authority_token = bind_active_execution_authority(
        ParentExecutionAuthority.unrestricted_root(),
    )
    try:
        with pytest.raises(ExecutionProtectedWorkAdmissionDeniedError):
            await child_runner.execute(request=Ping(), delegate=ChildDelegate())
        assert calls == 0
    finally:
        reset_active_execution_authority(authority_token)
        reset_active_execution_budget(budget_token)
        reset_active_execution_deadline_scope(*tokens)
        reset_active_execution_identity(identity_token)


def test_q19_from_registry_requires_deadline_resolver_with_durable_budget() -> None:
    from intergrax.contracts.admitted_root_governance_identity import (
        AdmittedRootGovernanceIdentity,
    )

    def _admit(_task: object) -> AdmittedRootGovernanceIdentity:
        return AdmittedRootGovernanceIdentity(
            tenant_id="t1",
            workspace_id="w1",
            principal_id="p1",
        )

    with pytest.raises(ValueError, match="deadline_authority_resolver"):
        NexusWorkerRuntime.from_registry(
            MagicMock(),
            run_budget_persistence=MagicMock(),
            deadline_authority_resolver=None,
            production_mode=False,
            admit_root_governance_identity=_admit,
        )


def test_q19_durable_execution_runtime_rejects_missing_resolver() -> None:
    with pytest.raises(Exception, match="deadline authority resolver"):
        ExecutionRuntime(
            MagicMock(),
            run_budget=RunBudget(max_wall_time_seconds=1.0),
            run_budget_persistence=MagicMock(),
        )


def test_q21_contracts_execution_deadline_have_no_runtime_imports() -> None:
    forbidden = ("contextvars", "intergrax.runtime")
    for path in _CONTRACTS_DEADLINE.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for prefix in forbidden:
                        assert not alias.name.startswith(prefix), path
            if isinstance(node, ast.ImportFrom) and node.module:
                for prefix in forbidden:
                    assert not node.module.startswith(prefix), path


def test_q22_resolver_accepts_clock_ports() -> None:
    persistence = InMemoryExecutionDeadlinePersistence()
    utc = _FakeUtcClock(datetime(2026, 1, 1, tzinfo=timezone.utc))
    monotonic = _FakeMonotonicClock()
    resolver = ExecutionDeadlineAuthorityResolver(
        persistence,
        utc_clock=utc,
        monotonic_clock=monotonic,
    )
    resolution = resolver.resolve_for_root(
        tenant_id="t1",
        run_id=mint_run_id(),
        run_budget=RunBudget(max_wall_time_seconds=5.0),
        existing_run_materialized=False,
    )
    assert resolution.snapshot.deadline_at_utc is not None


def test_q11_llm_execute_passes_bounded_timeout_to_resilience() -> None:
    monotonic = _FakeMonotonicClock(100.0)
    projection = ExecutionDeadlineProjection(
        deadline_at_utc=datetime(2026, 1, 1, 0, 0, 4, tzinfo=timezone.utc),
        remaining_seconds=4.0,
        is_expired=False,
        global_deadline_monotonic=104.0,
    )
    tokens = bind_active_execution_deadline_scope(
        projection=projection,
        admission=CanonicalHardProtectedWorkAdmission(
            projection=projection,
            cancellation_view=StaticCancellationView(cancelled=False),
            monotonic_clock=monotonic,
        ),
        monotonic_clock=monotonic,
    )
    captured: list[float | None] = []

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
    adapter.call_config = adapter.call_config.__class__(timeout_sec=30.0)

    def _spy_execute_with_resilience(physical_attempt, *, provider, config, retry_fn, tenant_id):
        captured.append(config.timeout_sec)
        return physical_attempt()

    try:
        import intergrax.llm_adapters.contracts.llm_adapter as mod

        original = mod.execute_with_resilience
        mod.execute_with_resilience = _spy_execute_with_resilience
        adapter._execute(lambda: "ok")
        assert captured == [4.0]
    finally:
        mod.execute_with_resilience = original
        reset_active_execution_deadline_scope(*tokens)


def test_q18_streaming_expired_blocks_factory() -> None:
    monotonic = _FakeMonotonicClock(1.0)
    projection = ExecutionDeadlineProjection(
        deadline_at_utc=datetime(2020, 1, 1, tzinfo=timezone.utc),
        remaining_seconds=0.0,
        is_expired=True,
        global_deadline_monotonic=1.0,
    )
    tokens = bind_active_execution_deadline_scope(
        projection=projection,
        admission=CanonicalHardProtectedWorkAdmission(
            projection=projection,
            cancellation_view=StaticCancellationView(cancelled=False),
            monotonic_clock=monotonic,
        ),
        monotonic_clock=monotonic,
    )
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


def test_wire_durable_execution_pairs_budget_and_deadline() -> None:
    from tests.unit.runtime.execution.deadline_authority.test_harness_02_r1_qualification import (
        _KV,
    )

    deps = wire_durable_execution_runtime_dependencies(kv_store=_KV())
    assert deps.run_budget_persistence is not None
    assert deps.deadline_authority_resolver is not None
