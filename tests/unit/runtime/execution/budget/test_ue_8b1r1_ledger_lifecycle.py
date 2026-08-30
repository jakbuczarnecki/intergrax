# © Artur Czarnecki. All rights reserved.

"""UE-8B1R1 — per-Run ledger lifecycle and same-Run identity proofs."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from intergrax.applications._shared.nexus_factory import build_nexus_loop_from_environment
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    require_active_execution_id,
)
from intergrax.runtime.execution.active_execution_budget import (
    bind_root_execution_budget,
    peek_active_execution_budget,
    require_active_execution_budget,
    reset_active_execution_budget,
)
from intergrax.runtime.execution.boundary import ExecutionBoundary, ExecutionIdentityBinding
from intergrax.runtime.execution.budget.ledger import create_execution_budget_ledger
from intergrax.runtime.execution.budget.models import BudgetUsageTotals
from intergrax.runtime.execution.child import ChildExecutionRunner
from intergrax.runtime.nexus.budget.budget_models import RunBudget
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskResult, TaskState

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@dataclass(frozen=True)
class Ping:
    value: str


@dataclass(frozen=True)
class Pong:
    value: str


def _minimal_env() -> ApplicationEnvironmentProfile:
    return ApplicationEnvironmentProfile.lab_defaults()


def _child_runner(loop: NexusLoop) -> ChildExecutionRunner[Ping, Pong]:
    return loop._graph_executor._child_runner  # noqa: SLF001


@pytest.mark.asyncio
async def test_per_run_isolation_on_long_lived_nexus_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = build_nexus_loop_from_environment(
        AgentRegistry(),
        env=_minimal_env(),
        run_budget=RunBudget(max_total_tokens=100),
    )
    run_a_available: list[int | None] = []
    run_b_available: list[int | None] = []
    call_count = 0

    async def _fake_impl(task: Task) -> TaskResult:
        nonlocal call_count
        call_count += 1
        from intergrax.contracts.execution_identity import require_active_execution_identity
        from intergrax.runtime.execution.budget.models import (
            ChildBudgetAllocationDecision,
            ExecutionBudgetAllocationMode,
        )

        state = require_active_execution_budget()
        ledger = state.ledger
        root_execution_id = require_active_execution_id()
        if call_count == 1:
            child_id = mint_execution_id()
            ledger.grant_child_budget(
                execution_id=child_id,
                parent_execution_id=root_execution_id,
                decision=ChildBudgetAllocationDecision(
                    mode=ExecutionBudgetAllocationMode.SHARED,
                ),
            )
            ledger.consume_budget(child_id, BudgetUsageTotals(total_tokens=70))
            ledger.release_child_budget(child_id)
            run_a_available.append(ledger.snapshot_root_available().max_total_tokens)
        else:
            run_b_available.append(ledger.snapshot_root_available().max_total_tokens)

        active_run_id, _ = require_active_execution_identity()
        return TaskResult(task_id=task.task_id, run_id=active_run_id, state=TaskState.COMPLETED)

    monkeypatch.setattr(loop, "_handle_task_impl", _fake_impl)
    task = Task(tenant_id="t1", user_id="u1", agent_id="agent-1", message="budget")
    from intergrax.runtime.task.unified_task_runner import UnifiedTaskRunner

    runner = UnifiedTaskRunner(loop, run_budget=RunBudget(max_total_tokens=100))
    await runner.run_task(task)
    await runner.run_task(task)

    assert run_a_available == [30]
    assert run_b_available == [100]


@pytest.mark.asyncio
async def test_same_run_children_share_ledger_instance() -> None:
    ledger = create_execution_budget_ledger(RunBudget(max_tool_calls=50))
    child_runner = ChildExecutionRunner[Ping, Pong](ledger=ledger)
    seen_ledgers: list[object] = []

    class ChildDelegate:
        async def execute(self, request: Ping) -> Pong:
            seen_ledgers.append(require_active_execution_budget().ledger)
            return Pong(value=request.value)

    class RootDelegate:
        async def execute(self, request: Ping) -> Pong:
            await child_runner.execute(request=Ping(value="a"), delegate=ChildDelegate())
            await child_runner.execute(request=Ping(value="b"), delegate=ChildDelegate())
            return Pong(value="ok")

    root_id = mint_execution_id()
    token = bind_root_execution_budget(execution_id=root_id, ledger=ledger)
    try:
        await ExecutionBoundary[Ping, Pong](
            RootDelegate(),
            identity=ExecutionIdentityBinding(
                run_id=mint_run_id(),
                attempt_id=mint_attempt_id(),
                execution_id=root_id,
            ),
            authority=ParentExecutionAuthority.scoped(("read",)),
        ).execute(Ping(value="root"))
    finally:
        reset_active_execution_budget(token)

    assert len(seen_ledgers) == 2
    assert seen_ledgers[0] is ledger
    assert seen_ledgers[1] is ledger


@pytest.mark.asyncio
async def test_nested_same_run_reuses_canonical_ledger() -> None:
    ledger = create_execution_budget_ledger(RunBudget(max_tool_calls=50))
    child_runner = ChildExecutionRunner[Ping, Pong](ledger=ledger)
    nested_ledgers: list[object] = []

    class GrandchildDelegate:
        async def execute(self, request: Ping) -> Pong:
            nested_ledgers.append(require_active_execution_budget().ledger)
            return Pong(value=request.value)

    class ChildDelegate:
        async def execute(self, request: Ping) -> Pong:
            nested_ledgers.append(require_active_execution_budget().ledger)
            return await child_runner.execute(request=request, delegate=GrandchildDelegate())

    class RootDelegate:
        async def execute(self, request: Ping) -> Pong:
            nested_ledgers.append(require_active_execution_budget().ledger)
            return await child_runner.execute(request=request, delegate=ChildDelegate())

    root_id = mint_execution_id()
    token = bind_root_execution_budget(execution_id=root_id, ledger=ledger)
    try:
        await ExecutionBoundary[Ping, Pong](
            RootDelegate(),
            identity=ExecutionIdentityBinding(
                run_id=mint_run_id(),
                attempt_id=mint_attempt_id(),
                execution_id=root_id,
            ),
            authority=ParentExecutionAuthority.scoped(("read",)),
        ).execute(Ping(value="nested"))
    finally:
        reset_active_execution_budget(token)

    assert nested_ledgers == [ledger, ledger, ledger]


def test_composition_does_not_store_mutable_ledger_on_child_runner() -> None:
    loop = build_nexus_loop_from_environment(
        AgentRegistry(),
        env=_minimal_env(),
        run_budget=RunBudget(max_tool_calls=10),
    )
    assert _child_runner(loop)._ledger is None


@pytest.mark.asyncio
async def test_handle_task_binds_root_execution_budget() -> None:
    from intergrax.runtime.task.unified_task_runner import UnifiedTaskRunner

    run_budget = RunBudget(max_total_tokens=42)
    loop = NexusLoop(AgentRegistry(), run_budget=run_budget)
    observed: list[int | None] = []

    async def _fake_impl(task: Task) -> TaskResult:
        state = peek_active_execution_budget()
        assert state is not None
        observed.append(state.ledger.snapshot_root_available().max_total_tokens)
        from intergrax.contracts.execution_identity import require_active_execution_identity

        run_id, _ = require_active_execution_identity()
        return TaskResult(task_id=task.task_id, run_id=run_id, state=TaskState.COMPLETED)

    loop._handle_task_impl = _fake_impl  # type: ignore[method-assign]
    task = Task(tenant_id="t1", user_id="u1", agent_id="agent-1", message="bind")
    runner = UnifiedTaskRunner(loop, run_budget=run_budget)
    await runner.run_task(task)

    assert observed == [42]
    assert peek_active_execution_budget() is None
