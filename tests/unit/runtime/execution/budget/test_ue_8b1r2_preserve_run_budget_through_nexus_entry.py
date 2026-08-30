# © Artur Czarnecki. All rights reserved.

"""UE-8B1R2 — preserve active per-Run budget ledger through Nexus entry."""

from __future__ import annotations

from contextvars import Token
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    require_active_execution_id,
    reset_active_execution_identity,
)
from intergrax.runtime.execution.active_execution_budget import (
    ActiveExecutionBudgetState,
    bind_active_execution_budget,
    bind_root_execution_budget,
    peek_active_execution_budget,
    require_active_execution_budget,
    reset_active_execution_budget,
)
from intergrax.runtime.execution.budget.ledger import (
    ExecutionBudgetLedger,
    ExecutionBudgetLedgerFactory,
    create_execution_budget_ledger,
    create_execution_budget_ledger_factory,
)
from intergrax.runtime.execution.budget.models import (
    BudgetUsageTotals,
    ChildBudgetAllocationContext,
    ChildBudgetAllocationDecision,
    ExecutionBudgetAllocationMode,
)
from intergrax.runtime.execution.budget.policy import DefaultSharedPoolBudgetPolicy
from intergrax.runtime.execution.child import ChildExecutionRunner
from intergrax.runtime.governance.active_execution_authority import (
    bind_active_execution_authority,
    reset_active_execution_authority,
)
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


def _task() -> Task:
    return Task(tenant_id="t1", user_id="u1", agent_id="agent-1", message="budget")


def _bind_upstream_context(
    *,
    run_id: RunId,
    attempt_id: AttemptId,
    execution_id: ExecutionId,
    ledger: ExecutionBudgetLedger,
    mode: ExecutionBudgetAllocationMode = ExecutionBudgetAllocationMode.SHARED,
    reservation_allowance: RunBudget | None = None,
) -> tuple[Token, Token, Token]:
    identity_token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    authority_token = bind_active_execution_authority(
        ParentExecutionAuthority.unrestricted_root(),
    )
    budget_token = bind_active_execution_budget(
        ActiveExecutionBudgetState(
            execution_id=execution_id,
            mode=mode,
            ledger=ledger,
            reservation_allowance=reservation_allowance,
        )
    )
    return identity_token, authority_token, budget_token


def _reset_upstream_context(
    *,
    identity_token: Token,
    authority_token: Token,
    budget_token: Token,
) -> None:
    reset_active_execution_budget(budget_token)
    reset_active_execution_authority(authority_token)
    reset_active_execution_identity(identity_token)


def _child_runner(loop: NexusLoop) -> ChildExecutionRunner[Ping, Pong]:
    return loop._graph_executor._child_runner  # noqa: SLF001


def _consume_root_pool(
    ledger: ExecutionBudgetLedger,
    *,
    root_execution_id: ExecutionId,
    amount: int,
) -> None:
    child_id = mint_execution_id()
    ledger.grant_child_budget(
        execution_id=child_id,
        parent_execution_id=root_execution_id,
        decision=ChildBudgetAllocationDecision(mode=ExecutionBudgetAllocationMode.SHARED),
    )
    ledger.consume_budget(child_id, BudgetUsageTotals(total_tokens=amount))
    ledger.release_child_budget(child_id)


@pytest.mark.asyncio
async def test_upstream_execution_with_active_ledger_does_not_call_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = create_execution_budget_ledger(RunBudget(max_total_tokens=100))
    inner_factory = create_execution_budget_ledger_factory(RunBudget(max_total_tokens=100))
    factory = MagicMock(spec=ExecutionBudgetLedgerFactory, wraps=inner_factory)
    loop = NexusLoop(AgentRegistry(), execution_budget_ledger_factory=factory)
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    identity_token, authority_token, budget_token = _bind_upstream_context(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        ledger=ledger,
    )

    async def _noop(task: Task) -> TaskResult:
        return TaskResult(task_id=task.task_id, run_id=run_id, state=TaskState.COMPLETED)

    monkeypatch.setattr(loop, "_handle_task_impl", _noop)
    try:
        await loop.handle_task(_task(), run_id=run_id, attempt_id=attempt_id)
    finally:
        _reset_upstream_context(
            identity_token=identity_token,
            authority_token=authority_token,
            budget_token=budget_token,
        )

    factory.create_ledger.assert_not_called()


@pytest.mark.asyncio
async def test_same_ledger_visible_upstream_nexus_and_nested_children(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = create_execution_budget_ledger(RunBudget(max_tool_calls=50))
    loop = NexusLoop(AgentRegistry(), run_budget=RunBudget(max_tool_calls=50))
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    child_runner = _child_runner(loop)
    seen: list[ExecutionBudgetLedger] = [ledger]

    async def _nested_impl(task: Task) -> TaskResult:
        seen.append(require_active_execution_budget().ledger)

        class GrandchildDelegate:
            async def execute(self, request: Ping) -> Pong:
                seen.append(require_active_execution_budget().ledger)
                return Pong(value=request.value)

        class ChildDelegate:
            async def execute(self, request: Ping) -> Pong:
                seen.append(require_active_execution_budget().ledger)
                return await child_runner.execute(
                    request=request,
                    delegate=GrandchildDelegate(),
                )

        await child_runner.execute(request=Ping(value="nexus"), delegate=ChildDelegate())
        return TaskResult(task_id=task.task_id, run_id=run_id, state=TaskState.COMPLETED)

    monkeypatch.setattr(loop, "_handle_task_impl", _nested_impl)
    identity_token, authority_token, budget_token = _bind_upstream_context(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        ledger=ledger,
    )
    try:
        await loop.handle_task(_task(), run_id=run_id, attempt_id=attempt_id)
    finally:
        _reset_upstream_context(
            identity_token=identity_token,
            authority_token=authority_token,
            budget_token=budget_token,
        )

    assert len(seen) == 4
    assert all(item is ledger for item in seen)


@pytest.mark.asyncio
async def test_upstream_partial_consumption_visible_inside_nexus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = create_execution_budget_ledger(RunBudget(max_total_tokens=100))
    loop = NexusLoop(AgentRegistry(), run_budget=RunBudget(max_total_tokens=100))
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    _consume_root_pool(ledger, root_execution_id=execution_id, amount=70)
    observed: list[int | None] = []

    async def _observe(task: Task) -> TaskResult:
        observed.append(
            require_active_execution_budget().ledger.snapshot_root_available().max_total_tokens
        )
        return TaskResult(task_id=task.task_id, run_id=run_id, state=TaskState.COMPLETED)

    monkeypatch.setattr(loop, "_handle_task_impl", _observe)
    identity_token, authority_token, budget_token = _bind_upstream_context(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        ledger=ledger,
    )
    try:
        await loop.handle_task(_task(), run_id=run_id, attempt_id=attempt_id)
    finally:
        _reset_upstream_context(
            identity_token=identity_token,
            authority_token=authority_token,
            budget_token=budget_token,
        )

    assert observed == [30]


@pytest.mark.asyncio
async def test_upstream_reserved_context_backed_by_same_ledger_in_nexus_children(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = create_execution_budget_ledger(RunBudget(max_tool_calls=100))
    root_id = mint_execution_id()
    reserved_id = mint_execution_id()
    policy = DefaultSharedPoolBudgetPolicy()
    reservation = RunBudget(max_tool_calls=10)
    grant = ledger.grant_child_budget(
        execution_id=reserved_id,
        parent_execution_id=root_id,
        decision=policy.resolve_child_budget(
            ChildBudgetAllocationContext(
                parent_execution_id=root_id,
                parent_allocation_mode=ExecutionBudgetAllocationMode.SHARED,
                parent_reservation_remaining=None,
                requested_budget=reservation,
            )
        ),
    )
    loop = NexusLoop(AgentRegistry(), run_budget=RunBudget(max_tool_calls=100))
    child_runner = _child_runner(loop)
    child_ledgers: list[ExecutionBudgetLedger] = []

    async def _reserved_impl(task: Task) -> TaskResult:
        state = require_active_execution_budget()
        assert state.mode is ExecutionBudgetAllocationMode.RESERVED
        assert state.ledger is ledger

        class ChildDelegate:
            async def execute(self, request: Ping) -> Pong:
                child_state = require_active_execution_budget()
                child_ledgers.append(child_state.ledger)
                assert child_state.ledger is ledger
                child_state.ledger.consume_budget(
                    require_active_execution_id(),
                    BudgetUsageTotals(tool_calls=4),
                )
                return Pong(value=request.value)

        await child_runner.execute(request=Ping(value="reserved"), delegate=ChildDelegate())
        return TaskResult(task_id=task.task_id, run_id=run_id, state=TaskState.COMPLETED)

    monkeypatch.setattr(loop, "_handle_task_impl", _reserved_impl)
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    identity_token, authority_token, budget_token = _bind_upstream_context(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=reserved_id,
        ledger=ledger,
        mode=ExecutionBudgetAllocationMode.RESERVED,
        reservation_allowance=grant.reservation_allowance,
    )
    try:
        await loop.handle_task(_task(), run_id=run_id, attempt_id=attempt_id)
    finally:
        _reset_upstream_context(
            identity_token=identity_token,
            authority_token=authority_token,
            budget_token=budget_token,
        )

    assert child_ledgers == [ledger]
    assert ledger.snapshot_reservation_remaining(reserved_id).max_tool_calls == 6


@pytest.mark.asyncio
async def test_factory_call_count_root_nexus_one_upstream_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from intergrax.runtime.task.unified_task_runner import UnifiedTaskRunner

    inner_factory = create_execution_budget_ledger_factory(RunBudget(max_total_tokens=100))
    factory = MagicMock(spec=ExecutionBudgetLedgerFactory, wraps=inner_factory)
    run_budget = RunBudget(max_total_tokens=100)
    loop = NexusLoop(
        AgentRegistry(),
        execution_budget_ledger_factory=factory,
        run_budget=run_budget,
    )

    async def _noop(task: Task) -> TaskResult:
        from intergrax.contracts.execution_identity import require_active_execution_identity

        active_run_id, _ = require_active_execution_identity()
        return TaskResult(task_id=task.task_id, run_id=active_run_id, state=TaskState.COMPLETED)

    monkeypatch.setattr(loop, "_handle_task_impl", _noop)
    runner = UnifiedTaskRunner(
        loop,
        execution_budget_ledger_factory=factory,
        run_budget=run_budget,
    )

    await runner.run_task(_task())
    assert factory.create_ledger.call_count == 1

    factory.create_ledger.reset_mock()
    ledger = create_execution_budget_ledger(RunBudget(max_total_tokens=100))
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    identity_token, authority_token, budget_token = _bind_upstream_context(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        ledger=ledger,
    )
    try:
        await loop.handle_task(_task(), run_id=run_id, attempt_id=attempt_id)
    finally:
        _reset_upstream_context(
            identity_token=identity_token,
            authority_token=authority_token,
            budget_token=budget_token,
        )

    factory.create_ledger.assert_not_called()


@pytest.mark.asyncio
async def test_active_execution_without_budget_context_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = NexusLoop(AgentRegistry(), run_budget=RunBudget(max_total_tokens=100))
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    parent_execution_id = mint_execution_id()
    identity_token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        parent_execution_id=parent_execution_id,
    )
    authority_token = bind_active_execution_authority(
        ParentExecutionAuthority.unrestricted_root(),
    )

    async def _noop(task: Task) -> TaskResult:
        return TaskResult(task_id=task.task_id, run_id=run_id, state=TaskState.COMPLETED)

    monkeypatch.setattr(loop, "_handle_task_impl", _noop)
    try:
        with pytest.raises(
            RuntimeError,
            match="active execution budget required",
        ):
            await loop.handle_task(_task(), run_id=run_id, attempt_id=attempt_id)
    finally:
        reset_active_execution_authority(authority_token)
        reset_active_execution_identity(identity_token)


@pytest.mark.asyncio
async def test_nexus_exception_preserves_upstream_budget_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = create_execution_budget_ledger(RunBudget(max_total_tokens=100))
    loop = NexusLoop(AgentRegistry(), run_budget=RunBudget(max_total_tokens=100))
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    _consume_root_pool(ledger, root_execution_id=execution_id, amount=25)

    async def _boom(task: Task) -> TaskResult:
        raise RuntimeError("nexus-fail")

    monkeypatch.setattr(loop, "_handle_task_impl", _boom)
    identity_token, authority_token, budget_token = _bind_upstream_context(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        ledger=ledger,
    )
    upstream_state = peek_active_execution_budget()
    assert upstream_state is not None
    try:
        with pytest.raises(RuntimeError, match="nexus-fail"):
            await loop.handle_task(_task(), run_id=run_id, attempt_id=attempt_id)
    finally:
        restored = peek_active_execution_budget()
        assert restored is upstream_state
        assert restored.ledger is ledger
        assert restored.ledger.snapshot_root_available().max_total_tokens == 75
        _reset_upstream_context(
            identity_token=identity_token,
            authority_token=authority_token,
            budget_token=budget_token,
        )


@pytest.mark.asyncio
async def test_nexus_return_restores_upstream_budget_state_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = create_execution_budget_ledger(RunBudget(max_total_tokens=100))
    loop = NexusLoop(AgentRegistry(), run_budget=RunBudget(max_total_tokens=100))
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    identity_token, authority_token, budget_token = _bind_upstream_context(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        ledger=ledger,
    )
    before = peek_active_execution_budget()
    assert before is not None

    async def _noop(task: Task) -> TaskResult:
        return TaskResult(task_id=task.task_id, run_id=run_id, state=TaskState.COMPLETED)

    monkeypatch.setattr(loop, "_handle_task_impl", _noop)
    try:
        await loop.handle_task(_task(), run_id=run_id, attempt_id=attempt_id)
    finally:
        after = peek_active_execution_budget()
        assert after is before
        assert after == before
        _reset_upstream_context(
            identity_token=identity_token,
            authority_token=authority_token,
            budget_token=budget_token,
        )

    assert peek_active_execution_budget() is None


@pytest.mark.asyncio
async def test_fresh_root_nexus_run_still_creates_one_ledger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from intergrax.runtime.task.unified_task_runner import UnifiedTaskRunner

    run_budget = RunBudget(max_total_tokens=42)
    loop = NexusLoop(AgentRegistry(), run_budget=run_budget)
    observed: list[int | None] = []

    async def _observe(task: Task) -> TaskResult:
        state = require_active_execution_budget()
        observed.append(state.ledger.snapshot_root_available().max_total_tokens)
        from intergrax.contracts.execution_identity import require_active_execution_identity

        active_run_id, _ = require_active_execution_identity()
        return TaskResult(task_id=task.task_id, run_id=active_run_id, state=TaskState.COMPLETED)

    monkeypatch.setattr(loop, "_handle_task_impl", _observe)
    runner = UnifiedTaskRunner(loop, run_budget=run_budget)
    await runner.run_task(_task())

    assert observed == [42]
    assert peek_active_execution_budget() is None
