# © Artur Czarnecki. All rights reserved.

"""UE-8B1R1 — shared-under-reserved immediate backing debit proofs."""

from __future__ import annotations

import threading
from dataclasses import dataclass

import pytest

from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
)
from intergrax.runtime.execution.active_execution_budget import (
    bind_root_execution_budget,
    reset_active_execution_budget,
)
from intergrax.runtime.execution.boundary import ExecutionBoundary, ExecutionIdentityBinding
from intergrax.runtime.execution.budget.ledger import create_execution_budget_ledger
from intergrax.runtime.execution.budget.models import (
    BudgetUsageTotals,
    ChildBudgetAllocationContext,
    ChildBudgetAllocationDecision,
    ExecutionBudgetAllocationMode,
    ExecutionBudgetError,
    ExecutionBudgetReservationError,
)
from intergrax.runtime.execution.budget.policy import DefaultSharedPoolBudgetPolicy
from intergrax.runtime.execution.child import ChildExecutionRunner
from intergrax.runtime.nexus.budget.budget_models import RunBudget

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@dataclass(frozen=True)
class Ping:
    value: str


@dataclass(frozen=True)
class Pong:
    value: str


def _ledger(tool_calls: int) -> object:
    return create_execution_budget_ledger(RunBudget(max_tool_calls=tool_calls))


def _root_identity(execution_id: object | None = None) -> ExecutionIdentityBinding:
    return ExecutionIdentityBinding(
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=execution_id or mint_execution_id(),
    )


@pytest.mark.asyncio
async def test_shared_under_reserved_repeated_consume_fails() -> None:
    ledger = _ledger(100)
    root_id = mint_execution_id()
    child_runner = ChildExecutionRunner[Ping, Pong](ledger=ledger)
    policy = DefaultSharedPoolBudgetPolicy()
    parent = root_id
    reserved = mint_execution_id()
    ledger.grant_child_budget(
        execution_id=reserved,
        parent_execution_id=parent,
        decision=policy.resolve_child_budget(
            ChildBudgetAllocationContext(
                parent_execution_id=parent,
                parent_allocation_mode=ExecutionBudgetAllocationMode.SHARED,
                parent_reservation_remaining=None,
                requested_budget=RunBudget(max_tool_calls=10),
            )
        ),
    )
    shared = mint_execution_id()
    ledger.grant_child_budget(
        execution_id=shared,
        parent_execution_id=reserved,
        decision=ChildBudgetAllocationDecision(mode=ExecutionBudgetAllocationMode.SHARED),
    )
    ledger.consume_budget(shared, BudgetUsageTotals(tool_calls=8))
    with pytest.raises(ExecutionBudgetError, match="exceeds effective grant"):
        ledger.consume_budget(shared, BudgetUsageTotals(tool_calls=3))


@pytest.mark.asyncio
async def test_shared_siblings_under_reserved_second_fails() -> None:
    ledger = _ledger(100)
    root_id = mint_execution_id()
    parent = root_id
    reserved = mint_execution_id()
    ledger.grant_child_budget(
        execution_id=reserved,
        parent_execution_id=parent,
        decision=ChildBudgetAllocationDecision(
            mode=ExecutionBudgetAllocationMode.RESERVED,
            reservation_request=RunBudget(max_tool_calls=10),
        ),
    )
    child_a = mint_execution_id()
    child_b = mint_execution_id()
    for child in (child_a, child_b):
        ledger.grant_child_budget(
            execution_id=child,
            parent_execution_id=reserved,
            decision=ChildBudgetAllocationDecision(mode=ExecutionBudgetAllocationMode.SHARED),
        )
    ledger.consume_budget(child_a, BudgetUsageTotals(tool_calls=7))
    with pytest.raises(ExecutionBudgetError, match="exceeds effective grant"):
        ledger.consume_budget(child_b, BudgetUsageTotals(tool_calls=5))


def test_parallel_shared_under_reserved_cannot_oversubscribe() -> None:
    ledger = _ledger(100)
    parent = mint_execution_id()
    reserved = mint_execution_id()
    ledger.grant_child_budget(
        execution_id=reserved,
        parent_execution_id=parent,
        decision=ChildBudgetAllocationDecision(
            mode=ExecutionBudgetAllocationMode.RESERVED,
            reservation_request=RunBudget(max_tool_calls=10),
        ),
    )
    successes: list[object] = []
    failures: list[Exception] = []
    barrier = threading.Barrier(2)

    def _attempt() -> None:
        child = mint_execution_id()
        barrier.wait()
        try:
            ledger.grant_child_budget(
                execution_id=child,
                parent_execution_id=reserved,
                decision=ChildBudgetAllocationDecision(
                    mode=ExecutionBudgetAllocationMode.SHARED
                ),
            )
            ledger.consume_budget(child, BudgetUsageTotals(tool_calls=7))
        except (ExecutionBudgetError, ExecutionBudgetReservationError) as exc:
            failures.append(exc)
        else:
            successes.append(child)

    threads = [threading.Thread(target=_attempt) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(successes) == 1
    assert len(failures) == 1


def test_shared_release_does_not_double_account_reserved_backing() -> None:
    ledger = _ledger(100)
    parent = mint_execution_id()
    reserved = mint_execution_id()
    ledger.grant_child_budget(
        execution_id=reserved,
        parent_execution_id=parent,
        decision=ChildBudgetAllocationDecision(
            mode=ExecutionBudgetAllocationMode.RESERVED,
            reservation_request=RunBudget(max_tool_calls=10),
        ),
    )
    shared = mint_execution_id()
    ledger.grant_child_budget(
        execution_id=shared,
        parent_execution_id=reserved,
        decision=ChildBudgetAllocationDecision(mode=ExecutionBudgetAllocationMode.SHARED),
    )
    ledger.consume_budget(shared, BudgetUsageTotals(tool_calls=6))
    assert ledger.snapshot_reservation_remaining(reserved).max_tool_calls == 4
    ledger.release_child_budget(shared)
    assert ledger.snapshot_reservation_remaining(reserved).max_tool_calls == 4
    ledger.release_child_budget(reserved)
    assert ledger.snapshot_root_available().max_tool_calls == 94


@pytest.mark.asyncio
async def test_root_shared_child_still_debits_root() -> None:
    ledger = _ledger(100)
    child_runner = ChildExecutionRunner[Ping, Pong](ledger=ledger)
    root_id = mint_execution_id()

    class ChildDelegate:
        async def execute(self, request: Ping) -> Pong:
            from intergrax.runtime.execution.active_execution_budget import (
                require_active_execution_budget,
            )
            from intergrax.contracts.execution_identity import require_active_execution_id

            state = require_active_execution_budget()
            state.ledger.consume_budget(
                require_active_execution_id(),
                BudgetUsageTotals(tool_calls=12),
            )
            return Pong(value=request.value)

    class RootDelegate:
        async def execute(self, request: Ping) -> Pong:
            return await child_runner.execute(request=request, delegate=ChildDelegate())

    token = bind_root_execution_budget(execution_id=root_id, ledger=ledger)
    try:
        await ExecutionBoundary[Ping, Pong](
            RootDelegate(),
            identity=_root_identity(root_id),
            authority=ParentExecutionAuthority.scoped(("read",)),
        ).execute(Ping(value="root-shared"))
    finally:
        reset_active_execution_budget(token)

    assert ledger.snapshot_root_available().max_tool_calls == 88


@pytest.mark.asyncio
async def test_reserved_child_regression_release_unused() -> None:
    ledger = _ledger(100)
    child_runner = ChildExecutionRunner[Ping, Pong](ledger=ledger)
    root_id = mint_execution_id()

    class ChildDelegate:
        async def execute(self, request: Ping) -> Pong:
            from intergrax.runtime.execution.active_execution_budget import (
                require_active_execution_budget,
            )
            from intergrax.contracts.execution_identity import require_active_execution_id

            state = require_active_execution_budget()
            state.ledger.consume_budget(
                require_active_execution_id(),
                BudgetUsageTotals(tool_calls=8),
            )
            return Pong(value=request.value)

    class RootDelegate:
        async def execute(self, request: Ping) -> Pong:
            return await child_runner.execute(
                request=request,
                delegate=ChildDelegate(),
                requested_budget=RunBudget(max_tool_calls=30),
            )

    token = bind_root_execution_budget(execution_id=root_id, ledger=ledger)
    try:
        await ExecutionBoundary[Ping, Pong](
            RootDelegate(),
            identity=_root_identity(root_id),
            authority=ParentExecutionAuthority.scoped(("read",)),
        ).execute(Ping(value="reserved"))
    finally:
        reset_active_execution_budget(token)

    assert ledger.snapshot_root_available().max_tool_calls == 92
