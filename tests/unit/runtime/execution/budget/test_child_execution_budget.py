# © Artur Czarnecki. All rights reserved.

"""UE-8B1 — ChildExecutionRunner budget allocation checkpoint tests."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
)
from intergrax.runtime.execution.active_execution_budget import (
    peek_active_execution_budget,
    require_active_execution_budget,
)
from intergrax.runtime.execution.boundary import ExecutionBoundary, ExecutionIdentityBinding
from intergrax.runtime.execution.budget.ledger import create_execution_budget_ledger
from intergrax.runtime.execution.budget.models import (
    BudgetUsageTotals,
    ChildBudgetAllocationContext,
    ChildBudgetAllocationDecision,
    ExecutionBudgetAllocationMode,
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


def _root_identity() -> ExecutionIdentityBinding:
    return ExecutionIdentityBinding(
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )


def _ledger(tool_calls: int | None = None) -> object:
    return create_execution_budget_ledger(RunBudget(max_tool_calls=tool_calls))


@pytest.mark.asyncio
async def test_root_shared_child_shared_nested_shared() -> None:
    ledger = _ledger(100)
    child_runner = ChildExecutionRunner[Ping, Pong](ledger=ledger)
    nested_modes: list[ExecutionBudgetAllocationMode] = []

    class GrandchildDelegate:
        async def execute(self, request: Ping) -> Pong:
            nested_modes.append(require_active_execution_budget().mode)
            return Pong(value=request.value)

    class ChildDelegate:
        async def execute(self, request: Ping) -> Pong:
            assert require_active_execution_budget().mode is ExecutionBudgetAllocationMode.SHARED
            return await child_runner.execute(
                request=request,
                delegate=GrandchildDelegate(),
            )

    class RootDelegate:
        async def execute(self, request: Ping) -> Pong:
            return await child_runner.execute(
                request=request,
                delegate=ChildDelegate(),
            )

    await ExecutionBoundary[Ping, Pong](
        RootDelegate(),
        identity=_root_identity(),
        authority=ParentExecutionAuthority.scoped(("read",)),
    ).execute(Ping(value="nested"))

    assert nested_modes == [ExecutionBudgetAllocationMode.SHARED]
    assert ledger.snapshot_root_available().max_tool_calls == 100


@pytest.mark.asyncio
async def test_reserved_child_and_nested_reservation_bounded() -> None:
    ledger = _ledger(100)
    child_runner = ChildExecutionRunner[Ping, Pong](ledger=ledger)
    nested_allowance: list[int | None] = []
    root_available_during_nested: list[int | None] = []

    class GrandchildDelegate:
        async def execute(self, request: Ping) -> Pong:
            state = require_active_execution_budget()
            assert state.mode is ExecutionBudgetAllocationMode.RESERVED
            nested_allowance.append(
                state.reservation_allowance.max_tool_calls
                if state.reservation_allowance is not None
                else None
            )
            root_available_during_nested.append(
                ledger.snapshot_root_available().max_tool_calls
            )
            return Pong(value=request.value)

    class ChildDelegate:
        async def execute(self, request: Ping) -> Pong:
            assert (
                require_active_execution_budget().mode
                is ExecutionBudgetAllocationMode.RESERVED
            )
            return await child_runner.execute(
                request=request,
                delegate=GrandchildDelegate(),
                requested_budget=RunBudget(max_tool_calls=10),
            )

    class RootDelegate:
        async def execute(self, request: Ping) -> Pong:
            return await child_runner.execute(
                request=request,
                delegate=ChildDelegate(),
                requested_budget=RunBudget(max_tool_calls=30),
            )

    await ExecutionBoundary[Ping, Pong](
        RootDelegate(),
        identity=_root_identity(),
        authority=ParentExecutionAuthority.scoped(("read",)),
    ).execute(Ping(value="nested"))

    assert nested_allowance == [10]
    assert root_available_during_nested == [80]


@pytest.mark.asyncio
async def test_sibling_reservation_isolation() -> None:
    ledger = _ledger(100)
    child_runner = ChildExecutionRunner[Ping, Pong](ledger=ledger)
    available_during: list[int | None] = []

    class ChildDelegate:
        async def execute(self, request: Ping) -> Pong:
            available_during.append(ledger.snapshot_root_available().max_tool_calls)
            return Pong(value=request.value)

    class RootDelegate:
        async def execute(self, request: Ping) -> Pong:
            await child_runner.execute(
                request=Ping(value="a"),
                delegate=ChildDelegate(),
                requested_budget=RunBudget(max_tool_calls=30),
            )
            await child_runner.execute(
                request=Ping(value="b"),
                delegate=ChildDelegate(),
                requested_budget=RunBudget(max_tool_calls=20),
            )
            return Pong(value=request.value)

    await ExecutionBoundary[Ping, Pong](
        RootDelegate(),
        identity=_root_identity(),
        authority=ParentExecutionAuthority.scoped(("read",)),
    ).execute(Ping(value="root"))

    assert available_during == [70, 80]


@pytest.mark.asyncio
async def test_reservation_released_on_child_exception() -> None:
    ledger = _ledger(100)
    child_runner = ChildExecutionRunner[Ping, Pong](ledger=ledger)

    class FailingDelegate:
        async def execute(self, request: Ping) -> Pong:
            raise RuntimeError("child failed")

    class RootDelegate:
        async def execute(self, request: Ping) -> Pong:
            with pytest.raises(RuntimeError, match="child failed"):
                await child_runner.execute(
                    request=request,
                    delegate=FailingDelegate(),
                    requested_budget=RunBudget(max_tool_calls=30),
                )
            return Pong(value="ok")

    await ExecutionBoundary[Ping, Pong](
        RootDelegate(),
        identity=_root_identity(),
        authority=ParentExecutionAuthority.scoped(("read",)),
    ).execute(Ping(value="root"))

    assert ledger.snapshot_root_available().max_tool_calls == 100


@pytest.mark.asyncio
async def test_authority_and_budget_checkpoints_are_independent() -> None:
    ledger = _ledger(100)
    child_runner = ChildExecutionRunner[Ping, Pong](ledger=ledger)
    observed_scopes: list[tuple[str, ...]] = []
    available_during_child: list[int | None] = []

    class ChildDelegate:
        async def execute(self, request: Ping) -> Pong:
            from intergrax.runtime.governance.active_execution_authority import (
                require_active_execution_authority,
            )

            observed_scopes.append(require_active_execution_authority().permission_scopes)
            assert peek_active_execution_budget() is not None
            available_during_child.append(ledger.snapshot_root_available().max_tool_calls)
            return Pong(value=request.value)

    class RootDelegate:
        async def execute(self, request: Ping) -> Pong:
            return await child_runner.execute(
                request=request,
                delegate=ChildDelegate(),
                requested_permission_scopes=("read",),
                requested_budget=RunBudget(max_tool_calls=5),
            )

    await ExecutionBoundary[Ping, Pong](
        RootDelegate(),
        identity=_root_identity(),
        authority=ParentExecutionAuthority.scoped(("read", "write")),
    ).execute(Ping(value="both"))

    assert observed_scopes == [("read",)]
    assert available_during_child == [95]


def test_default_policy_explicit_request_creates_reservation_intent() -> None:
    policy = DefaultSharedPoolBudgetPolicy()
    decision = policy.resolve_child_budget(
        ChildBudgetAllocationContext(
            parent_execution_id=mint_execution_id(),
            parent_allocation_mode=ExecutionBudgetAllocationMode.SHARED,
            parent_reservation_remaining=None,
            requested_budget=RunBudget(max_tool_calls=7),
        )
    )
    assert decision.mode is ExecutionBudgetAllocationMode.RESERVED
    assert decision.reservation_request is not None
    assert decision.reservation_request.max_tool_calls == 7


@pytest.mark.asyncio
async def test_oversubscribed_sibling_reservation_fails_closed() -> None:
    ledger = _ledger(100)
    child_runner = ChildExecutionRunner[Ping, Pong](ledger=ledger)
    first_started = asyncio.Event()
    release_first = asyncio.Event()

    class HoldDelegate:
        async def execute(self, request: Ping) -> Pong:
            first_started.set()
            await release_first.wait()
            return Pong(value=request.value)

    class RootDelegate:
        async def execute(self, request: Ping) -> Pong:
            first_task = asyncio.create_task(
                child_runner.execute(
                    request=Ping(value="first"),
                    delegate=HoldDelegate(),
                    requested_budget=RunBudget(max_tool_calls=70),
                )
            )
            await first_started.wait()
            assert ledger.snapshot_root_available().max_tool_calls == 30
            with pytest.raises(ExecutionBudgetReservationError, match="exceeds available"):
                await child_runner.execute(
                    request=Ping(value="second"),
                    delegate=HoldDelegate(),
                    requested_budget=RunBudget(max_tool_calls=70),
                )
            release_first.set()
            await first_task
            return Pong(value="ok")

    await ExecutionBoundary[Ping, Pong](
        RootDelegate(),
        identity=_root_identity(),
        authority=ParentExecutionAuthority.scoped(("read",)),
    ).execute(Ping(value="root"))

    assert ledger.snapshot_root_available().max_tool_calls == 100