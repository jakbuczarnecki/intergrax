# © Artur Czarnecki. All rights reserved.

"""UE-8B1 — canonical execution budget ledger unit tests."""

from __future__ import annotations

import pytest

from intergrax.contracts.execution_identity import mint_execution_id
from intergrax.runtime.execution.budget.ledger import create_execution_budget_ledger
from intergrax.runtime.execution.budget.models import (
    BudgetUsageTotals,
    ChildBudgetAllocationDecision,
    ExecutionBudgetAllocationMode,
    ExecutionBudgetError,
    ExecutionBudgetReservationError,
)
from intergrax.runtime.nexus.budget.budget_models import RunBudget

pytestmark = pytest.mark.unit


def _parent_id() -> object:
    return mint_execution_id()


def _child_id() -> object:
    return mint_execution_id()


def test_reserve_reduces_root_available() -> None:
    ledger = create_execution_budget_ledger(RunBudget(max_tool_calls=100))
    parent = _parent_id()
    child = _child_id()
    assert ledger.snapshot_root_available().max_tool_calls == 100
    ledger.grant_child_budget(
        execution_id=child,
        parent_execution_id=parent,
        decision=ChildBudgetAllocationDecision(
            mode=ExecutionBudgetAllocationMode.RESERVED,
            reservation_request=RunBudget(max_tool_calls=30),
        ),
    )
    assert ledger.snapshot_root_available().max_tool_calls == 70


def test_consumption_reduces_reservation_remaining() -> None:
    ledger = create_execution_budget_ledger(RunBudget(max_tool_calls=100))
    parent = _parent_id()
    child = _child_id()
    ledger.grant_child_budget(
        execution_id=child,
        parent_execution_id=parent,
        decision=ChildBudgetAllocationDecision(
            mode=ExecutionBudgetAllocationMode.RESERVED,
            reservation_request=RunBudget(max_tool_calls=30),
        ),
    )
    ledger.consume_budget(child, BudgetUsageTotals(tool_calls=8))
    assert ledger.snapshot_reservation_remaining(child).max_tool_calls == 22


def test_release_unused_reservation_on_completion() -> None:
    ledger = create_execution_budget_ledger(RunBudget(max_tool_calls=100))
    parent = _parent_id()
    child = _child_id()
    ledger.grant_child_budget(
        execution_id=child,
        parent_execution_id=parent,
        decision=ChildBudgetAllocationDecision(
            mode=ExecutionBudgetAllocationMode.RESERVED,
            reservation_request=RunBudget(max_tool_calls=30),
        ),
    )
    ledger.consume_budget(child, BudgetUsageTotals(tool_calls=8))
    ledger.release_child_budget(child)
    assert ledger.snapshot_root_available().max_tool_calls == 92


def test_two_children_reserve_without_oversubscribing() -> None:
    ledger = create_execution_budget_ledger(RunBudget(max_tool_calls=100))
    parent = _parent_id()
    child_a = _child_id()
    child_b = _child_id()
    ledger.grant_child_budget(
        execution_id=child_a,
        parent_execution_id=parent,
        decision=ChildBudgetAllocationDecision(
            mode=ExecutionBudgetAllocationMode.RESERVED,
            reservation_request=RunBudget(max_tool_calls=30),
        ),
    )
    ledger.grant_child_budget(
        execution_id=child_b,
        parent_execution_id=parent,
        decision=ChildBudgetAllocationDecision(
            mode=ExecutionBudgetAllocationMode.RESERVED,
            reservation_request=RunBudget(max_tool_calls=20),
        ),
    )
    assert ledger.snapshot_root_available().max_tool_calls == 50


def test_reservation_greater_than_available_fails_closed() -> None:
    ledger = create_execution_budget_ledger(RunBudget(max_tool_calls=100))
    parent = _parent_id()
    child = _child_id()
    ledger.grant_child_budget(
        execution_id=child,
        parent_execution_id=parent,
        decision=ChildBudgetAllocationDecision(
            mode=ExecutionBudgetAllocationMode.RESERVED,
            reservation_request=RunBudget(max_tool_calls=70),
        ),
    )
    with pytest.raises(ExecutionBudgetReservationError, match="exceeds available"):
        ledger.grant_child_budget(
            execution_id=_child_id(),
            parent_execution_id=parent,
            decision=ChildBudgetAllocationDecision(
                mode=ExecutionBudgetAllocationMode.RESERVED,
                reservation_request=RunBudget(max_tool_calls=70),
            ),
        )


def test_unlimited_dimension_stays_unlimited() -> None:
    ledger = create_execution_budget_ledger(RunBudget(max_tool_calls=100))
    parent = _parent_id()
    child = _child_id()
    ledger.grant_child_budget(
        execution_id=child,
        parent_execution_id=parent,
        decision=ChildBudgetAllocationDecision(
            mode=ExecutionBudgetAllocationMode.RESERVED,
            reservation_request=RunBudget(max_tool_calls=30),
        ),
    )
    assert ledger.snapshot_root_available().max_llm_calls is None


def test_consumption_cannot_exceed_effective_grant() -> None:
    ledger = create_execution_budget_ledger(RunBudget(max_tool_calls=100))
    parent = _parent_id()
    child = _child_id()
    ledger.grant_child_budget(
        execution_id=child,
        parent_execution_id=parent,
        decision=ChildBudgetAllocationDecision(
            mode=ExecutionBudgetAllocationMode.RESERVED,
            reservation_request=RunBudget(max_tool_calls=10),
        ),
    )
    with pytest.raises(ExecutionBudgetError, match="exceeds effective grant"):
        ledger.consume_budget(child, BudgetUsageTotals(tool_calls=11))


def test_double_release_fails_closed() -> None:
    ledger = create_execution_budget_ledger(RunBudget(max_tool_calls=100))
    parent = _parent_id()
    child = _child_id()
    ledger.grant_child_budget(
        execution_id=child,
        parent_execution_id=parent,
        decision=ChildBudgetAllocationDecision(
            mode=ExecutionBudgetAllocationMode.RESERVED,
            reservation_request=RunBudget(max_tool_calls=10),
        ),
    )
    ledger.release_child_budget(child)
    with pytest.raises(ExecutionBudgetReservationError, match="already released"):
        ledger.release_child_budget(child)


def test_unknown_reservation_cannot_mutate_ledger() -> None:
    ledger = create_execution_budget_ledger(RunBudget(max_tool_calls=100))
    unknown = _child_id()
    with pytest.raises(ExecutionBudgetReservationError, match="unknown"):
        ledger.release_child_budget(unknown)
    with pytest.raises(ExecutionBudgetReservationError, match="unknown"):
        ledger.consume_budget(unknown, BudgetUsageTotals(tool_calls=1))


def test_shared_under_reserved_immediate_backing_debit() -> None:
    ledger = create_execution_budget_ledger(RunBudget(max_tool_calls=100))
    parent = _parent_id()
    reserved = _child_id()
    ledger.grant_child_budget(
        execution_id=reserved,
        parent_execution_id=parent,
        decision=ChildBudgetAllocationDecision(
            mode=ExecutionBudgetAllocationMode.RESERVED,
            reservation_request=RunBudget(max_tool_calls=10),
        ),
    )
    shared = _child_id()
    ledger.grant_child_budget(
        execution_id=shared,
        parent_execution_id=reserved,
        decision=ChildBudgetAllocationDecision(mode=ExecutionBudgetAllocationMode.SHARED),
    )
    ledger.consume_budget(shared, BudgetUsageTotals(tool_calls=8))
    assert ledger.snapshot_reservation_remaining(reserved).max_tool_calls == 2
    with pytest.raises(ExecutionBudgetError, match="exceeds effective grant"):
        ledger.consume_budget(shared, BudgetUsageTotals(tool_calls=3))
