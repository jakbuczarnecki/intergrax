# © Artur Czarnecki. All rights reserved.

"""UE-8B2R2 — partial reservation shared-backing fallback semantics."""

from __future__ import annotations

import threading

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


def _grant_reserved(
    ledger: object,
    *,
    parent: object,
    child: object,
    reservation_request: RunBudget,
) -> None:
    ledger.grant_child_budget(
        execution_id=child,
        parent_execution_id=parent,
        decision=ChildBudgetAllocationDecision(
            mode=ExecutionBudgetAllocationMode.RESERVED,
            reservation_request=reservation_request,
        ),
    )


def _grant_shared(ledger: object, *, parent: object, child: object) -> None:
    ledger.grant_child_budget(
        execution_id=child,
        parent_execution_id=parent,
        decision=ChildBudgetAllocationDecision(mode=ExecutionBudgetAllocationMode.SHARED),
    )


def test_tools_reserved_tokens_from_shared_pool() -> None:
    ledger = create_execution_budget_ledger(
        RunBudget(max_total_tokens=1000, max_tool_calls=100)
    )
    parent = _parent_id()
    child = _child_id()
    _grant_reserved(ledger, parent=parent, child=child, reservation_request=RunBudget(max_tool_calls=10))
    ledger.consume_budget(
        child,
        BudgetUsageTotals(tool_calls=1, total_tokens=100),
    )
    assert ledger.snapshot_root_available().max_total_tokens == 900
    assert ledger.snapshot_reservation_remaining(child).max_tool_calls == 9


def test_tokens_only_consume_uses_shared_pool() -> None:
    ledger = create_execution_budget_ledger(
        RunBudget(max_total_tokens=1000, max_tool_calls=100)
    )
    parent = _parent_id()
    child = _child_id()
    _grant_reserved(ledger, parent=parent, child=child, reservation_request=RunBudget(max_tool_calls=10))
    ledger.consume_budget(child, BudgetUsageTotals(total_tokens=50))
    assert ledger.snapshot_root_available().max_total_tokens == 950
    assert ledger.snapshot_reservation_remaining(child).max_tool_calls == 10


def test_reserved_tools_exhausted_fails_even_with_root_tools() -> None:
    ledger = create_execution_budget_ledger(
        RunBudget(max_total_tokens=1000, max_tool_calls=100)
    )
    parent = _parent_id()
    child = _child_id()
    _grant_reserved(ledger, parent=parent, child=child, reservation_request=RunBudget(max_tool_calls=10))
    ledger.consume_budget(child, BudgetUsageTotals(tool_calls=10))
    with pytest.raises(ExecutionBudgetError, match="exceeds effective grant"):
        ledger.consume_budget(child, BudgetUsageTotals(tool_calls=1))


def test_tokens_reserved_tools_from_shared_pool() -> None:
    ledger = create_execution_budget_ledger(
        RunBudget(max_total_tokens=1000, max_tool_calls=100)
    )
    parent = _parent_id()
    child = _child_id()
    _grant_reserved(
        ledger,
        parent=parent,
        child=child,
        reservation_request=RunBudget(max_total_tokens=200),
    )
    ledger.consume_budget(
        child,
        BudgetUsageTotals(total_tokens=100, tool_calls=1),
    )
    assert ledger.snapshot_reservation_remaining(child).max_total_tokens == 100
    assert ledger.snapshot_root_available().max_tool_calls == 99


def test_tokens_reservation_exhausted_fails_despite_root_tokens() -> None:
    ledger = create_execution_budget_ledger(
        RunBudget(max_total_tokens=1000, max_tool_calls=100)
    )
    parent = _parent_id()
    child = _child_id()
    _grant_reserved(
        ledger,
        parent=parent,
        child=child,
        reservation_request=RunBudget(max_total_tokens=200),
    )
    ledger.consume_budget(child, BudgetUsageTotals(total_tokens=200))
    with pytest.raises(ExecutionBudgetError, match="exceeds effective grant"):
        ledger.consume_budget(child, BudgetUsageTotals(total_tokens=1))


def test_unreserved_tool_dimension_still_works_from_shared_pool() -> None:
    ledger = create_execution_budget_ledger(
        RunBudget(max_total_tokens=1000, max_tool_calls=100)
    )
    parent = _parent_id()
    child = _child_id()
    _grant_reserved(
        ledger,
        parent=parent,
        child=child,
        reservation_request=RunBudget(max_total_tokens=200),
    )
    ledger.consume_budget(child, BudgetUsageTotals(tool_calls=5))
    assert ledger.snapshot_root_available().max_tool_calls == 95
    assert ledger.snapshot_reservation_remaining(child).max_total_tokens == 200


def test_nested_child_under_partially_reserved_parent() -> None:
    ledger = create_execution_budget_ledger(
        RunBudget(max_total_tokens=1000, max_tool_calls=100)
    )
    root_parent = _parent_id()
    reserved_parent = _child_id()
    nested_child = _child_id()
    _grant_reserved(
        ledger,
        parent=root_parent,
        child=reserved_parent,
        reservation_request=RunBudget(max_tool_calls=10),
    )
    _grant_shared(ledger, parent=reserved_parent, child=nested_child)
    ledger.consume_budget(
        nested_child,
        BudgetUsageTotals(tool_calls=2, total_tokens=100),
    )
    assert ledger.snapshot_reservation_remaining(reserved_parent).max_tool_calls == 8
    assert ledger.snapshot_root_available().max_total_tokens == 900


def test_parallel_siblings_cannot_overspend_shared_fallback() -> None:
    ledger = create_execution_budget_ledger(
        RunBudget(max_total_tokens=100, max_tool_calls=100)
    )
    parent = _parent_id()
    reserved_parent = _child_id()
    _grant_reserved(
        ledger,
        parent=parent,
        child=reserved_parent,
        reservation_request=RunBudget(max_tool_calls=10),
    )
    successes: list[object] = []
    failures: list[Exception] = []
    barrier = threading.Barrier(2)

    def _attempt() -> None:
        child = _child_id()
        barrier.wait()
        try:
            _grant_shared(ledger, parent=reserved_parent, child=child)
            ledger.consume_budget(child, BudgetUsageTotals(total_tokens=70))
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
    assert ledger.snapshot_root_available().max_total_tokens == 30


def test_mixed_consume_no_double_accounting() -> None:
    ledger = create_execution_budget_ledger(
        RunBudget(max_total_tokens=1000, max_tool_calls=100)
    )
    parent = _parent_id()
    child = _child_id()
    _grant_reserved(ledger, parent=parent, child=child, reservation_request=RunBudget(max_tool_calls=10))
    ledger.consume_budget(
        child,
        BudgetUsageTotals(tool_calls=1, total_tokens=100),
    )
    assert ledger.snapshot_root_available().max_total_tokens == 900
    assert ledger.snapshot_root_available().max_tool_calls == 91
    assert ledger.snapshot_reservation_remaining(child).max_tool_calls == 9
    ledger.release_child_budget(child)
    assert ledger.snapshot_root_available().max_total_tokens == 900
    assert ledger.snapshot_root_available().max_tool_calls == 99


def test_existing_full_reservation_still_works() -> None:
    ledger = create_execution_budget_ledger(RunBudget(max_tool_calls=100))
    parent = _parent_id()
    child = _child_id()
    _grant_reserved(
        ledger,
        parent=parent,
        child=child,
        reservation_request=RunBudget(max_tool_calls=30),
    )
    ledger.consume_budget(child, BudgetUsageTotals(tool_calls=8))
    assert ledger.snapshot_reservation_remaining(child).max_tool_calls == 22
    ledger.release_child_budget(child)
    assert ledger.snapshot_root_available().max_tool_calls == 92
