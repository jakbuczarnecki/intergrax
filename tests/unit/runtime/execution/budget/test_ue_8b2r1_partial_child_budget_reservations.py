# © Artur Czarnecki. All rights reserved.

"""UE-8B2R1 — partial child budget reservation semantics."""

from __future__ import annotations

import pytest

from intergrax.contracts.execution_identity import mint_execution_id
from intergrax.runtime.execution.budget.ledger import create_execution_budget_ledger
from intergrax.runtime.execution.budget.models import (
    ChildBudgetAllocationDecision,
    ExecutionBudgetAllocationMode,
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


def test_partial_tools_only_reservation_passes() -> None:
    ledger = create_execution_budget_ledger(
        RunBudget(max_total_tokens=1000, max_tool_calls=100)
    )
    parent = _parent_id()
    child = _child_id()
    _grant_reserved(
        ledger,
        parent=parent,
        child=child,
        reservation_request=RunBudget(max_tool_calls=10),
    )
    root = ledger.snapshot_root_available()
    assert root.max_total_tokens == 1000
    assert root.max_tool_calls == 90
    allowance = ledger.snapshot_reservation_remaining(child)
    assert allowance.max_tool_calls == 10
    assert allowance.max_total_tokens == 0


def test_partial_tokens_only_reservation_passes() -> None:
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
    root = ledger.snapshot_root_available()
    assert root.max_total_tokens == 800
    assert root.max_tool_calls == 100
    allowance = ledger.snapshot_reservation_remaining(child)
    assert allowance.max_total_tokens == 200
    assert allowance.max_tool_calls == 0


def test_tokens_reservation_above_root_fails() -> None:
    ledger = create_execution_budget_ledger(
        RunBudget(max_total_tokens=1000, max_tool_calls=100)
    )
    with pytest.raises(ExecutionBudgetReservationError, match="exceeds available"):
        _grant_reserved(
            ledger,
            parent=_parent_id(),
            child=_child_id(),
            reservation_request=RunBudget(max_total_tokens=1200),
        )


def test_tools_reservation_above_root_fails() -> None:
    ledger = create_execution_budget_ledger(
        RunBudget(max_total_tokens=1000, max_tool_calls=100)
    )
    with pytest.raises(ExecutionBudgetReservationError, match="exceeds available"):
        _grant_reserved(
            ledger,
            parent=_parent_id(),
            child=_child_id(),
            reservation_request=RunBudget(max_tool_calls=101),
        )


def test_mixed_partial_reservation_passes() -> None:
    ledger = create_execution_budget_ledger(
        RunBudget(max_total_tokens=1000, max_tool_calls=100)
    )
    parent = _parent_id()
    child = _child_id()
    _grant_reserved(
        ledger,
        parent=parent,
        child=child,
        reservation_request=RunBudget(max_total_tokens=200, max_tool_calls=10),
    )
    root = ledger.snapshot_root_available()
    assert root.max_total_tokens == 800
    assert root.max_tool_calls == 90
    allowance = ledger.snapshot_reservation_remaining(child)
    assert allowance.max_total_tokens == 200
    assert allowance.max_tool_calls == 10


def test_root_unlimited_dimension_stays_unlimited_after_partial_reservation() -> None:
    ledger = create_execution_budget_ledger(
        RunBudget(max_total_tokens=1000, max_tool_calls=100)
    )
    parent = _parent_id()
    child = _child_id()
    _grant_reserved(
        ledger,
        parent=parent,
        child=child,
        reservation_request=RunBudget(max_tool_calls=10),
    )
    assert ledger.snapshot_root_available().max_llm_calls is None


def test_unspecified_child_dimension_is_not_unlimited_reservation() -> None:
    ledger = create_execution_budget_ledger(
        RunBudget(max_total_tokens=1000, max_tool_calls=100)
    )
    child = _child_id()
    _grant_reserved(
        ledger,
        parent=_parent_id(),
        child=child,
        reservation_request=RunBudget(max_tool_calls=10),
    )
    allowance = ledger.snapshot_reservation_remaining(child)
    assert allowance.max_total_tokens == 0
    assert allowance.max_total_tokens is not None
    assert allowance.max_llm_calls == 0
    assert allowance.max_llm_calls is not None
