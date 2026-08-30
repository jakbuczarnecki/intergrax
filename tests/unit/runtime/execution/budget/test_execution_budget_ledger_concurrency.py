# © Artur Czarnecki. All rights reserved.

"""UE-8B1 — parallel reservation concurrency tests for execution budget ledger."""

from __future__ import annotations

import threading

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


def test_parallel_reservations_do_not_oversubscribe_root() -> None:
    ledger = create_execution_budget_ledger(RunBudget(max_tool_calls=100))
    parent = mint_execution_id()
    successes: list[object] = []
    failures: list[Exception] = []
    barrier = threading.Barrier(2)

    def _attempt() -> None:
        barrier.wait()
        child = mint_execution_id()
        try:
            ledger.grant_child_budget(
                execution_id=child,
                parent_execution_id=parent,
                decision=ChildBudgetAllocationDecision(
                    mode=ExecutionBudgetAllocationMode.RESERVED,
                    reservation_request=RunBudget(max_tool_calls=70),
                ),
            )
        except ExecutionBudgetReservationError as exc:
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
    assert ledger.snapshot_root_available().max_tool_calls == 30
