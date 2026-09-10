# © Artur Czarnecki. All rights reserved.

"""Trusted runtime carrier for active execution budget during governed work (UE-8B1)."""

from __future__ import annotations

import time
from contextvars import ContextVar, Token
from dataclasses import dataclass

from intergrax.contracts.execution_identity import ExecutionId
from intergrax.runtime.execution.budget.wall_time_checkpoint import reset_wall_time_accounting
from intergrax.runtime.execution.budget.ledger import (
    ExecutionBudgetLedger,
    ROOT_BUDGET_POOL_PARENT,
)
from intergrax.runtime.execution.budget.models import ExecutionBudgetAllocationMode
from intergrax.runtime.nexus.budget.budget_models import RunBudget


@dataclass(frozen=True, slots=True)
class ActiveExecutionBudgetState:
    """Effective budget allocation for the active execution."""

    execution_id: ExecutionId
    mode: ExecutionBudgetAllocationMode
    ledger: ExecutionBudgetLedger
    reservation_allowance: RunBudget | None = None
    global_deadline_monotonic: float | None = None


_active_execution_budget: ContextVar[ActiveExecutionBudgetState | None] = ContextVar(
    "active_execution_budget",
    default=None,
)


def bind_active_execution_budget(state: ActiveExecutionBudgetState) -> Token:
    state.ledger.ensure_shared_participant(
        state.execution_id,
        parent_execution_id=ROOT_BUDGET_POOL_PARENT,
    )
    reset_wall_time_accounting()
    return _active_execution_budget.set(state)


def reset_active_execution_budget(token: Token) -> None:
    _active_execution_budget.reset(token)


def peek_active_execution_budget() -> ActiveExecutionBudgetState | None:
    return _active_execution_budget.get()


def peek_active_execution_global_deadline_monotonic() -> float | None:
    """Absolute monotonic deadline for the active execution scope, when configured."""
    state = peek_active_execution_budget()
    if state is None:
        return None
    return state.global_deadline_monotonic


def require_active_execution_budget() -> ActiveExecutionBudgetState:
    state = peek_active_execution_budget()
    if state is None:
        raise RuntimeError("active execution budget required")
    return state


def bind_root_execution_budget(
    *,
    execution_id: ExecutionId,
    ledger: ExecutionBudgetLedger,
    run_budget: RunBudget | None = None,
) -> Token:
    """Bind the canonical per-Run ledger at root execution entry."""
    global_deadline_monotonic: float | None = None
    if run_budget is not None and run_budget.max_wall_time_seconds is not None:
        global_deadline_monotonic = (
            time.monotonic() + run_budget.max_wall_time_seconds
        )
    return bind_active_execution_budget(
        ActiveExecutionBudgetState(
            execution_id=execution_id,
            mode=ExecutionBudgetAllocationMode.SHARED,
            ledger=ledger,
            global_deadline_monotonic=global_deadline_monotonic,
        )
    )
