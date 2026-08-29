# © Artur Czarnecki. All rights reserved.

"""Trusted runtime carrier for active execution budget during governed work (UE-8B1)."""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass

from intergrax.contracts.execution_identity import ExecutionId
from intergrax.runtime.execution.budget.ledger import ExecutionBudgetLedger
from intergrax.runtime.execution.budget.models import ExecutionBudgetAllocationMode
from intergrax.runtime.nexus.budget.budget_models import RunBudget


@dataclass(frozen=True, slots=True)
class ActiveExecutionBudgetState:
    """Effective budget allocation for the active execution."""

    execution_id: ExecutionId
    mode: ExecutionBudgetAllocationMode
    ledger: ExecutionBudgetLedger
    reservation_allowance: RunBudget | None = None


_active_execution_budget: ContextVar[ActiveExecutionBudgetState | None] = ContextVar(
    "active_execution_budget",
    default=None,
)


def bind_active_execution_budget(state: ActiveExecutionBudgetState) -> Token:
    return _active_execution_budget.set(state)


def reset_active_execution_budget(token: Token) -> None:
    _active_execution_budget.reset(token)


def peek_active_execution_budget() -> ActiveExecutionBudgetState | None:
    return _active_execution_budget.get()


def require_active_execution_budget() -> ActiveExecutionBudgetState:
    state = peek_active_execution_budget()
    if state is None:
        raise RuntimeError("active execution budget required")
    return state


def bind_root_execution_budget(
    *,
    execution_id: ExecutionId,
    ledger: ExecutionBudgetLedger,
) -> Token:
    """Bind the canonical per-Run ledger at root execution entry."""
    return bind_active_execution_budget(
        ActiveExecutionBudgetState(
            execution_id=execution_id,
            mode=ExecutionBudgetAllocationMode.SHARED,
            ledger=ledger,
        )
    )
