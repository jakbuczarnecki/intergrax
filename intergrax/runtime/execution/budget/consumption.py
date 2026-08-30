# © Artur Czarnecki. All rights reserved.

"""Canonical runtime budget consumption helpers (UE-8B2)."""

from __future__ import annotations

from intergrax.contracts.execution_identity import peek_active_execution_id, require_active_execution_id
from intergrax.runtime.execution.active_execution_budget import require_active_execution_budget
from intergrax.runtime.execution.budget.ledger import ROOT_BUDGET_POOL_PARENT
from intergrax.runtime.execution.budget.models import BudgetUsageTotals
from intergrax.runtime.execution.budget.wall_time_checkpoint import wall_time_delta_since_checkpoint


def _consume_governed(amounts: BudgetUsageTotals) -> None:
    if peek_active_execution_id() is None:
        return
    budget_state = require_active_execution_budget()
    execution_id = require_active_execution_id()
    ledger = budget_state.ledger
    ledger.ensure_shared_participant(
        execution_id,
        parent_execution_id=ROOT_BUDGET_POOL_PARENT,
    )
    ledger.consume_budget(execution_id, amounts)


def consume_llm_call() -> None:
    """Charge one real provider/model invocation."""
    _consume_governed(BudgetUsageTotals(llm_calls=1))


def consume_llm_token_usage(
    *,
    input_tokens: int,
    output_tokens: int,
    total_tokens: int,
) -> None:
    """Charge normalized token usage for one completed provider call."""
    _consume_governed(
        BudgetUsageTotals(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )
    )


def consume_tool_call() -> None:
    """Charge one real tool invocation."""
    _consume_governed(BudgetUsageTotals(tool_calls=1))


def consume_rag_invocation() -> None:
    """Charge one real RAG retrieval invocation."""
    _consume_governed(BudgetUsageTotals(rag_invocations=1))


def consume_websearch_invocation() -> None:
    """Charge one real websearch invocation."""
    _consume_governed(BudgetUsageTotals(websearch_invocations=1))


def consume_planner_iteration() -> None:
    """Charge one bounded planner iteration."""
    _consume_governed(BudgetUsageTotals(planner_iterations=1))


def consume_replan() -> None:
    """Charge one real replan request."""
    _consume_governed(BudgetUsageTotals(replans=1))


def consume_wall_time_delta(elapsed_seconds: float) -> None:
    """Charge only positive wall-time delta since the last canonical accounting point."""
    delta = wall_time_delta_since_checkpoint(elapsed_seconds)
    if delta <= 0.0:
        return
    _consume_governed(BudgetUsageTotals(wall_time_seconds=delta))
