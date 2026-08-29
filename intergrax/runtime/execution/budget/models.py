# © Artur Czarnecki. All rights reserved.

"""Execution budget allocation models (UE-8B1)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from intergrax.contracts.execution_identity import ExecutionId
from intergrax.runtime.nexus.budget.budget_models import RunBudget

# UE-8B2: BudgetEnforcer, budget_ticks, and LLM/tool/RAG/websearch usage counters
# must feed this canonical ledger so runtime consumption is globally accounted.
# Do not maintain duplicate permanent counters outside ExecutionBudgetLedger.


class ExecutionBudgetAllocationMode(Enum):
    """How a child execution participates in hierarchical budget accounting."""

    SHARED = "shared"
    RESERVED = "reserved"


@dataclass(frozen=True, slots=True)
class BudgetUsageTotals:
    """Finite consumption totals across all RunBudget dimensions."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0
    tool_calls: int = 0
    rag_invocations: int = 0
    websearch_invocations: int = 0
    wall_time_seconds: float = 0.0
    planner_iterations: int = 0
    replans: int = 0

    def add(self, other: BudgetUsageTotals) -> BudgetUsageTotals:
        return BudgetUsageTotals(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            llm_calls=self.llm_calls + other.llm_calls,
            tool_calls=self.tool_calls + other.tool_calls,
            rag_invocations=self.rag_invocations + other.rag_invocations,
            websearch_invocations=self.websearch_invocations + other.websearch_invocations,
            wall_time_seconds=self.wall_time_seconds + other.wall_time_seconds,
            planner_iterations=self.planner_iterations + other.planner_iterations,
            replans=self.replans + other.replans,
        )

    def subtract(self, other: BudgetUsageTotals) -> BudgetUsageTotals:
        return BudgetUsageTotals(
            input_tokens=self.input_tokens - other.input_tokens,
            output_tokens=self.output_tokens - other.output_tokens,
            total_tokens=self.total_tokens - other.total_tokens,
            llm_calls=self.llm_calls - other.llm_calls,
            tool_calls=self.tool_calls - other.tool_calls,
            rag_invocations=self.rag_invocations - other.rag_invocations,
            websearch_invocations=self.websearch_invocations - other.websearch_invocations,
            wall_time_seconds=self.wall_time_seconds - other.wall_time_seconds,
            planner_iterations=self.planner_iterations - other.planner_iterations,
            replans=self.replans - other.replans,
        )


def run_budget_to_usage_totals(budget: RunBudget) -> BudgetUsageTotals:
    """Map explicit finite RunBudget limits to usage totals (None dimensions become zero)."""
    return BudgetUsageTotals(
        input_tokens=budget.max_input_tokens or 0,
        output_tokens=budget.max_output_tokens or 0,
        total_tokens=budget.max_total_tokens or 0,
        llm_calls=budget.max_llm_calls or 0,
        tool_calls=budget.max_tool_calls or 0,
        rag_invocations=budget.max_rag_invocations or 0,
        websearch_invocations=budget.max_websearch_invocations or 0,
        wall_time_seconds=budget.max_wall_time_seconds or 0.0,
        planner_iterations=budget.max_planner_iterations or 0,
        replans=budget.max_replans or 0,
    )


def usage_totals_to_run_budget(totals: BudgetUsageTotals) -> RunBudget:
    """Map usage totals back to RunBudget with explicit finite fields."""
    return RunBudget(
        max_input_tokens=totals.input_tokens,
        max_output_tokens=totals.output_tokens,
        max_total_tokens=totals.total_tokens,
        max_llm_calls=totals.llm_calls,
        max_tool_calls=totals.tool_calls,
        max_rag_invocations=totals.rag_invocations,
        max_websearch_invocations=totals.websearch_invocations,
        max_wall_time_seconds=totals.wall_time_seconds,
        max_planner_iterations=totals.planner_iterations,
        max_replans=totals.replans,
    )


@dataclass(frozen=True, slots=True)
class ChildBudgetAllocationContext:
    """Inputs for resolving child execution budget allocation."""

    parent_execution_id: ExecutionId
    parent_allocation_mode: ExecutionBudgetAllocationMode
    parent_reservation_remaining: RunBudget | None
    requested_budget: RunBudget | None


@dataclass(frozen=True, slots=True)
class ChildBudgetAllocationDecision:
    """Policy intent for child budget participation (validated by the ledger)."""

    mode: ExecutionBudgetAllocationMode
    reservation_request: RunBudget | None = None


@dataclass(frozen=True, slots=True)
class ExecutionBudgetReservationGrant:
    """Ledger-validated grant for a child execution."""

    execution_id: ExecutionId
    parent_execution_id: ExecutionId
    mode: ExecutionBudgetAllocationMode
    reservation_allowance: RunBudget | None


class ExecutionBudgetError(RuntimeError):
    """Raised when budget allocation, reservation, or consumption is invalid."""


class ExecutionBudgetReservationError(ExecutionBudgetError):
    """Raised when a reservation cannot be granted or released."""
