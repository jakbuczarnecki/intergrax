# © Artur Czarnecki. All rights reserved.

"""Execution budget allocation models (UE-8B1)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from intergrax.contracts.execution_identity import ExecutionId
from intergrax.runtime.nexus.budget.budget_models import RunBudget

_UNLIMITED_DIMENSION = 2**62


def _finite_or_unlimited_int(value: int | None) -> int:
    return value if value is not None else _UNLIMITED_DIMENSION


def _finite_or_unlimited_float(value: float | None) -> float:
    return value if value is not None else float(_UNLIMITED_DIMENSION)


def _usage_int_to_optional(value: int) -> int | None:
    return None if value >= _UNLIMITED_DIMENSION else value


def _usage_float_to_optional(value: float) -> float | None:
    return None if value >= float(_UNLIMITED_DIMENSION) else value

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
    """Map RunBudget limits to usage totals (``None`` dimensions are unlimited)."""
    return BudgetUsageTotals(
        input_tokens=_finite_or_unlimited_int(budget.max_input_tokens),
        output_tokens=_finite_or_unlimited_int(budget.max_output_tokens),
        total_tokens=_finite_or_unlimited_int(budget.max_total_tokens),
        llm_calls=_finite_or_unlimited_int(budget.max_llm_calls),
        tool_calls=_finite_or_unlimited_int(budget.max_tool_calls),
        rag_invocations=_finite_or_unlimited_int(budget.max_rag_invocations),
        websearch_invocations=_finite_or_unlimited_int(budget.max_websearch_invocations),
        wall_time_seconds=_finite_or_unlimited_float(budget.max_wall_time_seconds),
        planner_iterations=_finite_or_unlimited_int(budget.max_planner_iterations),
        replans=_finite_or_unlimited_int(budget.max_replans),
    )


def usage_totals_to_run_budget(totals: BudgetUsageTotals) -> RunBudget:
    """Map usage totals back to RunBudget with explicit finite fields."""
    return RunBudget(
        max_input_tokens=_usage_int_to_optional(totals.input_tokens),
        max_output_tokens=_usage_int_to_optional(totals.output_tokens),
        max_total_tokens=_usage_int_to_optional(totals.total_tokens),
        max_llm_calls=_usage_int_to_optional(totals.llm_calls),
        max_tool_calls=_usage_int_to_optional(totals.tool_calls),
        max_rag_invocations=_usage_int_to_optional(totals.rag_invocations),
        max_websearch_invocations=_usage_int_to_optional(totals.websearch_invocations),
        max_wall_time_seconds=_usage_float_to_optional(totals.wall_time_seconds),
        max_planner_iterations=_usage_int_to_optional(totals.planner_iterations),
        max_replans=_usage_int_to_optional(totals.replans),
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
