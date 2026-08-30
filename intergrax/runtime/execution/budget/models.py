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


def _finite_or_zero_int(value: int | None) -> int:
    return value if value is not None else 0


def _finite_or_zero_float(value: float | None) -> float:
    return value if value is not None else 0.0


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
    """Map root RunBudget limits to usage totals (``None`` dimensions are unlimited)."""
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


@dataclass(frozen=True, slots=True)
class BudgetReservationScope:
    """Explicit set of RunBudget dimensions included in a partial reservation."""

    input_tokens: bool = False
    output_tokens: bool = False
    total_tokens: bool = False
    llm_calls: bool = False
    tool_calls: bool = False
    rag_invocations: bool = False
    websearch_invocations: bool = False
    wall_time_seconds: bool = False
    planner_iterations: bool = False
    replans: bool = False

    @property
    def is_empty(self) -> bool:
        return not (
            self.input_tokens
            or self.output_tokens
            or self.total_tokens
            or self.llm_calls
            or self.tool_calls
            or self.rag_invocations
            or self.websearch_invocations
            or self.wall_time_seconds
            or self.planner_iterations
            or self.replans
        )

    def select(self, amounts: BudgetUsageTotals) -> BudgetUsageTotals:
        """Keep usage only for dimensions explicitly covered by this scope."""
        return BudgetUsageTotals(
            input_tokens=amounts.input_tokens if self.input_tokens else 0,
            output_tokens=amounts.output_tokens if self.output_tokens else 0,
            total_tokens=amounts.total_tokens if self.total_tokens else 0,
            llm_calls=amounts.llm_calls if self.llm_calls else 0,
            tool_calls=amounts.tool_calls if self.tool_calls else 0,
            rag_invocations=amounts.rag_invocations if self.rag_invocations else 0,
            websearch_invocations=amounts.websearch_invocations
            if self.websearch_invocations
            else 0,
            wall_time_seconds=amounts.wall_time_seconds if self.wall_time_seconds else 0.0,
            planner_iterations=amounts.planner_iterations if self.planner_iterations else 0,
            replans=amounts.replans if self.replans else 0,
        )

    def complement_select(self, amounts: BudgetUsageTotals) -> BudgetUsageTotals:
        """Keep usage only for dimensions not covered by this scope."""
        scoped = self.select(amounts)
        return BudgetUsageTotals(
            input_tokens=amounts.input_tokens - scoped.input_tokens,
            output_tokens=amounts.output_tokens - scoped.output_tokens,
            total_tokens=amounts.total_tokens - scoped.total_tokens,
            llm_calls=amounts.llm_calls - scoped.llm_calls,
            tool_calls=amounts.tool_calls - scoped.tool_calls,
            rag_invocations=amounts.rag_invocations - scoped.rag_invocations,
            websearch_invocations=amounts.websearch_invocations - scoped.websearch_invocations,
            wall_time_seconds=amounts.wall_time_seconds - scoped.wall_time_seconds,
            planner_iterations=amounts.planner_iterations - scoped.planner_iterations,
            replans=amounts.replans - scoped.replans,
        )


@dataclass(frozen=True, slots=True)
class ParsedBudgetReservation:
    """Typed partial reservation: finite allowance plus explicit dimension scope."""

    allowance: BudgetUsageTotals
    scope: BudgetReservationScope


def reservation_scope_from_request(budget: RunBudget) -> BudgetReservationScope:
    """Derive which dimensions a child reservation request explicitly names."""
    return BudgetReservationScope(
        input_tokens=budget.max_input_tokens is not None,
        output_tokens=budget.max_output_tokens is not None,
        total_tokens=budget.max_total_tokens is not None,
        llm_calls=budget.max_llm_calls is not None,
        tool_calls=budget.max_tool_calls is not None,
        rag_invocations=budget.max_rag_invocations is not None,
        websearch_invocations=budget.max_websearch_invocations is not None,
        wall_time_seconds=budget.max_wall_time_seconds is not None,
        planner_iterations=budget.max_planner_iterations is not None,
        replans=budget.max_replans is not None,
    )


def parse_reservation_request(budget: RunBudget) -> ParsedBudgetReservation:
    """Map child reservation request (``None`` dimensions are not reserved)."""
    scope = reservation_scope_from_request(budget)
    return ParsedBudgetReservation(
        allowance=scope.select(
            BudgetUsageTotals(
                input_tokens=_finite_or_zero_int(budget.max_input_tokens),
                output_tokens=_finite_or_zero_int(budget.max_output_tokens),
                total_tokens=_finite_or_zero_int(budget.max_total_tokens),
                llm_calls=_finite_or_zero_int(budget.max_llm_calls),
                tool_calls=_finite_or_zero_int(budget.max_tool_calls),
                rag_invocations=_finite_or_zero_int(budget.max_rag_invocations),
                websearch_invocations=_finite_or_zero_int(budget.max_websearch_invocations),
                wall_time_seconds=_finite_or_zero_float(budget.max_wall_time_seconds),
                planner_iterations=_finite_or_zero_int(budget.max_planner_iterations),
                replans=_finite_or_zero_int(budget.max_replans),
            )
        ),
        scope=scope,
    )


def reservation_request_to_usage_totals(budget: RunBudget) -> BudgetUsageTotals:
    """Map child reservation request (``None`` dimensions are not requested)."""
    return parse_reservation_request(budget).allowance


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


def reservation_remaining_to_run_budget(
    remaining: BudgetUsageTotals,
    scope: BudgetReservationScope,
) -> RunBudget:
    """Map remaining allowance to RunBudget; non-reserved dimensions report ``0``."""
    mapped = usage_totals_to_run_budget(remaining)
    return RunBudget(
        max_input_tokens=mapped.max_input_tokens if scope.input_tokens else 0,
        max_output_tokens=mapped.max_output_tokens if scope.output_tokens else 0,
        max_total_tokens=mapped.max_total_tokens if scope.total_tokens else 0,
        max_llm_calls=mapped.max_llm_calls if scope.llm_calls else 0,
        max_tool_calls=mapped.max_tool_calls if scope.tool_calls else 0,
        max_rag_invocations=mapped.max_rag_invocations if scope.rag_invocations else 0,
        max_websearch_invocations=mapped.max_websearch_invocations
        if scope.websearch_invocations
        else 0,
        max_wall_time_seconds=mapped.max_wall_time_seconds if scope.wall_time_seconds else 0.0,
        max_planner_iterations=mapped.max_planner_iterations if scope.planner_iterations else 0,
        max_replans=mapped.max_replans if scope.replans else 0,
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
