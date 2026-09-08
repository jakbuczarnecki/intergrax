# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical finite budget consumption totals shared across execution accounting."""

from __future__ import annotations

from dataclasses import dataclass


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
