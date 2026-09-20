# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Cost and duration aggregation over agent execution results (neutral contracts)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from intergrax.contracts.agent_execution_result import AgentExecutionResult


def tokens_to_cost_units(total_tokens: int) -> float:
    """Laboratory cost proxy: one cost unit per LLM token."""
    return float(max(0, int(total_tokens)))


@dataclass(frozen=True)
class AggregatedExecutionMetrics:
    cost: float
    duration_ms: int
    total_tokens: int

    def as_llm_usage(self) -> Dict[str, Any]:
        return {
            "cost": self.cost,
            "total_tokens": self.total_tokens,
            "duration_ms": self.duration_ms,
        }


def aggregate_execution_metrics(
    executions: List[AgentExecutionResult],
) -> AggregatedExecutionMetrics:
    """Sum agent-level cost/tokens; use max duration across parallel-capable runs."""
    cost = sum(float(execution.cost or 0.0) for execution in executions)
    total_tokens = int(round(cost))
    duration_ms = 0
    for execution in executions:
        if execution.duration_seconds is not None:
            duration_ms = max(duration_ms, int(execution.duration_seconds * 1000))
    return AggregatedExecutionMetrics(
        cost=cost,
        duration_ms=duration_ms,
        total_tokens=total_tokens,
    )
