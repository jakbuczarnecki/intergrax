# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Cost and duration extraction from runtime answers and agent results (Phase D.5)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from intergrax.contracts.agent_execution_result import AgentExecutionResult
from intergrax.runtime.nexus.responses.response_schema import RuntimeAnswer


def tokens_to_cost_units(total_tokens: int) -> float:
    """
    Laboratory cost proxy: one cost unit per LLM token.

    Matches legacy ``EvalRunner`` behaviour until provider-specific pricing is wired.
    """
    return float(max(0, int(total_tokens)))


def extract_cost_from_runtime_answer(answer: RuntimeAnswer) -> Optional[float]:
    """Derive cost from explicit stats or LLM usage report on a RuntimeAnswer."""
    stats = answer.stats
    explicit = stats.extra.get("cost") if stats else None
    if explicit is not None:
        try:
            return float(explicit)
        except (TypeError, ValueError):
            pass

    report = answer.llm_usage_report
    if report is not None:
        return tokens_to_cost_units(report.total.total_tokens)

    if stats and stats.total_tokens is not None:
        return tokens_to_cost_units(stats.total_tokens)

    return None


def extract_duration_seconds_from_runtime_answer(answer: RuntimeAnswer) -> Optional[float]:
    """Derive wall-clock duration in seconds from runtime stats."""
    stats = answer.stats
    if stats and stats.duration_ms is not None:
        return stats.duration_ms / 1000.0

    report = answer.llm_usage_report
    if report is not None and report.total.duration_ms:
        return report.total.duration_ms / 1000.0

    return None


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
