# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations
from dataclasses import dataclass
from typing import List

from intergrax.runtime.replay.metrics import ExecutionMetrics


@dataclass(slots=True)
class RunHistorySummary:
    runs: int = 0
    avg_steps: float = 0.0
    avg_llm_calls: float = 0.0
    avg_tool_calls: float = 0.0
    avg_tokens: float = 0.0
    avg_duration: float = 0.0
    llm_call_trend: float = 0.0
    tool_call_trend: float = 0.0


class RunHistoryAnalyzer:
    """
    Analyzes behavioral trends across multiple runs.
    """

    def analyze(self, metrics_list: List[ExecutionMetrics]) -> RunHistorySummary:
        if not metrics_list:
            return RunHistorySummary()

        runs = len(metrics_list)

        avg_steps = sum(m.step_count for m in metrics_list) / runs
        avg_llm_calls = sum(m.total_llm_calls for m in metrics_list) / runs
        avg_tool_calls = sum(m.total_tool_calls for m in metrics_list) / runs
        avg_tokens = sum(m.total_tokens for m in metrics_list) / runs
        avg_duration = sum(m.duration or 0 for m in metrics_list) / runs

        # trend = last - first
        llm_call_trend = metrics_list[-1].total_llm_calls - metrics_list[0].total_llm_calls
        tool_call_trend = metrics_list[-1].total_tool_calls - metrics_list[0].total_tool_calls

        return RunHistorySummary(
            runs=runs,
            avg_steps=avg_steps,
            avg_llm_calls=avg_llm_calls,
            avg_tool_calls=avg_tool_calls,
            avg_tokens=avg_tokens,
            avg_duration=avg_duration,
            llm_call_trend=llm_call_trend,
            tool_call_trend=tool_call_trend,
        )
