# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from intergrax.runtime.replay.models import ReconstructedRun


@dataclass(slots=True)
class ExecutionMetrics:
    step_count: int
    total_llm_calls: int
    total_tool_calls: int
    total_artifacts: int
    total_tokens: int
    duration: Optional[float]
    tool_steps_ratio: float
    llm_steps_ratio: float


class ExecutionMetricsEngine:
    """
    Computes behavioral metrics of a reconstructed run.
    """

    def compute(self, run: ReconstructedRun) -> ExecutionMetrics:
        step_count = len(run.steps)
        total_llm_calls = len(run.llm_calls)
        total_tool_calls = len(run.tool_calls)
        total_artifacts = len(run.artifacts)

        total_tokens = sum(call.total_tokens for call in run.llm_calls)

        duration = None
        if run.steps and run.steps[0].started_at and run.steps[-1].finished_at:
            duration = run.steps[-1].finished_at - run.steps[0].started_at

        tool_steps = sum(1 for s in run.steps if s.tool_calls)
        llm_steps = sum(1 for s in run.steps if s.llm_calls)

        tool_steps_ratio = tool_steps / step_count if step_count else 0.0
        llm_steps_ratio = llm_steps / step_count if step_count else 0.0

        return ExecutionMetrics(
            step_count=step_count,
            total_llm_calls=total_llm_calls,
            total_tool_calls=total_tool_calls,
            total_artifacts=total_artifacts,
            total_tokens=total_tokens,
            duration=duration,
            tool_steps_ratio=tool_steps_ratio,
            llm_steps_ratio=llm_steps_ratio,
        )
