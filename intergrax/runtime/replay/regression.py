# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations
from dataclasses import dataclass
from typing import List

from intergrax.runtime.replay.run_diff import RunDiff
from intergrax.runtime.replay.history import RunHistorySummary


@dataclass(slots=True)
class RegressionSignals:
    step_explosion: bool = False
    llm_cost_spike: bool = False
    tool_usage_drop: bool = False
    final_answer_changed: bool = False


class RegressionDetector:
    """
    Detects behavioral regressions in agent execution.
    """

    def detect_from_diff(self, diff: RunDiff) -> RegressionSignals:
        return RegressionSignals(
            step_explosion=diff.step_count_changed,
            llm_cost_spike=diff.llm_call_delta > 3,
            tool_usage_drop=diff.tool_call_delta < 0,
            final_answer_changed=diff.final_answer_changed,
        )

    def detect_from_history(self, history: RunHistorySummary) -> RegressionSignals:
        return RegressionSignals(
            llm_cost_spike=history.llm_call_trend > 3,
            tool_usage_drop=history.tool_call_trend < 0,
        )
