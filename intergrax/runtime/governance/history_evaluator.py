# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations
from typing import List

from intergrax.runtime.governance.history_policy_config import HistoryPolicyConfig
from intergrax.runtime.replay.metrics import ExecutionMetrics
from intergrax.runtime.replay.history import RunHistoryAnalyzer, RunHistorySummary
from intergrax.runtime.replay.regression import RegressionSignals


class HistoryAwareEvaluator:
    """
    Enriches regression signals using historical behavior trends.
    """

    def __init__(self, config: HistoryPolicyConfig) -> None:
        self._config = config
        self._history_analyzer = RunHistoryAnalyzer()

    def evaluate(
        self,
        current: ExecutionMetrics,
        previous_runs: List[ExecutionMetrics],
    ) -> RegressionSignals:

        if not previous_runs:
            return RegressionSignals()

        history = self._history_analyzer.analyze(previous_runs)

        return RegressionSignals(
            llm_cost_spike=(
                self._config.llm_spike_ratio is not None
                and current.total_llm_calls > history.avg_llm_calls * self._config.llm_spike_ratio
            ),
            tool_usage_drop=(
                self._config.tool_drop_ratio is not None
                and current.total_tool_calls < history.avg_tool_calls * self._config.tool_drop_ratio
            ),
            step_explosion=(
                self._config.step_spike_ratio is not None
                and current.step_count > history.avg_steps * self._config.step_spike_ratio
            ),
        )
