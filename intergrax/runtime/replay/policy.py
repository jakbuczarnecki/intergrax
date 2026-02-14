# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import List

from intergrax.runtime.replay.metrics import ExecutionMetrics
from intergrax.runtime.replay.policy_config import ExecutionPolicyConfig
from intergrax.runtime.replay.regression import RegressionSignals
from intergrax.runtime.replay.run_diff import RunDiff


class PolicyDecisionType(str, Enum):
    ALLOW = "allow"
    WARN = "warn"
    BLOCK = "block"


@dataclass(slots=True)
class PolicyDecision:
    decision: PolicyDecisionType
    reasons: List[str]


class ExecutionPolicyEngine:
    """
    Evaluates whether agent execution behavior is acceptable.
    """

    def __init__(self, config: ExecutionPolicyConfig) -> None:
        self._config = config

    def evaluate(
        self,
        metrics: ExecutionMetrics,
        regression: RegressionSignals,
        diff: RunDiff | None = None,
    ) -> PolicyDecision:

        reasons: List[str] = []

        if self._config.max_total_tokens is not None:
            if metrics.total_tokens > self._config.max_total_tokens:
                reasons.append("Token usage too high")

        if self._config.max_llm_call_delta is not None and regression.llm_cost_spike:
            reasons.append("LLM cost spike detected")

        if self._config.min_tool_calls is not None:
            if metrics.total_tool_calls < self._config.min_tool_calls:
                reasons.append("Too few tool calls")

        if self._config.max_steps is not None:
            if metrics.step_count > self._config.max_steps:
                reasons.append("Too many steps")

        if self._config.fail_on_answer_change and regression.final_answer_changed:
            reasons.append("Final answer changed")

        if not reasons:
            return PolicyDecision(PolicyDecisionType.ALLOW, [])

        return PolicyDecision(PolicyDecisionType.WARN, reasons)
