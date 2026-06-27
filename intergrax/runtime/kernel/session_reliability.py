# © Artur Czarnecki. All rights reserved.

"""Agent session reliability state wired from Tier-3 ReliabilityProfile (ACP-PROD-4)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.runtime.wiring.reliability_runtime_bridge import ReliabilityWiringOptions
from intergrax.contracts.agent_run_enums import AgentRunErrorCode
from intergrax.contracts.resilience_policy import FailureClass, ResiliencePolicy


_RETRIABLE_CODES = frozenset(
    {
        AgentRunErrorCode.TOOL_FAILED,
        AgentRunErrorCode.LLM_FAILED,
        AgentRunErrorCode.RAG_FAILED,
        AgentRunErrorCode.INTERNAL_ERROR,
    }
)


@dataclass
class AgentSessionReliability:
    """Harness-owned retry and circuit-breaker counters for one agent run."""

    resilience_policy: ResiliencePolicy
    circuit_breaker_failure_threshold: int = 5
    checkpoint_interval_steps: int = 1
    consecutive_failures: int = 0
    circuit_open: bool = False
    attempt_count: int = 0

    @classmethod
    def from_wiring_options(cls, options: ReliabilityWiringOptions) -> AgentSessionReliability:
        return cls(
            resilience_policy=options.resilience_policy,
            circuit_breaker_failure_threshold=options.circuit_breaker_failure_threshold,
            checkpoint_interval_steps=max(1, options.checkpoint_interval_steps),
        )

    def should_checkpoint(self, step_index: int) -> bool:
        if self.checkpoint_interval_steps <= 1:
            return True
        return (step_index + 1) % self.checkpoint_interval_steps == 0

    def is_retriable(self, error_code: AgentRunErrorCode | None) -> bool:
        if error_code is None or self.circuit_open:
            return False
        if error_code not in _RETRIABLE_CODES:
            return False
        if self.attempt_count >= self.resilience_policy.max_attempts:
            return False
        action = self.resilience_policy.action_for(FailureClass.DEPENDENCY_ERROR)
        return action.value in {"retry", "retry_alternate", "retry_run"}

    def record_failure(self, error_code: AgentRunErrorCode | None) -> None:
        if error_code in _RETRIABLE_CODES:
            self.consecutive_failures += 1
            self.attempt_count += 1
        if self.consecutive_failures >= self.circuit_breaker_failure_threshold:
            self.circuit_open = True

    def record_success(self) -> None:
        self.consecutive_failures = 0
        self.attempt_count = 0
