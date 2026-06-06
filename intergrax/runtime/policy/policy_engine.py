# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Unified policy facade (architecture §42.11, Appendix B.03).

Consolidates live runtime governance and optional replay/eval policy behind one entry point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from intergrax.contracts.agent_decision import AgentDecision
from intergrax.contracts.execution_interrupt import ExecutionInterrupt
from intergrax.contracts.runtime_policy import PolicyDecision as RuntimePolicyDecision
from intergrax.runtime.policy.runtime_policy_engine import RuntimePolicyEngine
from intergrax.runtime.replay.metrics import ExecutionMetrics
from intergrax.runtime.replay.policy import (
    ExecutionPolicyEngine,
    PolicyDecision as ReplayPolicyDecision,
    PolicyDecisionType,
)
from intergrax.runtime.replay.policy_config import ExecutionPolicyConfig
from intergrax.runtime.replay.regression import RegressionSignals
from intergrax.runtime.replay.run_diff import RunDiff
from intergrax.llm_adapters.governance.llm_cost import LLMCostEvaluation, evaluate_llm_run_cost

PolicyEngineInput = Union["PolicyEngine", RuntimePolicyEngine, None]


@dataclass
class PolicyEngine:
    """
    Single facade for policy evaluation during live runs and post-run replay.

    - ``evaluate_decision`` / ``evaluate_interrupt`` — live UAEP / Nexus governance
    - ``evaluate_replay`` — experiment / regression guard (optional)
    """

    runtime: RuntimePolicyEngine = field(default_factory=RuntimePolicyEngine)
    replay: Optional[ExecutionPolicyEngine] = None

    @classmethod
    def with_replay_config(cls, config: ExecutionPolicyConfig) -> PolicyEngine:
        return cls(replay=ExecutionPolicyEngine(config))

    def evaluate_decision(
        self,
        decision: AgentDecision,
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> RuntimePolicyDecision:
        return self.runtime.evaluate_decision(decision, context=context)

    def evaluate_interrupt(self, interrupt: ExecutionInterrupt) -> RuntimePolicyDecision:
        return self.runtime.evaluate_interrupt(interrupt)

    def evaluate_pre_llm(
        self,
        *,
        tenant_id: str,
        agent_id: str,
        message_count: int,
        context: Optional[Dict[str, Any]] = None,
    ) -> RuntimePolicyDecision:
        return self.runtime.evaluate_pre_llm(
            tenant_id=tenant_id,
            agent_id=agent_id,
            message_count=message_count,
            context=context,
        )

    def evaluate_pre_output(
        self,
        *,
        tenant_id: str,
        agent_id: str,
        output_chars: int,
        context: Optional[Dict[str, Any]] = None,
    ) -> RuntimePolicyDecision:
        return self.runtime.evaluate_pre_output(
            tenant_id=tenant_id,
            agent_id=agent_id,
            output_chars=output_chars,
            context=context,
        )

    def evaluate_replay(
        self,
        metrics: ExecutionMetrics,
        regression: RegressionSignals,
        diff: RunDiff | None = None,
    ) -> ReplayPolicyDecision:
        if self.replay is None:
            return ReplayPolicyDecision(PolicyDecisionType.ALLOW, [])
        return self.replay.evaluate(metrics, regression, diff=diff)

    def evaluate_llm_cost_on_task_completed(
        self,
        *,
        tenant_id: str,
        run_id: str,
    ) -> tuple[LLMCostEvaluation, ReplayPolicyDecision]:
        """
        Governance signal after task completion (Phase Q-L.6).

        Returns cost evaluation and a replay-style decision (WARN when soft threshold exceeded).
        """
        cost = evaluate_llm_run_cost(tenant_id=tenant_id, run_id=run_id)
        if cost.warn_threshold_exceeded:
            return cost, ReplayPolicyDecision(
                PolicyDecisionType.WARN,
                [f"llm_cost:{r}" for r in cost.reasons],
            )
        return cost, ReplayPolicyDecision(PolicyDecisionType.ALLOW, [])


def coerce_policy_engine(engine: PolicyEngineInput) -> PolicyEngine:
    """Normalize legacy ``RuntimePolicyEngine`` injections to ``PolicyEngine``."""
    if engine is None:
        return PolicyEngine()
    if isinstance(engine, PolicyEngine):
        return engine
    return PolicyEngine(runtime=engine)


def coerce_replay_policy_engine(
    engine: ExecutionPolicyEngine | PolicyEngine,
) -> PolicyEngine:
    """Normalize replay governance input to a facade with ``evaluate_replay`` configured."""
    if isinstance(engine, PolicyEngine):
        if engine.replay is None:
            raise ValueError(
                "PolicyEngine requires replay configuration; use PolicyEngine.with_replay_config(...)"
            )
        return engine
    return PolicyEngine(replay=engine)
