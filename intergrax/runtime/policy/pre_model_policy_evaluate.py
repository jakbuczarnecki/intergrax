# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Neutral PRE_MODEL policy evaluation helpers (no agents/authoring imports)."""

from __future__ import annotations

from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.contracts.runtime_policy_context import PreModelPolicyContext
from intergrax.runtime.policy.policy_engine import PolicyEngine


class PreModelPolicyBlockedError(RuntimeError):
    """Raised when PRE_MODEL policy denies model/provider invocation."""

    def __init__(self, decision: PolicyDecision) -> None:
        self.decision = decision
        super().__init__(decision.reason or "pre_model_policy_denied")


def evaluate_pre_model_policy(
    policy_engine: PolicyEngine,
    *,
    tenant_id: str,
    principal_id: str,
    agent_id: str | None = None,
    message_count: int = 1,
    context: PreModelPolicyContext | None = None,
) -> PolicyDecision:
    return policy_engine.evaluate_pre_llm(
        tenant_id=tenant_id,
        principal_id=principal_id,
        agent_id=agent_id,
        message_count=message_count,
        context=context,
    )
