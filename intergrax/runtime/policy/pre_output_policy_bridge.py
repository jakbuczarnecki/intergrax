# © Artur Czarnecki. All rights reserved.

"""Pre-output policy enforcement on Nexus finalization (AUDIT-IDEAL-5.1)."""

from __future__ import annotations

from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.runtime.policy.policy_engine import PolicyEngine
from intergrax.runtime.task.task import Task


def evaluate_pre_output_for_task(
    policy_engine: PolicyEngine,
    task: Task,
    *,
    answer: str,
) -> PolicyDecision:
    agent_id = task.agent_id or "unknown"
    return policy_engine.evaluate_pre_output(
        tenant_id=task.tenant_id,
        agent_id=agent_id,
        output_chars=len(answer or ""),
    )


def apply_pre_output_policy(
    policy_engine: PolicyEngine,
    task: Task,
    *,
    answer: str,
) -> tuple[str, PolicyDecision]:
    decision = evaluate_pre_output_for_task(policy_engine, task, answer=answer)
    if decision.action is PolicyAction.DENY:
        return "[POLICY_BLOCKED] Output blocked by pre-output policy.", decision
    return answer, decision
