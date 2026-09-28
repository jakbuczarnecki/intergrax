# © Artur Czarnecki. All rights reserved.

"""Pre-output policy enforcement on Nexus finalization (AUDIT-IDEAL-5.1)."""

from __future__ import annotations

from intergrax.contracts.governed_execution_governance_evidence import (
    GovernedExecutionEvaluationPoint,
)
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.runtime.governance.governance_evidence_recorder import (
    GovernanceEvidenceRecorder,
)
from intergrax.runtime.governance.governance_policy_decision_evidence_recording import (
    record_governance_policy_decision_evidence,
)
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
    governance_evidence_recorder: GovernanceEvidenceRecorder | None = None,
) -> tuple[str, PolicyDecision]:
    decision = evaluate_pre_output_for_task(policy_engine, task, answer=answer)
    record_governance_policy_decision_evidence(
        governance_evidence_recorder,
        evaluation_point=GovernedExecutionEvaluationPoint.PRE_OUTPUT,
        decision=decision,
        tenant_id=task.tenant_id,
        workspace_id=str(task.metadata.get("workspace_id") or task.tenant_id),
        principal_id=task.user_id,
        action="terminal_output",
        resource_type="task_output",
        resource_scope=task.agent_id or "",
        digest_payload={
            "task_id": task.task_id,
            "tenant_id": task.tenant_id,
            "agent_id": task.agent_id or "",
            "output_chars": len(answer or ""),
            "policy_rule_id": decision.policy_rule_id or "",
        },
        idempotency_prefix="pre_output",
    )
    if decision.action is PolicyAction.DENY:
        return "[POLICY_BLOCKED] Output blocked by pre-output policy.", decision
    return answer, decision
