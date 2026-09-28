# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Project canonical runtime ``PolicyDecision`` values into GR-8 facts (GR-13 adoption)."""

from __future__ import annotations

from intergrax.contracts.evaluated_policy_decision import request_digest_for_payload
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    TaskId,
    peek_active_execution_identity,
    peek_active_execution_task_id,
    require_active_execution_id,
)
from intergrax.contracts.governed_execution_governance_evidence import (
    GovernedExecutionEvaluationPoint,
    build_governance_fact_from_policy_decision,
    is_canonical_governance_evidence_policy_action,
)
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.runtime.governance.active_execution_governance_identity import (
    require_active_execution_governance_identity,
)
from intergrax.runtime.governance.execution_guard import GovernanceEvaluation
from intergrax.runtime.governance.governance_evidence_recorder import (
    GovernanceEvidenceRecorder,
)
from intergrax.runtime.policy.rules.evaluation import PolicyEnforcementDecision
from intergrax.runtime.policy.rules.schema import PolicyRuleAction
from intergrax.runtime.replay.policy import PolicyDecision as ReplayPolicyDecision
from intergrax.runtime.replay.policy import PolicyDecisionType


def _execution_correlation() -> tuple[
    TaskId | None, RunId | None, AttemptId | None, ExecutionId | None
]:
    bound = peek_active_execution_identity()
    if bound is None:
        return None, None, None, None
    run_id, attempt_id = bound
    task_id = peek_active_execution_task_id()
    try:
        execution_id = require_active_execution_id()
    except RuntimeError:
        execution_id = None
    return task_id, run_id, attempt_id, execution_id


def runtime_policy_decision_from_declarative_enforcement(
    decision: PolicyEnforcementDecision,
) -> PolicyDecision:
    if decision.action is PolicyRuleAction.DENY:
        action = PolicyAction.DENY
    elif decision.action is PolicyRuleAction.REQUIRE_HITL:
        action = PolicyAction.REQUIRE_HUMAN
    else:
        action = PolicyAction.ALLOW
    reason = decision.reasons[0] if decision.reasons else decision.action.value
    rule_id = (
        decision.matched_rule_ids[0]
        if decision.matched_rule_ids
        else "declarative.tool_invocation"
    )
    return PolicyDecision(
        action=action,
        reason=reason,
        policy_rule_id=rule_id,
    )


def runtime_policy_decision_from_replay_policy_decision(
    replay_decision: ReplayPolicyDecision,
) -> PolicyDecision:
    if replay_decision.decision is PolicyDecisionType.BLOCK:
        action = PolicyAction.DENY
    elif replay_decision.decision is PolicyDecisionType.WARN:
        action = PolicyAction.ESCALATE
    else:
        action = PolicyAction.ALLOW
    reason = (
        "; ".join(replay_decision.reasons)
        if replay_decision.reasons
        else "post_run_evaluation"
    )
    return PolicyDecision(
        action=action,
        reason=reason,
        policy_rule_id="post_run.governance_evaluation",
    )


def runtime_policy_decision_from_post_run_evaluation(
    evaluation: GovernanceEvaluation,
) -> PolicyDecision:
    return runtime_policy_decision_from_replay_policy_decision(evaluation.decision)


def record_governance_policy_decision_evidence(
    recorder: GovernanceEvidenceRecorder | None,
    *,
    evaluation_point: GovernedExecutionEvaluationPoint,
    decision: PolicyDecision,
    tenant_id: str,
    workspace_id: str,
    principal_id: str,
    action: str,
    resource_type: str = "",
    resource_scope: str = "",
    task_id: TaskId | None = None,
    run_id: RunId | None = None,
    attempt_id: AttemptId | None = None,
    execution_id: ExecutionId | None = None,
    digest_payload: dict[str, str | int | bool],
    idempotency_prefix: str,
) -> None:
    if recorder is None or recorder.persistence is None:
        return
    if not is_canonical_governance_evidence_policy_action(decision.action):
        return
    bound_task_id, bound_run_id, bound_attempt_id, bound_execution_id = (
        _execution_correlation()
    )
    resolved_task_id = task_id if task_id is not None else bound_task_id
    resolved_run_id = run_id if run_id is not None else bound_run_id
    resolved_attempt_id = attempt_id if attempt_id is not None else bound_attempt_id
    resolved_execution_id = (
        execution_id if execution_id is not None else bound_execution_id
    )
    digest = request_digest_for_payload(digest_payload)
    idempotency_key = f"{idempotency_prefix}:{digest}:{decision.action.value}"
    fact = build_governance_fact_from_policy_decision(
        evaluation_point=evaluation_point,
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        principal_id=principal_id,
        decision=decision,
        request_digest=digest,
        idempotency_key=idempotency_key,
        action=action,
        resource_type=resource_type,
        resource_scope=resource_scope,
        task_id=resolved_task_id,
        run_id=resolved_run_id,
        attempt_id=resolved_attempt_id,
        execution_id=resolved_execution_id,
    )
    recorder.record(fact)


def record_tool_plan_or_access_evidence(
    recorder: GovernanceEvidenceRecorder | None,
    *,
    tenant_id: str,
    workspace_id: str,
    principal_id: str,
    agent_id: str,
    requested_tool_count: int,
    allowed_tool_count: int,
    use_tools_requested: bool,
    use_tools_allowed: bool,
) -> None:
    narrowed = allowed_tool_count < requested_tool_count or (
        use_tools_requested and not use_tools_allowed
    )
    decision = PolicyDecision(
        action=PolicyAction.DENY if narrowed else PolicyAction.ALLOW,
        reason="tool_plan_narrowed_by_access_policy"
        if narrowed
        else "tool_plan_allowed",
        policy_rule_id="tool_access_policy.plan",
    )
    record_governance_policy_decision_evidence(
        recorder,
        evaluation_point=GovernedExecutionEvaluationPoint.TOOL_PLAN_OR_ACCESS,
        decision=decision,
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        principal_id=principal_id,
        action="tool_invocation_plan",
        resource_type="tool_plan",
        resource_scope=agent_id,
        digest_payload={
            "agent_id": agent_id,
            "requested_tool_count": requested_tool_count,
            "allowed_tool_count": allowed_tool_count,
            "use_tools_requested": use_tools_requested,
            "use_tools_allowed": use_tools_allowed,
        },
        idempotency_prefix="tool_plan_or_access",
    )


def record_governance_policy_decision_evidence_for_active_identity(
    recorder: GovernanceEvidenceRecorder | None,
    *,
    evaluation_point: GovernedExecutionEvaluationPoint,
    decision: PolicyDecision,
    action: str,
    resource_type: str = "",
    resource_scope: str = "",
    digest_payload: dict[str, str | int | bool],
    idempotency_prefix: str,
    task_id: TaskId | None = None,
    run_id: RunId | None = None,
    attempt_id: AttemptId | None = None,
    execution_id: ExecutionId | None = None,
) -> None:
    identity = require_active_execution_governance_identity()
    record_governance_policy_decision_evidence(
        recorder,
        evaluation_point=evaluation_point,
        decision=decision,
        tenant_id=identity.tenant_id,
        workspace_id=identity.workspace_id,
        principal_id=identity.principal_id,
        action=action,
        resource_type=resource_type,
        resource_scope=resource_scope,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        digest_payload=digest_payload,
        idempotency_prefix=idempotency_prefix,
    )


__all__ = [
    "record_governance_policy_decision_evidence",
    "record_tool_plan_or_access_evidence",
    "record_governance_policy_decision_evidence_for_active_identity",
    "runtime_policy_decision_from_declarative_enforcement",
    "runtime_policy_decision_from_post_run_evaluation",
    "runtime_policy_decision_from_replay_policy_decision",
]
