# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Neutral PRE_MODEL policy evaluation and evidence for structured inference (GR-10-R2)."""

from __future__ import annotations

from collections.abc import Sequence

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
)
from intergrax.contracts.inference_profile_id import InferenceProfileId
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.contracts.runtime_policy_context import PreModelPolicyContext
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.runtime.execution.failure_evidence.active_context import (
    peek_active_execution_evidence_context,
)
from intergrax.runtime.execution.lineage.active_lineage import peek_active_execution_lineage
from intergrax.runtime.governance.active_execution_governance_identity import (
    peek_active_execution_governance_identity,
    require_active_execution_governance_identity,
)
from intergrax.runtime.governance.governance_evidence_recorder import GovernanceEvidenceRecorder
from intergrax.runtime.policy.policy_engine import PolicyEngine
from intergrax.runtime.policy.pre_model_policy_bridge import (
    PreModelPolicyBlockedError,
    evaluate_pre_model_policy,
)

_PRE_MODEL_ALLOWED_ACTIONS = frozenset({PolicyAction.ALLOW})
_PRE_MODEL_BLOCKED_ACTIONS = frozenset(
    {
        PolicyAction.DENY,
        PolicyAction.REQUIRE_HUMAN,
        PolicyAction.ESCALATE,
        PolicyAction.MODIFY,
    },
)


class PreModelPolicyConfigurationError(RuntimeError):
    """PRE_MODEL cannot run — missing policy dependency or governance identity."""


def _resolve_tenant_id() -> str:
    governance = peek_active_execution_governance_identity()
    if governance is not None:
        return governance.tenant_id
    lineage = peek_active_execution_lineage()
    if lineage is not None:
        return lineage.scope.tenant_id
    evidence = peek_active_execution_evidence_context()
    if evidence is not None:
        return evidence.tenant_id
    raise PreModelPolicyConfigurationError("pre_model tenant_id unavailable")


def _resolve_governance_identity_for_evidence() -> tuple[str, str, str]:
    try:
        identity = require_active_execution_governance_identity()
    except RuntimeError as exc:
        raise PreModelPolicyConfigurationError(
            "pre_model governance identity unavailable",
        ) from exc
    return identity.tenant_id, identity.workspace_id, identity.principal_id


def _execution_correlation() -> tuple[TaskId | None, RunId | None, AttemptId | None, ExecutionId | None]:
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


def _record_pre_model_evidence(
    recorder: GovernanceEvidenceRecorder | None,
    *,
    tenant_id: str,
    workspace_id: str,
    principal_id: str,
    decision: PolicyDecision,
    request_digest: str,
    idempotency_key: str,
    model_scope: str,
    inference_profile_id: InferenceProfileId | None,
) -> None:
    if recorder is None or recorder.persistence is None:
        return
    if decision.action not in (
        PolicyAction.ALLOW,
        PolicyAction.DENY,
        PolicyAction.REQUIRE_HUMAN,
    ):
        return
    task_id, run_id, attempt_id, execution_id = _execution_correlation()
    resource_scope = model_scope
    if inference_profile_id is not None:
        resource_scope = f"{inference_profile_id}:{model_scope}"
    fact = build_governance_fact_from_policy_decision(
        evaluation_point=GovernedExecutionEvaluationPoint.PRE_MODEL,
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        principal_id=principal_id,
        decision=decision,
        request_digest=request_digest,
        idempotency_key=idempotency_key,
        action="structured_inference.model_invoke",
        resource_type="llm_model",
        resource_scope=resource_scope,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    recorder.record(fact)


def _fail_closed_decision(decision: PolicyDecision) -> None:
    if decision.action is PolicyAction.DENY:
        raise PreModelPolicyBlockedError(decision)
    raise PreModelPolicyBlockedError(
        PolicyDecision(
            action=PolicyAction.DENY,
            reason=f"pre_model_unsupported_action:{decision.action.value}",
            policy_rule_id=decision.policy_rule_id or "platform.pre_model_unsupported_action",
        ),
    )


def enforce_pre_model_before_structured_inference(
    policy_engine: PolicyEngine | None,
    governance_evidence_recorder: GovernanceEvidenceRecorder | None,
    *,
    adapter: LLMAdapter,
    messages: Sequence[ChatMessage],
    inference_profile_id: InferenceProfileId | None,
) -> None:
    """Evaluate PRE_MODEL policy; on ALLOW return; otherwise fail closed before provider I/O."""
    if policy_engine is None:
        raise PreModelPolicyConfigurationError("pre_model policy engine required")

    tenant_id = _resolve_tenant_id()
    tenant_id_ev, workspace_id, principal_id = _resolve_governance_identity_for_evidence()
    if tenant_id_ev != tenant_id:
        raise PreModelPolicyConfigurationError("pre_model tenant_id identity mismatch")

    message_count = len(messages)
    model_id = getattr(adapter, "model", "") or ""
    if not isinstance(model_id, str):
        model_id = str(model_id)

    context = PreModelPolicyContext(model_id=model_id.strip())
    decision = evaluate_pre_model_policy(
        policy_engine,
        tenant_id=tenant_id,
        agent_id="",
        message_count=message_count,
        context=context,
    )

    digest = request_digest_for_payload(
        {
            "tenant_id": tenant_id,
            "message_count": message_count,
            "model_id": model_id,
            "inference_profile_id": (
                str(inference_profile_id) if inference_profile_id is not None else ""
            ),
            "policy_action": decision.action.value,
            "policy_rule_id": decision.policy_rule_id,
        }
    )
    idempotency_key = f"pre_model:{digest}:{decision.action.value}"
    _record_pre_model_evidence(
        governance_evidence_recorder,
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        principal_id=principal_id,
        decision=decision,
        request_digest=digest,
        idempotency_key=idempotency_key,
        model_scope=model_id.strip() or getattr(adapter, "provider", "") or "",
        inference_profile_id=inference_profile_id,
    )

    if decision.action in _PRE_MODEL_BLOCKED_ACTIONS:
        _fail_closed_decision(decision)
    if decision.action not in _PRE_MODEL_ALLOWED_ACTIONS:
        _fail_closed_decision(decision)


__all__ = [
    "PreModelPolicyConfigurationError",
    "enforce_pre_model_before_structured_inference",
]
