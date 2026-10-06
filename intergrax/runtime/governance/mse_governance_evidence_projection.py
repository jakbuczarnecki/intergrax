# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Shared MEANINGFUL_SIDE_EFFECT Governance fact idempotency + proof ref projection."""

from __future__ import annotations

from intergrax.contracts.collaborative_work import CollaborativeWorkEnforcementRequest
from intergrax.contracts.evaluated_policy_decision import request_digest_for_payload
from intergrax.contracts.governed_execution_governance_evidence import (
    governance_evidence_id_from_idempotency,
    is_canonical_governance_evidence_policy_action,
)
from intergrax.contracts.governed_proof import (
    EVIDENCE_KIND_GOVERNANCE_DECISION_FACT,
    GovernanceEvidenceRef,
)
from intergrax.contracts.meaningful_side_effect import (
    MeaningfulSideEffectRequest,
)
from intergrax.contracts.runtime_policy import PolicyDecision


def mse_governance_idempotency_key(
    request: CollaborativeWorkEnforcementRequest,
    decision: PolicyDecision,
) -> str | None:
    """Deterministic idempotency key for MEANINGFUL_SIDE_EFFECT facts (matches recorder)."""
    if not is_canonical_governance_evidence_policy_action(decision.action):
        return None
    side_effect = request.meaningful_side_effect_request
    tenant_id = request.tenant_id
    workspace_id = request.workspace_id
    principal_id = request.acting_principal_id
    task_id = None
    run_id = None
    attempt_id = None
    execution_id = None
    if type(side_effect) is MeaningfulSideEffectRequest:
        if side_effect.tenant_id:
            tenant_id = side_effect.tenant_id
        principal_id = side_effect.principal_id or principal_id
        task_id = side_effect.task_id
        run_id = side_effect.run_id
        attempt_id = side_effect.attempt_id
        execution_id = side_effect.execution_id
    digest_payload: dict[str, object] = {
        "tenant_id": tenant_id,
        "workspace_id": workspace_id,
        "principal_id": principal_id,
        "operation_id": request.operation_id,
        "resource_scope": request.resource_scope,
        "policy_rule_id": decision.policy_rule_id,
    }
    if task_id is not None:
        digest_payload["task_id"] = str(task_id)
    if run_id is not None:
        digest_payload["run_id"] = str(run_id)
    if attempt_id is not None:
        digest_payload["attempt_id"] = str(attempt_id)
    if execution_id is not None:
        digest_payload["execution_id"] = str(execution_id)
    digest = request_digest_for_payload(digest_payload)
    return f"mse:{digest}:{decision.action.value}"


def _scoped_mse_governance_idempotency_key(
    request: CollaborativeWorkEnforcementRequest,
    base_key: str,
) -> str:
    side_effect = request.meaningful_side_effect_request
    if type(side_effect) is MeaningfulSideEffectRequest:
        if (
            side_effect.task_id is not None
            and side_effect.run_id is not None
            and side_effect.attempt_id is not None
            and side_effect.execution_id is not None
        ):
            return f"{base_key}:run:{side_effect.run_id}:exec:{side_effect.execution_id}"
    return base_key


def mse_governance_evidence_ref(
    request: CollaborativeWorkEnforcementRequest,
    decision: PolicyDecision,
) -> GovernanceEvidenceRef | None:
    """Pointer to the GovernanceDecisionEvidenceFact for this MSE authorization."""
    idempotency_key = mse_governance_idempotency_key(request, decision)
    if idempotency_key is None:
        return None
    scoped_key = _scoped_mse_governance_idempotency_key(request, idempotency_key)
    evidence_id = governance_evidence_id_from_idempotency(scoped_key)
    policy_decision_ref = decision.decision_id.strip() or None
    return GovernanceEvidenceRef(
        kind=EVIDENCE_KIND_GOVERNANCE_DECISION_FACT,
        evidence_id=evidence_id,
        policy_decision_ref=policy_decision_ref,
    )


__all__ = [
    "mse_governance_evidence_ref",
    "mse_governance_idempotency_key",
]
