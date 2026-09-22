# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Adapt neutral catalog approval evidence to Nexus runtime governance carriers."""

from __future__ import annotations

from intergrax.contracts.declarative_hitl import DeclarativeHitlApprovalGrant
from intergrax.contracts.execution_bound_catalog_tool_invocation import (
    ExecutionBoundCatalogToolInvokeRequest,
)
from intergrax.contracts.tool_invocation_governance_approval_evidence import (
    ToolInvocationGovernanceApprovalEvidence,
)


def declarative_hitl_grant_from_invocation_evidence(
    evidence: ToolInvocationGovernanceApprovalEvidence,
) -> DeclarativeHitlApprovalGrant:
    """Bridge request-local evidence to existing RuntimeState governance carrier."""
    return DeclarativeHitlApprovalGrant(
        grant_id=evidence.evidence_ref,
        invocation_scope_id=evidence.invocation_scope_id,
        task_id=evidence.task_id,
        run_id=evidence.run_id,
        step_id=evidence.step_id,
        tool_id=evidence.tool_id,
        agent_id=evidence.agent_id,
        idempotency_key=evidence.idempotency_key,
        matched_rule_ids=evidence.matched_rule_ids,
        human_request_id=evidence.human_request_id,
        policy_provenance_digest=evidence.policy_provenance_digest,
        pause_id=evidence.pause_id,
        approved_at=evidence.approved_at,
    )


def require_invocation_evidence_matches_request(
    evidence: ToolInvocationGovernanceApprovalEvidence,
    request: ExecutionBoundCatalogToolInvokeRequest,
) -> None:
    """Fail closed when scoped evidence does not match the invocation request."""
    if evidence.tenant_id.strip() != request.tenant_id.strip():
        raise ValueError("governance approval evidence tenant_id mismatch")
    if evidence.task_id != request.task_id:
        raise ValueError("governance approval evidence task_id mismatch")
    if evidence.run_id != request.run_id:
        raise ValueError("governance approval evidence run_id mismatch")
    if evidence.step_id != request.step_id:
        raise ValueError("governance approval evidence step_id mismatch")
    if evidence.tool_id != request.tool_id:
        raise ValueError("governance approval evidence tool_id mismatch")
    if evidence.agent_id != request.agent_id:
        raise ValueError("governance approval evidence agent_id mismatch")


__all__ = [
    "declarative_hitl_grant_from_invocation_evidence",
    "require_invocation_evidence_matches_request",
]
