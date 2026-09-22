# © Artur Czarnecki. All rights reserved.

"""Adapt declarative HITL grants to neutral tool invocation governance evidence."""

from __future__ import annotations

from intergrax.contracts.declarative_hitl import DeclarativeHitlApprovalGrant
from intergrax.contracts.tool_invocation_governance_approval_evidence import (
    ToolInvocationGovernanceApprovalEvidence,
)


def tool_invocation_governance_approval_evidence_from_declarative_hitl(
    grant: DeclarativeHitlApprovalGrant,
    *,
    tenant_id: str,
) -> ToolInvocationGovernanceApprovalEvidence:
    """Adapt declarative HITL grant artifacts to the neutral invocation evidence shape."""
    return ToolInvocationGovernanceApprovalEvidence(
        evidence_ref=grant.grant_id,
        invocation_scope_id=grant.invocation_scope_id,
        tenant_id=tenant_id,
        task_id=grant.task_id,
        run_id=grant.run_id,
        step_id=grant.step_id,
        tool_id=grant.tool_id,
        agent_id=grant.agent_id,
        idempotency_key=grant.idempotency_key,
        matched_rule_ids=grant.matched_rule_ids,
        human_request_id=grant.human_request_id,
        policy_provenance_digest=grant.policy_provenance_digest,
        pause_id=grant.pause_id,
        approved_at=grant.approved_at,
    )


__all__ = [
    "tool_invocation_governance_approval_evidence_from_declarative_hitl",
]
