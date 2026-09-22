# © Artur Czarnecki. All rights reserved.

"""Provider-neutral governance approval evidence for catalog tool invocation."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.declarative_hitl import DeclarativeHitlApprovalGrant


@dataclass(frozen=True, slots=True)
class ToolInvocationGovernanceApprovalEvidence:
    """Scoped human/policy approval for one execution-bound catalog tool invocation.

    Carries the minimum identity required for ``ToolAuthorizationRequest.approval_evidence_ref``
    and downstream scope correlation. Does not execute or validate policy.
    """

    evidence_ref: str
    invocation_scope_id: str
    task_id: str
    run_id: str
    step_id: str
    tool_id: str
    agent_id: str
    idempotency_key: str | None = None
    matched_rule_ids: tuple[str, ...] = ()
    human_request_id: str = ""
    policy_provenance_digest: str | None = None
    pause_id: str = ""
    approved_at: str = ""


def tool_invocation_governance_approval_evidence_from_declarative_hitl(
    grant: DeclarativeHitlApprovalGrant,
) -> ToolInvocationGovernanceApprovalEvidence:
    """Adapt declarative HITL grant artifacts to the neutral invocation evidence shape."""
    return ToolInvocationGovernanceApprovalEvidence(
        evidence_ref=grant.grant_id,
        invocation_scope_id=grant.invocation_scope_id,
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
    "ToolInvocationGovernanceApprovalEvidence",
    "tool_invocation_governance_approval_evidence_from_declarative_hitl",
]
