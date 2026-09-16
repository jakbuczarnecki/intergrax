# © Artur Czarnecki. All rights reserved.

"""MP-4R7 execution-time authorization gate — delegates to canonical platform validator."""

from __future__ import annotations

from intergrax.contracts.decision_authorization import (
    DecisionExecutionAction,
    DecisionExecutionAuthorization,
    DecisionGovernanceDisposition,
    DecisionGovernancePolicyContext,
)
from intergrax.contracts.decision_record import AuthoritativeAcceptedDecision
from intergrax.runtime.decision_authorization import validate_execution_authorization_bundle
from intergrax.runtime.decision_flow import DecisionFlowResult


def post_human_governance_disposition_from_flow(
    flow: DecisionFlowResult[object],
) -> DecisionGovernanceDisposition | None:
    """Derive post-human governance disposition from canonical decision-flow result."""
    if flow.authorization is not None:
        return DecisionGovernanceDisposition.ALLOW
    reason = flow.authority_reason
    if reason == "decision_governance_denied":
        return DecisionGovernanceDisposition.DENY
    if reason == "decision_governance_human_review_required_again":
        return DecisionGovernanceDisposition.REQUIRE_HUMAN
    return None


def validate_mp4r7_protected_execution_authorization(
    *,
    authorization: DecisionExecutionAuthorization,
    decision: AuthoritativeAcceptedDecision[object],
    action: DecisionExecutionAction,
    current_policy_context: DecisionGovernancePolicyContext,
) -> None:
    """Fail closed unless authorization matches decision, action, and current policy context."""
    validate_execution_authorization_bundle(
        authorization=authorization,
        decision=decision,
        action=action,
        current_policy_context=current_policy_context,
    )


__all__ = [
    "post_human_governance_disposition_from_flow",
    "validate_mp4r7_protected_execution_authorization",
]
