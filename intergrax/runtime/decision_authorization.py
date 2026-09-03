# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Decision governance authorization runtime helpers (DS-GOV-01).

Validate governance plugin output and bind execution authorization to exact
authoritative decision versions without introducing a second policy engine.
"""

from __future__ import annotations

from typing import TypeVar

from intergrax.contracts.decision_authorization import (
    DecisionExecutionAction,
    DecisionExecutionAuthorization,
    DecisionGovernanceDecision,
    DecisionGovernanceDisposition,
    DecisionGovernanceEvaluationInput,
    decision_execution_authorization,
    validate_execution_authorization_for_action,
    validate_execution_authorization_for_decision,
    validate_execution_authorization_for_policy_context,
    validate_governance_decision_against_input,
)
from intergrax.contracts.decision_record import AuthoritativeAcceptedDecision
from intergrax.contracts.decision_verification import (
    VerificationDisposition,
    VerificationResult,
)

T = TypeVar("T")


def validate_execution_authorization_bundle(
    *,
    authorization: DecisionExecutionAuthorization,
    decision: AuthoritativeAcceptedDecision[T],
    action: DecisionExecutionAction,
) -> None:
    """Reject stale or mismatched execution authorization before side effects."""
    validate_execution_authorization_for_decision(
        authorization=authorization,
        decision=decision,
    )
    validate_execution_authorization_for_action(
        authorization=authorization,
        action=action,
    )
    validate_execution_authorization_for_policy_context(
        authorization=authorization,
        policy_context=authorization.policy_context,
    )


def mint_validated_execution_authorization(
    *,
    evaluation_input: DecisionGovernanceEvaluationInput[T],
    governance_decision: DecisionGovernanceDecision,
) -> DecisionExecutionAuthorization:
    """Validate governance output, then mint execution authorization on ALLOW."""
    validate_governance_decision_against_input(
        evaluation_input=evaluation_input,
        governance_decision=governance_decision,
    )
    if governance_decision.disposition is not DecisionGovernanceDisposition.ALLOW:
        raise ValueError(
            "execution authorization requires DecisionGovernanceDisposition.ALLOW",
        )
    authorization = decision_execution_authorization(
        governance_decision=governance_decision,
    )
    validate_execution_authorization_bundle(
        authorization=authorization,
        decision=evaluation_input.decision,
        action=evaluation_input.action,
    )
    return authorization


def verification_pass_does_not_imply_governance_allow(
    *,
    verification_result: VerificationResult,
    governance_decision: DecisionGovernanceDecision,
) -> bool:
    """Return whether verification passed while governance denied authorization."""
    if type(verification_result) is not VerificationResult:
        raise TypeError("verification_result must be VerificationResult")
    if type(governance_decision) is not DecisionGovernanceDecision:
        raise TypeError("governance_decision must be DecisionGovernanceDecision")
    return (
        verification_result.disposition is VerificationDisposition.PASSED
        and governance_decision.disposition is DecisionGovernanceDisposition.DENY
    )


def human_approval_does_not_imply_governance_allow(
    *,
    human_outcome_approved: bool,
    governance_decision: DecisionGovernanceDecision,
) -> bool:
    """Return whether human approval coexists with governance denial."""
    if type(governance_decision) is not DecisionGovernanceDecision:
        raise TypeError("governance_decision must be DecisionGovernanceDecision")
    return (
        human_outcome_approved
        and governance_decision.disposition is DecisionGovernanceDisposition.DENY
    )
