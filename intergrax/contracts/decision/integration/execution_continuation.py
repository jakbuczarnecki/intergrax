# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Neutral Decision human review → Execution continuation integration (MP-4R3/MP-4R7)."""

from __future__ import annotations

from intergrax.contracts.decision_human_review import (
    DecisionHumanReviewDecision,
    DecisionHumanReviewOutcome,
)
from intergrax.contracts.execution_continuation import (
    ExecutionContinuationError,
    ExecutionContinuationErrorCode,
    ExecutionContinuationResolutionCommand,
    ExecutionHumanVerdict,
    PendingExecutionContinuation,
    execution_continuation_resolution_command_for_pending_human_verdict,
)


def decision_human_review_outcome_to_execution_verdict(
    outcome: DecisionHumanReviewOutcome,
) -> ExecutionHumanVerdict:
    """Map canonical Decision human review outcome to execution continuation verdict."""
    if type(outcome) is not DecisionHumanReviewOutcome:
        raise TypeError("outcome must be DecisionHumanReviewOutcome")
    if outcome is DecisionHumanReviewOutcome.APPROVED:
        return ExecutionHumanVerdict.APPROVE
    if outcome is DecisionHumanReviewOutcome.REJECTED:
        return ExecutionHumanVerdict.REJECT
    if outcome is DecisionHumanReviewOutcome.ESCALATED:
        return ExecutionHumanVerdict.ESCALATE
    raise ValueError(f"unsupported DecisionHumanReviewOutcome: {outcome!s}")


def execution_continuation_resolution_command_from_decision_human_review_decision(
    pending: PendingExecutionContinuation,
    decision: DecisionHumanReviewDecision,
    *,
    resolved_at: str,
) -> ExecutionContinuationResolutionCommand:
    """Project one consumed Decision human review decision onto continuation resolution."""
    if type(decision) is not DecisionHumanReviewDecision:
        raise TypeError("decision must be DecisionHumanReviewDecision")
    provenance_request_id = decision.provenance.human_request_id.strip()
    if provenance_request_id != str(decision.request_id):
        raise ExecutionContinuationError(
            "decision human_request_id must match request_id",
            code=ExecutionContinuationErrorCode.IDENTITY_MISMATCH,
        )
    if decision.approver.tenant_id != decision.proposal_ref.identity.tenant_id:
        raise ExecutionContinuationError(
            "decision approver tenant_id mismatch",
            code=ExecutionContinuationErrorCode.IDENTITY_MISMATCH,
        )
    verdict = decision_human_review_outcome_to_execution_verdict(decision.outcome)
    return execution_continuation_resolution_command_for_pending_human_verdict(
        pending,
        verdict=verdict,
        approver=decision.approver,
        human_request_id=provenance_request_id,
        resolved_at=resolved_at,
    )


__all__ = [
    "decision_human_review_outcome_to_execution_verdict",
    "execution_continuation_resolution_command_from_decision_human_review_decision",
]
