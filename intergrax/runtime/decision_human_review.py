# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Decision human review lifecycle helpers (DS-GOV-02).

Validate plugin/adapter human review output and transition lifecycle stages
without owning transport, UI, or synchronous human polling.
"""

from __future__ import annotations

from intergrax.contracts.decision_human_review import (
    DecisionHumanReviewDecision,
    DecisionHumanReviewOutcome,
    DecisionHumanReviewPending,
    DecisionHumanReviewProvenance,
    DecisionHumanReviewRequest,
    decision_human_review_decision,
    validate_human_review_decision_against_request,
    validate_human_review_decision_for_proposal,
)
from intergrax.contracts.decision_lifecycle import (
    DecisionLifecycleStage,
    DecisionLifecycleState,
    transition_decision_lifecycle,
)
from intergrax.contracts.decision_record import DecisionProposalRef
from intergrax.contracts.human_approver import HumanApproverEvidence
from intergrax.runtime.human.models import HumanDecisionRecord, HumanResponseVerdict


class DecisionHumanReviewAdapterError(ValueError):
    """Raised when platform human transport cannot be adapted safely."""


_VERDICT_TO_OUTCOME: dict[HumanResponseVerdict, DecisionHumanReviewOutcome] = {
    HumanResponseVerdict.APPROVE: DecisionHumanReviewOutcome.APPROVED,
    HumanResponseVerdict.REJECT: DecisionHumanReviewOutcome.REJECTED,
    HumanResponseVerdict.ESCALATE: DecisionHumanReviewOutcome.ESCALATED,
}


def decision_human_review_outcome_from_human_verdict(
    verdict: HumanResponseVerdict,
) -> DecisionHumanReviewOutcome:
    """Map platform human response verdict to canonical Decision review outcome."""
    if type(verdict) is not HumanResponseVerdict:
        raise TypeError("verdict must be HumanResponseVerdict")
    outcome = _VERDICT_TO_OUTCOME.get(verdict)
    if outcome is None:
        raise DecisionHumanReviewAdapterError(
            "HumanResponseVerdict.UNKNOWN is not an authoritative Decision outcome",
        )
    return outcome


def request_decision_human_review(
    request: DecisionHumanReviewRequest,
) -> DecisionHumanReviewPending:
    """Return canonical pending semantic state for one human review request."""
    if type(request) is not DecisionHumanReviewRequest:
        raise TypeError("request must be DecisionHumanReviewRequest")
    return DecisionHumanReviewPending(request=request)


def transition_lifecycle_for_human_review_request(
    *,
    lifecycle_state: DecisionLifecycleState,
) -> DecisionLifecycleState:
    """Move lifecycle into adjudication when human judgment is required."""
    if type(lifecycle_state) is not DecisionLifecycleState:
        raise TypeError("lifecycle_state must be DecisionLifecycleState")
    if lifecycle_state.stage not in (
        DecisionLifecycleStage.VERIFICATION,
        DecisionLifecycleStage.REVISION,
        DecisionLifecycleStage.ADJUDICATION,
    ):
        raise ValueError(
            "human review lifecycle transition requires current stage "
            "verification, revision, or adjudication",
        )
    if lifecycle_state.stage is DecisionLifecycleStage.ADJUDICATION:
        return lifecycle_state
    return transition_decision_lifecycle(
        lifecycle_state,
        DecisionLifecycleStage.ADJUDICATION,
    )


def validate_consumed_human_review_decision(
    *,
    request: DecisionHumanReviewRequest,
    decision: DecisionHumanReviewDecision,
    target_proposal_ref: DecisionProposalRef,
) -> None:
    """Validate request binding, then reject stale approval for another proposal."""
    validate_human_review_decision_against_request(request=request, decision=decision)
    validate_human_review_decision_for_proposal(
        decision=decision,
        proposal_ref=target_proposal_ref,
    )


def decision_human_review_decision_from_human_record(
    *,
    request: DecisionHumanReviewRequest,
    record: HumanDecisionRecord,
) -> DecisionHumanReviewDecision:
    """Adapt one persisted platform human record into a Decision review decision."""
    if type(request) is not DecisionHumanReviewRequest:
        raise TypeError("request must be DecisionHumanReviewRequest")
    if type(record) is not HumanDecisionRecord:
        raise TypeError("record must be HumanDecisionRecord")
    if record.human_request_id and record.human_request_id != str(request.request_id):
        raise DecisionHumanReviewAdapterError(
            "human record human_request_id must match request.request_id when set",
        )
    if record.tenant_id != request.proposal_ref.identity.tenant_id:
        raise DecisionHumanReviewAdapterError(
            "human record tenant_id must match proposal_ref identity tenant_id",
        )
    outcome = decision_human_review_outcome_from_human_verdict(record.verdict)
    provenance = DecisionHumanReviewProvenance(
        human_record_id=record.decision_id,
        human_request_id=record.human_request_id or str(request.request_id),
    )
    return decision_human_review_decision(
        request=request,
        outcome=outcome,
        approver=record.approver,
        provenance=provenance,
    )


def human_decision_record_from_review_decision(
    *,
    decision: DecisionHumanReviewDecision,
    response_text: str = "",
    notes: str = "",
) -> HumanDecisionRecord:
    """Project one canonical Decision review decision into platform persistence shape."""
    if type(decision) is not DecisionHumanReviewDecision:
        raise TypeError("decision must be DecisionHumanReviewDecision")
    proposal = decision.proposal_ref
    execution = proposal.identity.execution
    verdict = _OUTCOME_TO_VERDICT[decision.outcome]
    return HumanDecisionRecord(
        task_id=str(execution.task_id),
        tenant_id=proposal.identity.tenant_id,
        approver=decision.approver,
        human_request_id=str(decision.request_id),
        verdict=verdict,
        response_text=response_text,
        run_id=str(execution.run_id),
        notes=notes,
        created_at_utc=_created_at_utc(),
    )


_OUTCOME_TO_VERDICT: dict[DecisionHumanReviewOutcome, HumanResponseVerdict] = {
    DecisionHumanReviewOutcome.APPROVED: HumanResponseVerdict.APPROVE,
    DecisionHumanReviewOutcome.REJECTED: HumanResponseVerdict.REJECT,
    DecisionHumanReviewOutcome.ESCALATED: HumanResponseVerdict.ESCALATE,
}


def _created_at_utc() -> str:
    from intergrax.utils.time_provider import SystemTimeProvider

    return SystemTimeProvider.utc_now().isoformat()
