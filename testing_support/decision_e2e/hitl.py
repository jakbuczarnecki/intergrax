# © Artur Czarnecki. All rights reserved.

"""Canonical HITL boundary helpers for DS-E2E-04."""

from __future__ import annotations

from dataclasses import dataclass, field

from intergrax.contracts.decision_human_review import (
    DecisionHumanReviewDecision,
    DecisionHumanReviewMismatchError,
    DecisionHumanReviewOutcome,
    DecisionHumanReviewPending,
    DecisionHumanReviewProvenance,
    DecisionHumanReviewRequest,
    decision_human_review_decision,
    validate_human_review_decision_for_proposal,
)
from intergrax.contracts.decision_record import DecisionProposalRef
from intergrax.contracts.human_approver import local_development_approver_evidence
from intergrax.runtime.decision_human_review import request_decision_human_review


@dataclass(slots=True)
class DurableDecisionHumanReviewPort:
    """Durable in-memory HITL port exercising canonical review APIs."""

    pending_requests: list[DecisionHumanReviewRequest] = field(default_factory=list)
    decisions: list[DecisionHumanReviewDecision] = field(default_factory=list)

    def request_review(self, request: DecisionHumanReviewRequest) -> DecisionHumanReviewPending:
        pending = request_decision_human_review(request)
        self.pending_requests.append(request)
        return pending

    def submit_decision(
        self,
        *,
        request: DecisionHumanReviewRequest,
        outcome: DecisionHumanReviewOutcome,
    ) -> DecisionHumanReviewDecision:
        decision = decision_human_review_decision(
            request=request,
            outcome=outcome,
            approver=local_development_approver_evidence(
                tenant_id=request.proposal_ref.identity.tenant_id,
            ),
            provenance=DecisionHumanReviewProvenance(
                human_record_id="hdec_decision_e2e_qualification",
                human_request_id=str(request.request_id),
            ),
        )
        validate_human_review_decision_for_proposal(
            decision=decision,
            proposal_ref=request.proposal_ref,
        )
        self.decisions.append(decision)
        return decision

    def submit_stale_decision(
        self,
        *,
        stale_request: DecisionHumanReviewRequest,
        current_proposal_ref: DecisionProposalRef,
        outcome: DecisionHumanReviewOutcome = DecisionHumanReviewOutcome.APPROVED,
    ) -> None:
        decision = decision_human_review_decision(
            request=stale_request,
            outcome=outcome,
            approver=local_development_approver_evidence(
                tenant_id=stale_request.proposal_ref.identity.tenant_id,
            ),
            provenance=DecisionHumanReviewProvenance(
                human_record_id="hdec_decision_e2e_stale",
                human_request_id=str(stale_request.request_id),
            ),
        )
        try:
            validate_human_review_decision_for_proposal(
                decision=decision,
                proposal_ref=current_proposal_ref,
            )
        except DecisionHumanReviewMismatchError:
            return
        raise AssertionError("stale human review must fail closed")
