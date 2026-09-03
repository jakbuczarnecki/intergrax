# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Decision human review handoff contracts (DS-GOV-02).

Canonical Decision-owned semantics for requesting and consuming human judgment
over an exact Decision Proposal version. Distinct from verification, revision,
governance authorization, and declarative tool-side HITL grants.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import NewType, Protocol, runtime_checkable
from uuid import uuid4

from intergrax.contracts.decision_record import DecisionProposalRef
from intergrax.contracts.decision_revision import proposal_refs_match
from intergrax.contracts.human_approver import HumanApproverEvidence

DecisionHumanReviewRequestId = NewType("DecisionHumanReviewRequestId", str)
DecisionHumanReviewReasonCode = NewType("DecisionHumanReviewReasonCode", str)

_REQUEST_ID_PREFIX = "dhrr_"
_REASON_CODE_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def validate_decision_human_review_request_id(
    value: object,
) -> DecisionHumanReviewRequestId:
    if type(value) is not str:
        raise TypeError(
            f"DecisionHumanReviewRequestId must be str, got {type(value).__name__}",
        )
    if not value or not value.strip():
        raise ValueError(
            "DecisionHumanReviewRequestId must be non-empty and not whitespace-only",
        )
    if value != value.strip():
        raise ValueError(
            "DecisionHumanReviewRequestId must not contain leading or trailing whitespace",
        )
    if not value.startswith(_REQUEST_ID_PREFIX):
        raise ValueError(
            f"DecisionHumanReviewRequestId must start with {_REQUEST_ID_PREFIX!r}",
        )
    return DecisionHumanReviewRequestId(value)


def mint_decision_human_review_request_id() -> DecisionHumanReviewRequestId:
    return DecisionHumanReviewRequestId(f"{_REQUEST_ID_PREFIX}{uuid4().hex}")


def validate_decision_human_review_reason_code(
    value: object,
) -> DecisionHumanReviewReasonCode:
    if type(value) is not str:
        raise TypeError(
            f"DecisionHumanReviewReasonCode must be str, got {type(value).__name__}",
        )
    if not value or not value.strip():
        raise ValueError(
            "DecisionHumanReviewReasonCode must be non-empty and not whitespace-only",
        )
    if value != value.strip():
        raise ValueError(
            "DecisionHumanReviewReasonCode must not contain leading or trailing whitespace",
        )
    if not _REASON_CODE_PATTERN.fullmatch(value):
        raise ValueError(
            "DecisionHumanReviewReasonCode must match [a-z][a-z0-9_]{0,63}",
        )
    return DecisionHumanReviewReasonCode(value)


def verification_challenged_human_review_reason() -> DecisionHumanReviewReasonCode:
    return validate_decision_human_review_reason_code("verification_challenged")


def revision_exhausted_human_review_reason() -> DecisionHumanReviewReasonCode:
    return validate_decision_human_review_reason_code("revision_exhausted")


def policy_requires_human_review_reason() -> DecisionHumanReviewReasonCode:
    return validate_decision_human_review_reason_code("policy_requires_human")


def adjudication_required_human_review_reason() -> DecisionHumanReviewReasonCode:
    return validate_decision_human_review_reason_code("adjudication_required")


class DecisionHumanReviewOutcome(str, Enum):
    """Canonical human judgment outcome for one exact Decision proposal."""

    APPROVED = "approved"
    REJECTED = "rejected"
    ESCALATED = "escalated"


class DecisionHumanReviewMismatchError(ValueError):
    """Raised when a human review artifact does not match its bound proposal."""


@dataclass(frozen=True, slots=True)
class DecisionHumanReviewRequest:
    """Immutable request for human judgment over one exact proposal version."""

    request_id: DecisionHumanReviewRequestId
    proposal_ref: DecisionProposalRef
    reason_code: DecisionHumanReviewReasonCode
    presentation_context_ref: str | None = None

    def __post_init__(self) -> None:
        validate_decision_human_review_request_id(self.request_id)
        if type(self.proposal_ref) is not DecisionProposalRef:
            raise TypeError(
                "DecisionHumanReviewRequest.proposal_ref must be DecisionProposalRef",
            )
        validate_decision_human_review_reason_code(self.reason_code)
        if self.presentation_context_ref is not None:
            if type(self.presentation_context_ref) is not str:
                raise TypeError(
                    "DecisionHumanReviewRequest.presentation_context_ref must be str",
                )
            if not self.presentation_context_ref.strip():
                raise ValueError(
                    "DecisionHumanReviewRequest.presentation_context_ref "
                    "must be non-empty when set",
                )


def decision_human_review_request(
    *,
    proposal_ref: DecisionProposalRef,
    reason_code: DecisionHumanReviewReasonCode | str,
    request_id: DecisionHumanReviewRequestId | None = None,
    presentation_context_ref: str | None = None,
) -> DecisionHumanReviewRequest:
    """Build one human review request bound to an exact proposal reference."""
    if type(proposal_ref) is not DecisionProposalRef:
        raise TypeError("proposal_ref must be DecisionProposalRef")
    resolved_reason = (
        reason_code
        if type(reason_code) is DecisionHumanReviewReasonCode
        else validate_decision_human_review_reason_code(reason_code)
    )
    resolved_request_id = (
        request_id
        if request_id is not None
        else mint_decision_human_review_request_id()
    )
    return DecisionHumanReviewRequest(
        request_id=resolved_request_id,
        proposal_ref=proposal_ref,
        reason_code=resolved_reason,
        presentation_context_ref=presentation_context_ref,
    )


@dataclass(frozen=True, slots=True)
class DecisionHumanReviewPending:
    """Semantic state indicating human judgment is pending for one request."""

    request: DecisionHumanReviewRequest

    def __post_init__(self) -> None:
        if type(self.request) is not DecisionHumanReviewRequest:
            raise TypeError(
                "DecisionHumanReviewPending.request must be DecisionHumanReviewRequest",
            )


@dataclass(frozen=True, slots=True)
class DecisionHumanReviewProvenance:
    """Canonical provenance link to persisted platform human decision storage."""

    human_record_id: str
    human_request_id: str

    def __post_init__(self) -> None:
        if type(self.human_record_id) is not str or not self.human_record_id.strip():
            raise ValueError("DecisionHumanReviewProvenance.human_record_id must be non-empty")
        if type(self.human_request_id) is not str or not self.human_request_id.strip():
            raise ValueError("DecisionHumanReviewProvenance.human_request_id must be non-empty")


@dataclass(frozen=True, slots=True)
class DecisionHumanReviewDecision:
    """Immutable human judgment artifact bound to one exact proposal version."""

    request_id: DecisionHumanReviewRequestId
    proposal_ref: DecisionProposalRef
    outcome: DecisionHumanReviewOutcome
    approver: HumanApproverEvidence
    provenance: DecisionHumanReviewProvenance

    def __post_init__(self) -> None:
        validate_decision_human_review_request_id(self.request_id)
        if type(self.proposal_ref) is not DecisionProposalRef:
            raise TypeError(
                "DecisionHumanReviewDecision.proposal_ref must be DecisionProposalRef",
            )
        if type(self.outcome) is not DecisionHumanReviewOutcome:
            raise TypeError(
                "DecisionHumanReviewDecision.outcome must be DecisionHumanReviewOutcome",
            )
        if type(self.approver) is not HumanApproverEvidence:
            raise TypeError(
                "DecisionHumanReviewDecision.approver must be HumanApproverEvidence",
            )
        if type(self.provenance) is not DecisionHumanReviewProvenance:
            raise TypeError(
                "DecisionHumanReviewDecision.provenance must be DecisionHumanReviewProvenance",
            )


def decision_human_review_decision(
    *,
    request: DecisionHumanReviewRequest,
    outcome: DecisionHumanReviewOutcome,
    approver: HumanApproverEvidence,
    provenance: DecisionHumanReviewProvenance,
) -> DecisionHumanReviewDecision:
    """Mint one human review decision from a prior request and approver evidence."""
    if type(request) is not DecisionHumanReviewRequest:
        raise TypeError("request must be DecisionHumanReviewRequest")
    if type(outcome) is not DecisionHumanReviewOutcome:
        raise TypeError("outcome must be DecisionHumanReviewOutcome")
    if type(approver) is not HumanApproverEvidence:
        raise TypeError("approver must be HumanApproverEvidence")
    if type(provenance) is not DecisionHumanReviewProvenance:
        raise TypeError("provenance must be DecisionHumanReviewProvenance")
    return DecisionHumanReviewDecision(
        request_id=request.request_id,
        proposal_ref=request.proposal_ref,
        outcome=outcome,
        approver=approver,
        provenance=provenance,
    )


def validate_human_review_decision_against_request(
    *,
    request: DecisionHumanReviewRequest,
    decision: DecisionHumanReviewDecision,
) -> None:
    """Reject human review decisions that do not match the originating request."""
    if type(request) is not DecisionHumanReviewRequest:
        raise TypeError("request must be DecisionHumanReviewRequest")
    if type(decision) is not DecisionHumanReviewDecision:
        raise TypeError("decision must be DecisionHumanReviewDecision")
    if decision.request_id != request.request_id:
        raise DecisionHumanReviewMismatchError(
            "human review decision request_id must match request.request_id",
        )
    if not proposal_refs_match(decision.proposal_ref, request.proposal_ref):
        raise DecisionHumanReviewMismatchError(
            "human review decision proposal_ref must match request.proposal_ref",
        )


def validate_human_review_decision_for_proposal(
    *,
    decision: DecisionHumanReviewDecision,
    proposal_ref: DecisionProposalRef,
) -> None:
    """Reject stale human approvals for a different proposal version or lineage."""
    if type(decision) is not DecisionHumanReviewDecision:
        raise TypeError("decision must be DecisionHumanReviewDecision")
    if type(proposal_ref) is not DecisionProposalRef:
        raise TypeError("proposal_ref must be DecisionProposalRef")
    if not proposal_refs_match(decision.proposal_ref, proposal_ref):
        raise DecisionHumanReviewMismatchError(
            "human review decision proposal_ref must match target proposal_ref",
        )


@runtime_checkable
class DecisionHumanReviewPort(Protocol):
    """Host boundary for requesting human judgment without owning transport."""

    def request_review(
        self,
        request: DecisionHumanReviewRequest,
    ) -> DecisionHumanReviewPending:
        """Request human judgment for one exact proposal version."""
        ...
