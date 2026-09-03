# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass

import pytest

from intergrax.contracts.decision_identity import (
    DecisionExecutionLineage,
    DecisionIdentity,
    DecisionId,
    DecisionScope,
    DecisionVersion,
    initial_decision_version,
    mint_decision_id,
    next_decision_version,
)
from intergrax.contracts.decision_human_review import (
    DecisionHumanReviewDecision,
    DecisionHumanReviewMismatchError,
    DecisionHumanReviewOutcome,
    DecisionHumanReviewProvenance,
    DecisionHumanReviewRequest,
    adjudication_required_human_review_reason,
    decision_human_review_decision,
    decision_human_review_request,
    mint_decision_human_review_request_id,
    revision_exhausted_human_review_reason,
    validate_decision_human_review_reason_code,
    validate_decision_human_review_request_id,
    validate_human_review_decision_against_request,
    validate_human_review_decision_for_proposal,
    verification_challenged_human_review_reason,
)
from intergrax.contracts.decision_record import (
    CandidateDecision,
    DecisionArtifact,
    DecisionProposalRef,
    candidate_decision_ref,
    decision_lineage_ref,
    decision_proposal_ref,
    decision_version_lineage,
    validate_decision_artifact_kind,
    validate_decision_branch_id,
)
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.human_approver import (
    HumanApproverAuthMode,
    local_development_approver_evidence,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@dataclass(frozen=True, slots=True)
class Payload:
    text: str


def _execution_lineage() -> DecisionExecutionLineage:
    return DecisionExecutionLineage(
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )


def _identity(
    *,
    version: DecisionVersion | None = None,
    decision_id: DecisionId | None = None,
    tenant_id: str = "tenant-a",
) -> DecisionIdentity:
    return DecisionIdentity(
        decision_id=mint_decision_id() if decision_id is None else decision_id,
        version=version or initial_decision_version(),
        scope=DecisionScope(namespace="demo", subject="case-1"),
        tenant_id=tenant_id,
        execution=_execution_lineage(),
    )


def _candidate(
    *,
    identity: DecisionIdentity | None = None,
    branch_id: str = "main",
) -> CandidateDecision[Payload]:
    resolved_identity = identity or _identity()
    lineage = decision_version_lineage(
        current=decision_lineage_ref(
            resolved_identity.version,
            validate_decision_branch_id(branch_id),
        ),
    )
    return CandidateDecision(
        identity=resolved_identity,
        artifact=DecisionArtifact(
            kind=validate_decision_artifact_kind("demo.payload"),
            content=Payload(text="draft"),
        ),
        lineage=lineage,
    )


def _request(proposal_ref: DecisionProposalRef) -> DecisionHumanReviewRequest:
    return decision_human_review_request(
        proposal_ref=proposal_ref,
        reason_code=verification_challenged_human_review_reason(),
    )


def _decision(
    request: DecisionHumanReviewRequest,
    *,
    outcome: DecisionHumanReviewOutcome = DecisionHumanReviewOutcome.APPROVED,
) -> DecisionHumanReviewDecision:
    return decision_human_review_decision(
        request=request,
        outcome=outcome,
        approver=local_development_approver_evidence(tenant_id=request.proposal_ref.identity.tenant_id),
        provenance=DecisionHumanReviewProvenance(
            human_record_id="hdec_test123",
            human_request_id=str(request.request_id),
        ),
    )


def test_request_binds_exact_proposal_ref() -> None:
    candidate = _candidate()
    proposal_ref = candidate_decision_ref(candidate)
    request = _request(proposal_ref)
    assert request.proposal_ref == proposal_ref
    assert request.reason_code == verification_challenged_human_review_reason()


def test_reason_codes_validate_and_include_known_values() -> None:
    assert revision_exhausted_human_review_reason().startswith("revision_")
    assert adjudication_required_human_review_reason().startswith("adjudication_")
    with pytest.raises(ValueError, match="must match"):
        validate_decision_human_review_reason_code("INVALID")


def test_request_id_mint_and_validate() -> None:
    request_id = mint_decision_human_review_request_id()
    assert validate_decision_human_review_request_id(request_id) == request_id
    with pytest.raises(ValueError, match="must start with"):
        validate_decision_human_review_request_id("bad-id")


def test_valid_approval_matches_request() -> None:
    candidate = _candidate()
    request = _request(candidate_decision_ref(candidate))
    decision = _decision(request)
    validate_human_review_decision_against_request(request=request, decision=decision)


def test_wrong_request_id_rejected() -> None:
    candidate = _candidate()
    request = _request(candidate_decision_ref(candidate))
    decision = _decision(request)
    other_request = decision_human_review_request(
        proposal_ref=request.proposal_ref,
        reason_code=request.reason_code,
        request_id=mint_decision_human_review_request_id(),
    )
    with pytest.raises(DecisionHumanReviewMismatchError, match="request_id"):
        validate_human_review_decision_against_request(
            request=other_request,
            decision=decision,
        )


def test_wrong_version_rejected() -> None:
    candidate_v1 = _candidate()
    request = _request(candidate_decision_ref(candidate_v1))
    decision = _decision(request)
    candidate_v2 = CandidateDecision(
        identity=DecisionIdentity(
            decision_id=candidate_v1.identity.decision_id,
            version=next_decision_version(candidate_v1.identity.version),
            scope=candidate_v1.identity.scope,
            tenant_id=candidate_v1.identity.tenant_id,
            execution=candidate_v1.identity.execution,
        ),
        artifact=candidate_v1.artifact,
        lineage=decision_version_lineage(
            current=decision_lineage_ref(next_decision_version(candidate_v1.identity.version)),
            parents=(candidate_decision_ref(candidate_v1).lineage_ref,),
        ),
    )
    with pytest.raises(DecisionHumanReviewMismatchError, match="proposal_ref"):
        validate_human_review_decision_for_proposal(
            decision=decision,
            proposal_ref=candidate_decision_ref(candidate_v2),
        )


def test_stale_approval_after_revision_invalid_for_v2() -> None:
    candidate_v1 = _candidate()
    request_v1 = _request(candidate_decision_ref(candidate_v1))
    approval_v1 = _decision(request_v1)
    validate_human_review_decision_for_proposal(
        decision=approval_v1,
        proposal_ref=candidate_decision_ref(candidate_v1),
    )
    candidate_v2 = CandidateDecision(
        identity=DecisionIdentity(
            decision_id=candidate_v1.identity.decision_id,
            version=next_decision_version(candidate_v1.identity.version),
            scope=candidate_v1.identity.scope,
            tenant_id=candidate_v1.identity.tenant_id,
            execution=candidate_v1.identity.execution,
        ),
        artifact=candidate_v1.artifact,
        lineage=decision_version_lineage(
            current=decision_lineage_ref(next_decision_version(candidate_v1.identity.version)),
            parents=(candidate_decision_ref(candidate_v1).lineage_ref,),
        ),
    )
    with pytest.raises(DecisionHumanReviewMismatchError):
        validate_human_review_decision_for_proposal(
            decision=approval_v1,
            proposal_ref=candidate_decision_ref(candidate_v2),
        )


def test_wrong_branch_rejected() -> None:
    identity = _identity(version=DecisionVersion(2))
    ref_a = decision_proposal_ref(
        identity=identity,
        lineage_ref=decision_lineage_ref(identity.version, validate_decision_branch_id("A")),
    )
    ref_b = decision_proposal_ref(
        identity=identity,
        lineage_ref=decision_lineage_ref(identity.version, validate_decision_branch_id("B")),
    )
    request = _request(ref_a)
    decision = _decision(request)
    with pytest.raises(DecisionHumanReviewMismatchError):
        validate_human_review_decision_for_proposal(decision=decision, proposal_ref=ref_b)


def test_wrong_execution_lineage_rejected() -> None:
    candidate_a = _candidate()
    request = _request(candidate_decision_ref(candidate_a))
    decision = _decision(request)
    candidate_b = _candidate(
        identity=DecisionIdentity(
            decision_id=candidate_a.identity.decision_id,
            version=candidate_a.identity.version,
            scope=candidate_a.identity.scope,
            tenant_id=candidate_a.identity.tenant_id,
            execution=DecisionExecutionLineage(
                task_id=mint_task_id(),
                run_id=mint_run_id(),
                attempt_id=mint_attempt_id(),
                execution_id=mint_execution_id(),
            ),
        ),
    )
    with pytest.raises(DecisionHumanReviewMismatchError):
        validate_human_review_decision_for_proposal(
            decision=decision,
            proposal_ref=candidate_decision_ref(candidate_b),
        )


def test_human_outcomes_are_decision_semantics_not_platform_verdicts() -> None:
    assert DecisionHumanReviewOutcome.APPROVED.value == "approved"
    assert DecisionHumanReviewOutcome.REJECTED.value == "rejected"
    assert DecisionHumanReviewOutcome.ESCALATED.value == "escalated"
    assert "unknown" not in {member.value for member in DecisionHumanReviewOutcome}


def test_approver_evidence_uses_platform_contract() -> None:
    candidate = _candidate()
    request = _request(candidate_decision_ref(candidate))
    decision = _decision(request)
    assert decision.approver.auth_mode is HumanApproverAuthMode.LOCAL_DEVELOPMENT
