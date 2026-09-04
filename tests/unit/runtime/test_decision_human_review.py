# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
    DecisionHumanReviewMismatchError,
    DecisionHumanReviewOutcome,
    DecisionHumanReviewRequest,
    decision_human_review_request,
    revision_exhausted_human_review_reason,
    verification_challenged_human_review_reason,
)
from intergrax.contracts.decision_lifecycle import (
    DecisionLifecycleStage,
    initial_decision_lifecycle_state,
    transition_decision_lifecycle,
)
from intergrax.contracts.decision_record import (
    AuthoritativeAcceptedDecision,
    CandidateDecision,
    DecisionArtifact,
    candidate_decision_ref,
    decision_lineage_ref,
    decision_version_lineage,
    validate_decision_artifact_kind,
    validate_decision_branch_id,
)
from intergrax.contracts.decision_revision import (
    DecisionRevisionDisposition,
    DecisionRevisionPolicy,
    decision_revision_authorization,
    decision_revision_policy,
    evaluate_decision_revision,
    initial_decision_revision_state,
)
from intergrax.contracts.decision_verification import (
    VerificationDisposition,
    VerificationResult,
    VerificationStageOutcome,
    validate_verification_finding_code,
    validate_verification_requirement_code,
    validate_verification_stage_kind,
    verification_challenge,
    verification_finding,
    verification_result,
    verification_stage_record,
)
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.human_approver import local_development_approver_evidence
from intergrax.runtime.decision_human_review import (
    DecisionHumanReviewAdapterError,
    decision_human_review_decision_from_human_record,
    decision_human_review_outcome_from_human_verdict,
    human_decision_record_from_review_decision,
    request_decision_human_review,
    transition_lifecycle_for_human_review_request,
    validate_consumed_human_review_decision,
)
from intergrax.runtime.decision_revision import mint_revised_candidate_decision
from intergrax.runtime.human.models import HumanResponseVerdict, build_human_decision_record

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_MODULE_PATHS = (
    Path("intergrax/contracts/decision_human_review.py"),
    Path("intergrax/runtime/decision_human_review.py"),
)
_FORBIDDEN_FRAGMENTS = (
    "runtime.nexus",
    "runtime.critic",
    "DeclarativeHitlApprovalGrant",
    "declarative_hitl",
    "L1Gateway",
    "CriticOrchestrator",
    "openai",
    "anthropic",
    "Any",
    "cast(",
    "type: ignore",
    "pyright: ignore",
    "getattr",
    "setattr",
    "hasattr",
    "inspect",
    "exec(",
    "eval(",
    "object.__setattr__",
    "dict[str, Any]",
)


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
) -> DecisionIdentity:
    return DecisionIdentity(
        decision_id=mint_decision_id() if decision_id is None else decision_id,
        version=version or initial_decision_version(),
        scope=DecisionScope(namespace="demo", subject="case-1"),
        tenant_id="tenant-a",
        execution=_execution_lineage(),
    )


def _candidate(
    *,
    identity: DecisionIdentity | None = None,
    branch_id: str = "main",
    text: str = "draft",
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
            content=Payload(text=text),
        ),
        lineage=lineage,
    )


def _challenged_result(proposal_ref) -> VerificationResult:
    finding = verification_finding(
        code=validate_verification_finding_code("verification.semantic.below_requirement"),
        message="below requirement",
    )
    stage = validate_verification_stage_kind("semantic")
    return verification_result(
        proposal_ref=proposal_ref,
        disposition=VerificationDisposition.CHALLENGED,
        stage_records=(
            verification_stage_record(
                proposal_ref=proposal_ref,
                stage=stage,
                outcome=VerificationStageOutcome.CHALLENGED,
                challenge=verification_challenge(
                    proposal_ref=proposal_ref,
                    stage=stage,
                    requirement_code=validate_verification_requirement_code(
                        "verification.semantic.below_requirement",
                    ),
                    finding=finding,
                ),
            ),
        ),
    )


def _review_request(candidate: CandidateDecision[Payload]) -> DecisionHumanReviewRequest:
    return decision_human_review_request(
        proposal_ref=candidate_decision_ref(candidate),
        reason_code=revision_exhausted_human_review_reason(),
    )


def test_forbidden_patterns_absent_in_human_review_modules() -> None:
    for module_path in _MODULE_PATHS:
        source = module_path.read_text(encoding="utf-8")
        for fragment in _FORBIDDEN_FRAGMENTS:
            assert fragment not in source


def test_request_review_returns_pending_state() -> None:
    candidate = _candidate()
    request = _review_request(candidate)
    pending = request_decision_human_review(request)
    assert pending.request == request


def test_lifecycle_moves_to_adjudication_for_human_review() -> None:
    candidate = _candidate()
    lifecycle = transition_decision_lifecycle(
        initial_decision_lifecycle_state(candidate.identity),
        DecisionLifecycleStage.VERIFICATION,
    )
    next_state = transition_lifecycle_for_human_review_request(lifecycle_state=lifecycle)
    assert next_state.stage is DecisionLifecycleStage.ADJUDICATION


def test_human_verdict_mapping_rejects_unknown() -> None:
    with pytest.raises(DecisionHumanReviewAdapterError, match="UNKNOWN"):
        decision_human_review_outcome_from_human_verdict(HumanResponseVerdict.UNKNOWN)


def test_adapter_from_human_record_preserves_proposal_ref() -> None:
    candidate = _candidate()
    request = _review_request(candidate)
    record = build_human_decision_record(
        task_id=str(candidate.identity.execution.task_id),
        tenant_id=candidate.identity.tenant_id,
        approver=local_development_approver_evidence(tenant_id=candidate.identity.tenant_id),
        verdict=HumanResponseVerdict.APPROVE,
        response_text="approved",
        human_request_id=str(request.request_id),
    )
    decision = decision_human_review_decision_from_human_record(
        request=request,
        record=record,
    )
    validate_consumed_human_review_decision(
        request=request,
        decision=decision,
        target_proposal_ref=candidate_decision_ref(candidate),
    )
    assert decision.outcome is DecisionHumanReviewOutcome.APPROVED


def test_human_record_projection_round_trip_request_id() -> None:
    candidate = _candidate()
    request = _review_request(candidate)
    record = build_human_decision_record(
        task_id=str(candidate.identity.execution.task_id),
        tenant_id=candidate.identity.tenant_id,
        approver=local_development_approver_evidence(tenant_id=candidate.identity.tenant_id),
        verdict=HumanResponseVerdict.REJECT,
        response_text="rejected",
        human_request_id=str(request.request_id),
    )
    decision = decision_human_review_decision_from_human_record(
        request=request,
        record=record,
    )
    projected = human_decision_record_from_review_decision(decision=decision)
    assert projected.human_request_id == str(request.request_id)
    assert projected.verdict is HumanResponseVerdict.REJECT


def test_full_safe_flow_up_to_human_approval_without_execution_authorization() -> None:
    challenged = _candidate()
    proposal_ref = candidate_decision_ref(challenged)
    policy = decision_revision_policy(max_revisions=0)
    revision_decision = evaluate_decision_revision(
        policy=policy,
        state=initial_decision_revision_state(proposal_ref),
        verification_result=_challenged_result(proposal_ref),
    )
    assert revision_decision.disposition is DecisionRevisionDisposition.EXHAUSTED
    request = decision_human_review_request(
        proposal_ref=proposal_ref,
        reason_code=verification_challenged_human_review_reason(),
    )
    pending = request_decision_human_review(request)
    lifecycle = transition_lifecycle_for_human_review_request(
        lifecycle_state=transition_decision_lifecycle(
            initial_decision_lifecycle_state(challenged.identity),
            DecisionLifecycleStage.VERIFICATION,
        ),
    )
    record = build_human_decision_record(
        task_id=str(challenged.identity.execution.task_id),
        tenant_id=challenged.identity.tenant_id,
        approver=local_development_approver_evidence(tenant_id=challenged.identity.tenant_id),
        verdict=HumanResponseVerdict.APPROVE,
        response_text="approved",
        human_request_id=str(request.request_id),
    )
    human_decision = decision_human_review_decision_from_human_record(
        request=pending.request,
        record=record,
    )
    accepted = AuthoritativeAcceptedDecision(
        identity=challenged.identity,
        artifact=challenged.artifact,
        lineage=challenged.lineage,
    )
    validate_consumed_human_review_decision(
        request=request,
        decision=human_decision,
        target_proposal_ref=proposal_ref,
    )
    assert lifecycle.stage is DecisionLifecycleStage.ADJUDICATION
    assert accepted.identity.version == challenged.identity.version


def test_revision_invalidates_human_authority_for_next_version() -> None:
    challenged_v1 = _candidate(text="v1")
    proposal_ref_v1 = candidate_decision_ref(challenged_v1)
    request_v1 = _review_request(challenged_v1)
    approval_v1 = decision_human_review_decision_from_human_record(
        request=request_v1,
        record=build_human_decision_record(
            task_id=str(challenged_v1.identity.execution.task_id),
            tenant_id=challenged_v1.identity.tenant_id,
            approver=local_development_approver_evidence(
                tenant_id=challenged_v1.identity.tenant_id,
            ),
            verdict=HumanResponseVerdict.APPROVE,
            response_text="approved",
            human_request_id=str(request_v1.request_id),
        ),
    )
    revision_decision = evaluate_decision_revision(
        policy=decision_revision_policy(max_revisions=1),
        state=initial_decision_revision_state(proposal_ref_v1),
        verification_result=_challenged_result(proposal_ref_v1),
    )
    authorization = decision_revision_authorization(revision_decision=revision_decision)
    revised_v2, _ = mint_revised_candidate_decision(
        challenged=challenged_v1,
        authorization=authorization,
        artifact_kind="demo.payload",
        revised_payload=Payload(text="v2"),
        revision_state=initial_decision_revision_state(proposal_ref_v1),
    )
    with pytest.raises(DecisionHumanReviewMismatchError):
        validate_consumed_human_review_decision(
            request=request_v1,
            decision=approval_v1,
            target_proposal_ref=candidate_decision_ref(revised_v2),
        )
