# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from intergrax.contracts.decision_authorization import (
    DecisionGovernanceDecision,
    DecisionGovernanceDisposition,
    DecisionGovernanceEvaluationInput,
    DecisionGovernanceMismatchError,
    authoritative_decision_ref,
    decision_execution_action,
    decision_governance_policy_context,
    evaluate_decision_governance_with,
    validate_decision_execution_action_kind,
)
from intergrax.contracts.decision_human_review import (
    DecisionHumanReviewOutcome,
    decision_human_review_decision,
    decision_human_review_request,
    revision_exhausted_human_review_reason,
)
from intergrax.contracts.decision_identity import (
    DecisionExecutionLineage,
    DecisionIdentity,
    DecisionScope,
    DecisionVersion,
    initial_decision_version,
    mint_decision_id,
    next_decision_version,
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
from intergrax.contracts.declarative_hitl import DeclarativeHitlApprovalGrant
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.human_approver import local_development_approver_evidence
from intergrax.contracts.decision_human_review import DecisionHumanReviewProvenance
from intergrax.runtime.decision_authorization import (
    human_approval_does_not_imply_governance_allow,
    mint_validated_execution_authorization,
    validate_execution_authorization_bundle,
    verification_pass_does_not_imply_governance_allow,
)
from intergrax.runtime.decision_human_review import (
    decision_human_review_decision_from_human_record,
    request_decision_human_review,
    validate_consumed_human_review_decision,
)
from intergrax.runtime.decision_revision import mint_revised_candidate_decision
from intergrax.runtime.human.models import HumanResponseVerdict, build_human_decision_record

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_MODULE_PATHS = (
    Path("intergrax/contracts/decision_authorization.py"),
    Path("intergrax/runtime/decision_authorization.py"),
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


class _StaticGovernanceEvaluator:
    def __init__(self, decision: DecisionGovernanceDecision) -> None:
        self._decision = decision

    def evaluate(
        self,
        *,
        evaluation_input: DecisionGovernanceEvaluationInput[Payload],
    ) -> DecisionGovernanceDecision:
        return self._decision


class _WrongVersionGovernanceEvaluator:
    def evaluate(
        self,
        *,
        evaluation_input: DecisionGovernanceEvaluationInput[Payload],
    ) -> DecisionGovernanceDecision:
        ref = authoritative_decision_ref(evaluation_input.decision)
        wrong_identity = DecisionIdentity(
            decision_id=evaluation_input.decision.identity.decision_id,
            version=next_decision_version(evaluation_input.decision.identity.version),
            scope=evaluation_input.decision.identity.scope,
            tenant_id=evaluation_input.decision.identity.tenant_id,
            execution=evaluation_input.decision.identity.execution,
        )
        from intergrax.contracts.decision_authorization import AuthoritativeDecisionRef

        wrong_ref = AuthoritativeDecisionRef(
            identity=wrong_identity,
            lineage_ref=decision_lineage_ref(wrong_identity.version),
        )
        return DecisionGovernanceDecision(
            disposition=DecisionGovernanceDisposition.ALLOW,
            decision_ref=wrong_ref,
            action=evaluation_input.action,
            policy_context=evaluation_input.policy_context,
            tenant_id=evaluation_input.decision.identity.tenant_id,
        )


def _execution_lineage() -> DecisionExecutionLineage:
    return DecisionExecutionLineage(
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )


def _candidate(*, text: str = "draft") -> CandidateDecision[Payload]:
    identity = DecisionIdentity(
        decision_id=mint_decision_id(),
        version=initial_decision_version(),
        scope=DecisionScope(namespace="demo", subject="case-1"),
        tenant_id="tenant-a",
        execution=_execution_lineage(),
    )
    lineage = decision_version_lineage(
        current=decision_lineage_ref(identity.version, validate_decision_branch_id("main")),
    )
    return CandidateDecision(
        identity=identity,
        artifact=DecisionArtifact(
            kind=validate_decision_artifact_kind("demo.payload"),
            content=Payload(text=text),
        ),
        lineage=lineage,
    )


def _accepted(candidate: CandidateDecision[Payload]) -> AuthoritativeAcceptedDecision[Payload]:
    return AuthoritativeAcceptedDecision(
        identity=candidate.identity,
        artifact=candidate.artifact,
        lineage=candidate.lineage,
    )


def _action() -> object:
    return decision_execution_action(
        kind=validate_decision_execution_action_kind("tool.side_effect"),
        subject="notify.ops",
    )


def _policy() -> object:
    return decision_governance_policy_context(
        policy_provenance_digest="policy-digest-a",
        matched_rule_ids=("rule.allow.notify",),
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


def _passed_result(proposal_ref) -> VerificationResult:
    return verification_result(
        proposal_ref=proposal_ref,
        disposition=VerificationDisposition.PASSED,
        stage_records=(
            verification_stage_record(
                proposal_ref=proposal_ref,
                stage=validate_verification_stage_kind("structural"),
                outcome=VerificationStageOutcome.PASSED,
            ),
        ),
    )


def test_forbidden_patterns_absent_in_authorization_modules() -> None:
    for module_path in _MODULE_PATHS:
        source = module_path.read_text(encoding="utf-8")
        for fragment in _FORBIDDEN_FRAGMENTS:
            assert fragment not in source


def test_three_authority_types_are_structurally_distinct() -> None:
    candidate = _candidate()
    accepted = _accepted(candidate)
    action = _action()
    policy = _policy()
    human_request = decision_human_review_request(
        proposal_ref=candidate_decision_ref(candidate),
        reason_code=revision_exhausted_human_review_reason(),
    )
    human_decision = decision_human_review_decision(
        request=human_request,
        outcome=DecisionHumanReviewOutcome.APPROVED,
        approver=local_development_approver_evidence(tenant_id=candidate.identity.tenant_id),
        provenance=DecisionHumanReviewProvenance(
            human_record_id="hdec_1",
            human_request_id=str(human_request.request_id),
        ),
    )
    governance = DecisionGovernanceDecision(
        disposition=DecisionGovernanceDisposition.ALLOW,
        decision_ref=authoritative_decision_ref(accepted),
        action=action,
        policy_context=policy,
        tenant_id=accepted.identity.tenant_id,
    )
    authorization = mint_validated_execution_authorization(
        evaluation_input=DecisionGovernanceEvaluationInput(
            decision=accepted,
            action=action,
            policy_context=policy,
        ),
        governance_decision=governance,
    )
    tool_grant = DeclarativeHitlApprovalGrant(
        grant_id="grant-1",
        invocation_scope_id="scope-1",
        task_id=str(candidate.identity.execution.task_id),
        run_id=str(candidate.identity.execution.run_id),
        step_id="step-1",
        tool_id="tool.notify",
        agent_id="agent-1",
        idempotency_key="idem-1",
        matched_rule_ids=("rule.hitl",),
        human_request_id="human-1",
        policy_provenance_digest="policy-digest-a",
        pause_id="pause-1",
        approved_at="2026-01-01T00:00:00Z",
    )
    assert type(human_decision).__name__ != type(authorization).__name__
    assert type(authorization).__name__ != type(tool_grant).__name__
    assert type(human_decision).__name__ != type(tool_grant).__name__


def test_full_safe_flow_through_governance_authorization() -> None:
    challenged = _candidate()
    proposal_ref = candidate_decision_ref(challenged)
    revision_decision = evaluate_decision_revision(
        policy=decision_revision_policy(max_revisions=0),
        state=initial_decision_revision_state(proposal_ref),
        verification_result=_challenged_result(proposal_ref),
    )
    assert revision_decision.disposition is DecisionRevisionDisposition.EXHAUSTED
    request = decision_human_review_request(
        proposal_ref=proposal_ref,
        reason_code=revision_exhausted_human_review_reason(),
    )
    pending = request_decision_human_review(request)
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
    validate_consumed_human_review_decision(
        request=request,
        decision=human_decision,
        target_proposal_ref=proposal_ref,
    )
    accepted = _accepted(challenged)
    action = _action()
    policy = _policy()
    governance = DecisionGovernanceDecision(
        disposition=DecisionGovernanceDisposition.ALLOW,
        decision_ref=authoritative_decision_ref(accepted),
        action=action,
        policy_context=policy,
        tenant_id=accepted.identity.tenant_id,
    )
    authorization = mint_validated_execution_authorization(
        evaluation_input=DecisionGovernanceEvaluationInput(
            decision=accepted,
            action=action,
            policy_context=policy,
        ),
        governance_decision=governance,
    )
    validate_execution_authorization_bundle(
        authorization=authorization,
        decision=accepted,
        action=action,
    )


def test_human_approved_and_governance_deny_remain_unauthorized() -> None:
    accepted = _accepted(_candidate())
    action = _action()
    policy = _policy()
    governance = DecisionGovernanceDecision(
        disposition=DecisionGovernanceDisposition.DENY,
        decision_ref=authoritative_decision_ref(accepted),
        action=action,
        policy_context=policy,
        tenant_id=accepted.identity.tenant_id,
    )
    assert human_approval_does_not_imply_governance_allow(
        human_outcome_approved=True,
        governance_decision=governance,
    )
    with pytest.raises(ValueError, match="ALLOW"):
        mint_validated_execution_authorization(
            evaluation_input=DecisionGovernanceEvaluationInput(
                decision=accepted,
                action=action,
                policy_context=policy,
            ),
            governance_decision=governance,
        )


def test_verification_pass_and_governance_deny_remain_unauthorized() -> None:
    candidate = _candidate()
    proposal_ref = candidate_decision_ref(candidate)
    accepted = _accepted(candidate)
    action = _action()
    policy = _policy()
    governance = DecisionGovernanceDecision(
        disposition=DecisionGovernanceDisposition.DENY,
        decision_ref=authoritative_decision_ref(accepted),
        action=action,
        policy_context=policy,
        tenant_id=accepted.identity.tenant_id,
    )
    assert verification_pass_does_not_imply_governance_allow(
        verification_result=_passed_result(proposal_ref),
        governance_decision=governance,
    )


def test_plugin_evaluator_output_validated_before_use() -> None:
    accepted = _accepted(_candidate())
    action = _action()
    policy = _policy()
    evaluation_input = DecisionGovernanceEvaluationInput(
        decision=accepted,
        action=action,
        policy_context=policy,
    )
    allow_decision = DecisionGovernanceDecision(
        disposition=DecisionGovernanceDisposition.ALLOW,
        decision_ref=authoritative_decision_ref(accepted),
        action=action,
        policy_context=policy,
        tenant_id=accepted.identity.tenant_id,
    )
    validated = evaluate_decision_governance_with(
        evaluator=_StaticGovernanceEvaluator(allow_decision),
        evaluation_input=evaluation_input,
    )
    assert validated.disposition is DecisionGovernanceDisposition.ALLOW


def test_plugin_cannot_substitute_different_decision_version() -> None:
    accepted = _accepted(_candidate())
    evaluation_input = DecisionGovernanceEvaluationInput(
        decision=accepted,
        action=_action(),
        policy_context=_policy(),
    )
    with pytest.raises(DecisionGovernanceMismatchError):
        evaluate_decision_governance_with(
            evaluator=_WrongVersionGovernanceEvaluator(),
            evaluation_input=evaluation_input,
        )


def test_revision_invalidates_execution_authorization_for_v2() -> None:
    challenged_v1 = _candidate(text="v1")
    proposal_ref_v1 = candidate_decision_ref(challenged_v1)
    accepted_v1 = _accepted(challenged_v1)
    action = _action()
    policy = _policy()
    governance_v1 = DecisionGovernanceDecision(
        disposition=DecisionGovernanceDisposition.ALLOW,
        decision_ref=authoritative_decision_ref(accepted_v1),
        action=action,
        policy_context=policy,
        tenant_id=accepted_v1.identity.tenant_id,
    )
    authorization_v1 = mint_validated_execution_authorization(
        evaluation_input=DecisionGovernanceEvaluationInput(
            decision=accepted_v1,
            action=action,
            policy_context=policy,
        ),
        governance_decision=governance_v1,
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
    accepted_v2 = _accepted(revised_v2)
    with pytest.raises(DecisionGovernanceMismatchError):
        validate_execution_authorization_bundle(
            authorization=authorization_v1,
            decision=accepted_v2,
            action=action,
        )
