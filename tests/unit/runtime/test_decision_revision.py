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
)
from intergrax.contracts.decision_lifecycle import (
    DecisionLifecycleStage,
    initial_decision_lifecycle_state,
    transition_decision_lifecycle,
)
from intergrax.contracts.decision_record import (
    CandidateDecision,
    DecisionArtifact,
    DecisionBranchId,
    DecisionProposalRef,
    candidate_decision_ref,
    decision_lineage_ref,
    decision_proposal_ref,
    decision_version_lineage,
    validate_decision_artifact_kind,
    validate_decision_branch_id,
)
from intergrax.contracts.decision_revision import (
    DecisionRevisionAuthorization,
    DecisionRevisionDisposition,
    DecisionRevisionState,
    DecisionRevisionStateMismatchError,
    decision_revision_authorization,
    decision_revision_policy,
    evaluate_decision_revision,
    initial_decision_revision_state,
    proposal_refs_match,
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
from intergrax.runtime.decision_revision import (
    DecisionRevisionAuthorizationMismatchError,
    mint_revised_candidate_decision,
    transition_lifecycle_for_revision,
    validate_revision_authorization_for_candidate,
)
from intergrax.runtime.critic.evaluator_loop_spec import EvaluatorLoopSpec

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_MODULE_PATHS = (
    Path("intergrax/contracts/decision_revision.py"),
    Path("intergrax/runtime/decision_revision.py"),
)
_FORBIDDEN_FRAGMENTS = (
    "runtime.nexus",
    "runtime.critic",
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


def _revision_state(
    candidate: CandidateDecision[Payload],
    *,
    revision_count: int = 0,
) -> DecisionRevisionState:
    return DecisionRevisionState(
        proposal_ref=candidate_decision_ref(candidate),
        revision_count=revision_count,
    )


def _initial_state(candidate: CandidateDecision[Payload]) -> DecisionRevisionState:
    return initial_decision_revision_state(candidate_decision_ref(candidate))


def _challenged_result(proposal_ref: DecisionProposalRef) -> VerificationResult:
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


def _passed_result(proposal_ref: DecisionProposalRef) -> VerificationResult:
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


def test_forbidden_patterns_absent_in_revision_modules() -> None:
    for module_path in _MODULE_PATHS:
        source = module_path.read_text(encoding="utf-8")
        for fragment in _FORBIDDEN_FRAGMENTS:
            assert fragment not in source


def test_revision_modules_do_not_import_critic_or_nexus() -> None:
    for module_path in _MODULE_PATHS:
        source = module_path.read_text(encoding="utf-8")
        assert "runtime.critic" not in source
        assert "runtime.nexus" not in source


def test_v1_challenged_count_zero_max_two_allowed() -> None:
    candidate = _candidate()
    proposal_ref = candidate_decision_ref(candidate)
    policy = decision_revision_policy(max_revisions=2)
    decision = evaluate_decision_revision(
        policy=policy,
        state=_initial_state(candidate),
        verification_result=_challenged_result(proposal_ref),
    )
    assert decision.disposition is DecisionRevisionDisposition.ALLOWED
    assert decision.revision_number == 1
    assert decision.policy == policy


def test_passing_result_revision_not_required() -> None:
    candidate = _candidate()
    proposal_ref = candidate_decision_ref(candidate)
    policy = decision_revision_policy(max_revisions=2)
    decision = evaluate_decision_revision(
        policy=policy,
        state=_initial_state(candidate),
        verification_result=_passed_result(proposal_ref),
    )
    assert decision.disposition is DecisionRevisionDisposition.NOT_REQUIRED
    assert decision.revision_number is None
    assert decision.policy == policy


def test_zero_budget_exhausted_on_challenge() -> None:
    candidate = _candidate()
    proposal_ref = candidate_decision_ref(candidate)
    policy = decision_revision_policy(max_revisions=0)
    decision = evaluate_decision_revision(
        policy=policy,
        state=_initial_state(candidate),
        verification_result=_challenged_result(proposal_ref),
    )
    assert decision.disposition is DecisionRevisionDisposition.EXHAUSTED
    assert decision.policy == policy


def test_count_one_max_two_allowed() -> None:
    candidate = _candidate()
    proposal_ref = candidate_decision_ref(candidate)
    decision = evaluate_decision_revision(
        policy=decision_revision_policy(max_revisions=2),
        state=_revision_state(candidate, revision_count=1),
        verification_result=_challenged_result(proposal_ref),
    )
    assert decision.disposition is DecisionRevisionDisposition.ALLOWED
    assert decision.revision_number == 2


def test_count_two_max_two_exhausted() -> None:
    candidate = _candidate()
    proposal_ref = candidate_decision_ref(candidate)
    decision = evaluate_decision_revision(
        policy=decision_revision_policy(max_revisions=2),
        state=_revision_state(candidate, revision_count=2),
        verification_result=_challenged_result(proposal_ref),
    )
    assert decision.disposition is DecisionRevisionDisposition.EXHAUSTED


def test_mint_revised_candidate_increments_version_once() -> None:
    challenged = _candidate()
    proposal_ref = candidate_decision_ref(challenged)
    revision_decision = evaluate_decision_revision(
        policy=decision_revision_policy(max_revisions=2),
        state=_initial_state(challenged),
        verification_result=_challenged_result(proposal_ref),
    )
    authorization = decision_revision_authorization(
        revision_decision=revision_decision,
    )
    revised, next_state = mint_revised_candidate_decision(
        challenged=challenged,
        authorization=authorization,
        artifact_kind="demo.payload",
        revised_payload=Payload(text="revised"),
        revision_state=_initial_state(challenged),
    )
    assert revised.identity.version.value == challenged.identity.version.value + 1
    assert next_state.revision_count == 1
    assert next_state.proposal_ref == candidate_decision_ref(revised)


def test_revised_parent_lineage_matches_exact_v1_ref() -> None:
    challenged = _candidate()
    proposal_ref = candidate_decision_ref(challenged)
    revision_decision = evaluate_decision_revision(
        policy=decision_revision_policy(max_revisions=2),
        state=_initial_state(challenged),
        verification_result=_challenged_result(proposal_ref),
    )
    authorization = decision_revision_authorization(
        revision_decision=revision_decision,
    )
    revised, _ = mint_revised_candidate_decision(
        challenged=challenged,
        authorization=authorization,
        artifact_kind="demo.payload",
        revised_payload=Payload(text="revised"),
        revision_state=_initial_state(challenged),
    )
    assert revised.lineage.parents == (proposal_ref.lineage_ref,)


def test_sibling_branch_authorization_rejected() -> None:
    decision_id = mint_decision_id()
    identity_a = _identity(version=initial_decision_version(), decision_id=decision_id)
    identity_b = _identity(version=initial_decision_version(), decision_id=decision_id)
    candidate_a = _candidate(identity=identity_a, branch_id="A")
    candidate_b = _candidate(identity=identity_b, branch_id="B")
    proposal_ref_a = candidate_decision_ref(candidate_a)
    revision_decision = evaluate_decision_revision(
        policy=decision_revision_policy(max_revisions=1),
        state=_initial_state(candidate_a),
        verification_result=_challenged_result(proposal_ref_a),
    )
    authorization = decision_revision_authorization(
        revision_decision=revision_decision,
    )
    with pytest.raises(DecisionRevisionAuthorizationMismatchError):
        mint_revised_candidate_decision(
            challenged=candidate_b,
            authorization=authorization,
            artifact_kind="demo.payload",
            revised_payload=Payload(text="revised"),
            revision_state=_initial_state(candidate_b),
        )


def test_stale_authorization_for_v1_cannot_revise_v2() -> None:
    challenged_v1 = _candidate()
    proposal_ref_v1 = candidate_decision_ref(challenged_v1)
    revision_decision = evaluate_decision_revision(
        policy=decision_revision_policy(max_revisions=2),
        state=_initial_state(challenged_v1),
        verification_result=_challenged_result(proposal_ref_v1),
    )
    authorization = decision_revision_authorization(
        revision_decision=revision_decision,
    )
    revised_v2, state_after_v2 = mint_revised_candidate_decision(
        challenged=challenged_v1,
        authorization=authorization,
        artifact_kind="demo.payload",
        revised_payload=Payload(text="v2"),
        revision_state=_initial_state(challenged_v1),
    )
    with pytest.raises(DecisionRevisionAuthorizationMismatchError):
        validate_revision_authorization_for_candidate(
            authorization=authorization,
            candidate=revised_v2,
        )
    assert state_after_v2.revision_count == 1
    assert state_after_v2.proposal_ref == candidate_decision_ref(revised_v2)


def test_identity_preserved_on_revision() -> None:
    challenged = _candidate()
    proposal_ref = candidate_decision_ref(challenged)
    revision_decision = evaluate_decision_revision(
        policy=decision_revision_policy(max_revisions=2),
        state=_initial_state(challenged),
        verification_result=_challenged_result(proposal_ref),
    )
    authorization = decision_revision_authorization(
        revision_decision=revision_decision,
    )
    revised, _ = mint_revised_candidate_decision(
        challenged=challenged,
        authorization=authorization,
        artifact_kind="demo.payload",
        revised_payload=Payload(text="revised"),
        revision_state=_initial_state(challenged),
    )
    assert revised.identity.decision_id == challenged.identity.decision_id
    assert revised.identity.scope == challenged.identity.scope
    assert revised.identity.tenant_id == challenged.identity.tenant_id
    assert revised.identity.execution == challenged.identity.execution


def test_revised_content_supplied_by_caller() -> None:
    challenged = _candidate(text="original")
    proposal_ref = candidate_decision_ref(challenged)
    revision_decision = evaluate_decision_revision(
        policy=decision_revision_policy(max_revisions=1),
        state=_initial_state(challenged),
        verification_result=_challenged_result(proposal_ref),
    )
    authorization = decision_revision_authorization(
        revision_decision=revision_decision,
    )
    revised, _ = mint_revised_candidate_decision(
        challenged=challenged,
        authorization=authorization,
        artifact_kind="demo.payload",
        revised_payload=Payload(text="caller-supplied"),
        revision_state=_initial_state(challenged),
    )
    assert revised.artifact.content.text == "caller-supplied"


def test_lifecycle_verification_to_revision_transition() -> None:
    candidate = _candidate()
    identity = candidate.identity
    lifecycle = transition_decision_lifecycle(
        initial_decision_lifecycle_state(identity),
        DecisionLifecycleStage.VERIFICATION,
    )
    proposal_ref = candidate_decision_ref(candidate)
    verification = _challenged_result(proposal_ref)
    revision_decision = evaluate_decision_revision(
        policy=decision_revision_policy(max_revisions=2),
        state=_initial_state(candidate),
        verification_result=verification,
    )
    next_lifecycle = transition_lifecycle_for_revision(
        lifecycle_state=lifecycle,
        verification_result=verification,
        revision_decision=revision_decision,
    )
    assert next_lifecycle.stage is DecisionLifecycleStage.REVISION
    assert next_lifecycle.transition_index == lifecycle.transition_index + 1


def test_legacy_evaluator_loop_semantic_parity_revision_permitted() -> None:
    """Legacy REVISE with remaining iterations maps to challenged + remaining budget."""
    remaining_iterations = 2
    legacy_spec = EvaluatorLoopSpec(
        max_iterations=remaining_iterations,
        revise_node_id="revise-node",
    )
    assert legacy_spec.max_iterations == remaining_iterations
    assert legacy_spec.revise_node_id == "revise-node"
    candidate = _candidate()
    proposal_ref = candidate_decision_ref(candidate)
    decision = evaluate_decision_revision(
        policy=decision_revision_policy(max_revisions=remaining_iterations),
        state=_initial_state(candidate),
        verification_result=_challenged_result(proposal_ref),
    )
    assert decision.disposition is DecisionRevisionDisposition.ALLOWED


def test_exhaustion_does_not_auto_hitl_or_reject() -> None:
    candidate = _candidate()
    proposal_ref = candidate_decision_ref(candidate)
    decision = evaluate_decision_revision(
        policy=decision_revision_policy(max_revisions=0),
        state=_initial_state(candidate),
        verification_result=_challenged_result(proposal_ref),
    )
    assert decision.disposition is DecisionRevisionDisposition.EXHAUSTED
    assert decision.disposition.value == "exhausted"


def test_proposal_refs_match_requires_exact_branch() -> None:
    identity = _identity()
    ref_a = decision_proposal_ref(
        identity=identity,
        lineage_ref=decision_lineage_ref(identity.version, DecisionBranchId("A")),
    )
    ref_b = decision_proposal_ref(
        identity=identity,
        lineage_ref=decision_lineage_ref(identity.version, DecisionBranchId("B")),
    )
    assert not proposal_refs_match(ref_a, ref_b)


def test_initial_state_binds_exact_proposal_ref_and_zero_count() -> None:
    candidate = _candidate()
    state = _initial_state(candidate)
    assert state.proposal_ref == candidate_decision_ref(candidate)
    assert state.revision_count == 0


def test_cross_decision_state_reuse_rejected() -> None:
    candidate_a = _candidate()
    candidate_b = _candidate()
    state_a = _initial_state(candidate_a)
    with pytest.raises(DecisionRevisionStateMismatchError):
        evaluate_decision_revision(
            policy=decision_revision_policy(max_revisions=2),
            state=state_a,
            verification_result=_challenged_result(candidate_decision_ref(candidate_b)),
        )


def test_cross_branch_state_reuse_rejected() -> None:
    decision_id = mint_decision_id()
    identity = _identity(version=initial_decision_version(), decision_id=decision_id)
    candidate_a = _candidate(identity=identity, branch_id="A")
    candidate_b = _candidate(identity=identity, branch_id="B")
    state_a = _initial_state(candidate_a)
    with pytest.raises(DecisionRevisionStateMismatchError):
        evaluate_decision_revision(
            policy=decision_revision_policy(max_revisions=2),
            state=state_a,
            verification_result=_challenged_result(candidate_decision_ref(candidate_b)),
        )


def test_cross_lineage_state_reuse_rejected() -> None:
    decision_id = mint_decision_id()
    version = initial_decision_version()
    identity_a = _identity(version=version, decision_id=decision_id)
    identity_b = _identity(version=version, decision_id=decision_id)
    candidate_a = _candidate(identity=identity_a, branch_id="main")
    candidate_b = _candidate(identity=identity_b, branch_id="main")
    state_a = _initial_state(candidate_a)
    with pytest.raises(DecisionRevisionStateMismatchError):
        evaluate_decision_revision(
            policy=decision_revision_policy(max_revisions=2),
            state=state_a,
            verification_result=_challenged_result(candidate_decision_ref(candidate_b)),
        )


def test_stale_v1_state_cannot_evaluate_v2_result() -> None:
    challenged_v1 = _candidate()
    proposal_ref_v1 = candidate_decision_ref(challenged_v1)
    revision_decision = evaluate_decision_revision(
        policy=decision_revision_policy(max_revisions=2),
        state=_initial_state(challenged_v1),
        verification_result=_challenged_result(proposal_ref_v1),
    )
    authorization = decision_revision_authorization(revision_decision=revision_decision)
    revised_v2, _ = mint_revised_candidate_decision(
        challenged=challenged_v1,
        authorization=authorization,
        artifact_kind="demo.payload",
        revised_payload=Payload(text="v2"),
        revision_state=_initial_state(challenged_v1),
    )
    state_v1 = _initial_state(challenged_v1)
    with pytest.raises(DecisionRevisionStateMismatchError):
        evaluate_decision_revision(
            policy=decision_revision_policy(max_revisions=2),
            state=state_v1,
            verification_result=_challenged_result(candidate_decision_ref(revised_v2)),
        )


def test_v1_to_v2_next_state_binds_exact_ref_and_count() -> None:
    challenged_v1 = _candidate()
    proposal_ref_v1 = candidate_decision_ref(challenged_v1)
    revision_decision = evaluate_decision_revision(
        policy=decision_revision_policy(max_revisions=2),
        state=_initial_state(challenged_v1),
        verification_result=_challenged_result(proposal_ref_v1),
    )
    authorization = decision_revision_authorization(revision_decision=revision_decision)
    revised_v2, state_v2 = mint_revised_candidate_decision(
        challenged=challenged_v1,
        authorization=authorization,
        artifact_kind="demo.payload",
        revised_payload=Payload(text="v2"),
        revision_state=_initial_state(challenged_v1),
    )
    assert state_v2.revision_count == 1
    assert state_v2.proposal_ref == candidate_decision_ref(revised_v2)


def test_v2_to_v3_next_state_binds_exact_ref_and_count() -> None:
    challenged_v1 = _candidate()
    proposal_ref_v1 = candidate_decision_ref(challenged_v1)
    policy = decision_revision_policy(max_revisions=3)
    revision_v1 = evaluate_decision_revision(
        policy=policy,
        state=_initial_state(challenged_v1),
        verification_result=_challenged_result(proposal_ref_v1),
    )
    auth_v1 = decision_revision_authorization(revision_decision=revision_v1)
    revised_v2, state_v2 = mint_revised_candidate_decision(
        challenged=challenged_v1,
        authorization=auth_v1,
        artifact_kind="demo.payload",
        revised_payload=Payload(text="v2"),
        revision_state=_initial_state(challenged_v1),
    )
    revision_v2 = evaluate_decision_revision(
        policy=policy,
        state=state_v2,
        verification_result=_challenged_result(candidate_decision_ref(revised_v2)),
    )
    auth_v2 = decision_revision_authorization(revision_decision=revision_v2)
    revised_v3, state_v3 = mint_revised_candidate_decision(
        challenged=revised_v2,
        authorization=auth_v2,
        artifact_kind="demo.payload",
        revised_payload=Payload(text="v3"),
        revision_state=state_v2,
    )
    assert state_v3.revision_count == 2
    assert state_v3.proposal_ref == candidate_decision_ref(revised_v3)


def test_authorization_inherits_decision_policy() -> None:
    candidate = _candidate()
    policy = decision_revision_policy(max_revisions=2)
    revision_decision = evaluate_decision_revision(
        policy=policy,
        state=_initial_state(candidate),
        verification_result=_challenged_result(candidate_decision_ref(candidate)),
    )
    authorization = decision_revision_authorization(revision_decision=revision_decision)
    assert authorization.policy == policy
    assert authorization.policy == revision_decision.policy


def test_authorization_revision_number_exceeding_policy_rejected() -> None:
    candidate = _candidate()
    proposal_ref = candidate_decision_ref(candidate)
    policy = decision_revision_policy(max_revisions=1)
    with pytest.raises(ValueError, match="exceeds policy.max_revisions"):
        DecisionRevisionAuthorization(
            proposal_ref=proposal_ref,
            policy=policy,
            revision_number=2,
        )


def test_stale_state_cannot_mint_with_v1_state_and_v2_candidate() -> None:
    challenged_v1 = _candidate()
    proposal_ref_v1 = candidate_decision_ref(challenged_v1)
    revision_decision = evaluate_decision_revision(
        policy=decision_revision_policy(max_revisions=2),
        state=_initial_state(challenged_v1),
        verification_result=_challenged_result(proposal_ref_v1),
    )
    authorization = decision_revision_authorization(revision_decision=revision_decision)
    revised_v2, state_v2 = mint_revised_candidate_decision(
        challenged=challenged_v1,
        authorization=authorization,
        artifact_kind="demo.payload",
        revised_payload=Payload(text="v2"),
        revision_state=_initial_state(challenged_v1),
    )
    state_v1 = _initial_state(challenged_v1)
    with pytest.raises(DecisionRevisionAuthorizationMismatchError):
        mint_revised_candidate_decision(
            challenged=revised_v2,
            authorization=authorization,
            artifact_kind="demo.payload",
            revised_payload=Payload(text="v3"),
            revision_state=state_v1,
        )
    assert state_v2.revision_count == 1


def test_auth_v1_with_state_v2_rejected() -> None:
    challenged_v1 = _candidate()
    proposal_ref_v1 = candidate_decision_ref(challenged_v1)
    revision_decision = evaluate_decision_revision(
        policy=decision_revision_policy(max_revisions=2),
        state=_initial_state(challenged_v1),
        verification_result=_challenged_result(proposal_ref_v1),
    )
    authorization = decision_revision_authorization(revision_decision=revision_decision)
    revised_v2, state_v2 = mint_revised_candidate_decision(
        challenged=challenged_v1,
        authorization=authorization,
        artifact_kind="demo.payload",
        revised_payload=Payload(text="v2"),
        revision_state=_initial_state(challenged_v1),
    )
    with pytest.raises(DecisionRevisionAuthorizationMismatchError):
        mint_revised_candidate_decision(
            challenged=challenged_v1,
            authorization=authorization,
            artifact_kind="demo.payload",
            revised_payload=Payload(text="retry"),
            revision_state=state_v2,
        )


def test_auth_v1_with_state_other_branch_rejected() -> None:
    decision_id = mint_decision_id()
    identity = _identity(version=initial_decision_version(), decision_id=decision_id)
    candidate_a = _candidate(identity=identity, branch_id="A")
    candidate_b = _candidate(identity=identity, branch_id="B")
    proposal_ref_a = candidate_decision_ref(candidate_a)
    revision_decision = evaluate_decision_revision(
        policy=decision_revision_policy(max_revisions=1),
        state=_initial_state(candidate_a),
        verification_result=_challenged_result(proposal_ref_a),
    )
    authorization = decision_revision_authorization(revision_decision=revision_decision)
    with pytest.raises(DecisionRevisionAuthorizationMismatchError):
        mint_revised_candidate_decision(
            challenged=candidate_a,
            authorization=authorization,
            artifact_kind="demo.payload",
            revised_payload=Payload(text="revised"),
            revision_state=_initial_state(candidate_b),
        )
