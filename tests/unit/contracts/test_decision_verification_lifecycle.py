# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import re
from pathlib import Path

import pytest

from intergrax.contracts.decision_identity import (
    DecisionExecutionLineage,
    DecisionId,
    DecisionIdentity,
    DecisionScope,
    DecisionVersion,
    initial_decision_version,
    mint_decision_id,
    next_decision_version,
)
from intergrax.contracts.decision_lifecycle import (
    DecisionLifecycleStage,
    DecisionLifecycleState,
    initial_decision_lifecycle_state,
    transition_decision_lifecycle,
)
from dataclasses import fields

from intergrax.contracts.decision_record import (
    DecisionBranchId,
    DecisionProposalRef,
    decision_lineage_ref,
    decision_proposal_ref,
)
from intergrax.contracts.decision_verification import (
    VerificationDisposition,
    VerificationFinding,
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
from intergrax.contracts.decision_verification_lifecycle import (
    handoff_verification_result,
    validate_decision_verification_handoff,
)
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    TaskId,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)

_MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "intergrax"
    / "contracts"
    / "decision_verification_lifecycle.py"
)

_NON_VERIFICATION_STAGES = tuple(
    stage
    for stage in DecisionLifecycleStage
    if stage is not DecisionLifecycleStage.VERIFICATION
)

_FORBIDDEN_PRODUCTION_PATTERNS = (
    r"\bAny\b",
    r"\bcast\b",
    r"type:\s*ignore",
    r"pyright:\s*ignore",
    r"\bgetattr\b",
    r"\bsetattr\b",
    r"\bhasattr\b",
    r"\binspect\b",
    r"\bexec\b",
    r"\beval\b",
    r"object\.__setattr__",
    r"dict\[str,\s*Any\]",
)


def _execution_lineage(
    *,
    task_id: TaskId | None = None,
    run_id: RunId | None = None,
    attempt_id: AttemptId | None = None,
    execution_id: ExecutionId | None = None,
) -> DecisionExecutionLineage:
    return DecisionExecutionLineage(
        task_id=mint_task_id() if task_id is None else task_id,
        run_id=mint_run_id() if run_id is None else run_id,
        attempt_id=mint_attempt_id() if attempt_id is None else attempt_id,
        execution_id=mint_execution_id() if execution_id is None else execution_id,
    )


def _identity(
    *,
    tenant_id: str = "tenant-a",
    namespace: str = "scope",
    subject: str = "subject-1",
    version: DecisionVersion | None = None,
    decision_id: DecisionId | None = None,
    execution: DecisionExecutionLineage | None = None,
) -> DecisionIdentity:
    return DecisionIdentity(
        decision_id=mint_decision_id() if decision_id is None else decision_id,
        version=version or initial_decision_version(),
        scope=DecisionScope(namespace=namespace, subject=subject),
        tenant_id=tenant_id,
        execution=execution or _execution_lineage(),
    )


def _proposal_ref(
    *,
    identity: DecisionIdentity | None = None,
    branch_id: str = "analysis-a",
    version: DecisionVersion | None = None,
) -> DecisionProposalRef:
    resolved_identity = identity or _identity()
    resolved_version = version or resolved_identity.version
    return decision_proposal_ref(
        identity=resolved_identity,
        lineage_ref=decision_lineage_ref(
            resolved_version,
            DecisionBranchId(branch_id),
        ),
    )


def _verification_state(
    identity: DecisionIdentity,
    *,
    transition_index: int = 1,
) -> DecisionLifecycleState:
    state = initial_decision_lifecycle_state(identity)
    state = transition_decision_lifecycle(state, DecisionLifecycleStage.VERIFICATION)
    if transition_index != state.transition_index:
        return DecisionLifecycleState(
            identity=identity,
            stage=DecisionLifecycleStage.VERIFICATION,
            transition_index=transition_index,
        )
    return state


def _passed_result(proposal_ref: DecisionProposalRef) -> VerificationResult:
    return verification_result(
        proposal_ref=proposal_ref,
        disposition=VerificationDisposition.PASSED,
        stage_records=(
            verification_stage_record(
                proposal_ref=proposal_ref,
                stage=validate_verification_stage_kind("structural_schema"),
                outcome=VerificationStageOutcome.PASSED,
            ),
        ),
    )


def _challenged_result(proposal_ref: DecisionProposalRef) -> VerificationResult:
    finding: VerificationFinding = verification_finding(
        code=validate_verification_finding_code("schema.missing_field"),
        message="required field summary is missing",
    )
    return verification_result(
        proposal_ref=proposal_ref,
        disposition=VerificationDisposition.CHALLENGED,
        stage_records=(
            verification_stage_record(
                proposal_ref=proposal_ref,
                stage=validate_verification_stage_kind("deterministic_rules"),
                outcome=VerificationStageOutcome.CHALLENGED,
                challenge=verification_challenge(
                    proposal_ref=proposal_ref,
                    stage=validate_verification_stage_kind("deterministic_rules"),
                    requirement_code=validate_verification_requirement_code("artifact.schema"),
                    finding=finding,
                ),
            ),
        ),
    )


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.parametrize(
    "disposition",
    [VerificationDisposition.PASSED, VerificationDisposition.CHALLENGED],
)
def test_valid_handoff_from_verification_stage(
    disposition: VerificationDisposition,
) -> None:
    identity = _identity()
    proposal_ref = _proposal_ref(identity=identity)
    state = _verification_state(identity, transition_index=2)
    result = (
        _passed_result(proposal_ref)
        if disposition is VerificationDisposition.PASSED
        else _challenged_result(proposal_ref)
    )

    handoff = handoff_verification_result(state=state, result=result)

    assert handoff.lifecycle_state is state
    assert handoff.verification_result is result
    revalidated = validate_decision_verification_handoff(handoff)
    assert revalidated == handoff
    assert revalidated is not handoff


@pytest.mark.unit
@pytest.mark.gate
def test_handoff_retains_exact_verification_result() -> None:
    identity = _identity(version=DecisionVersion(2))
    proposal_ref = _proposal_ref(identity=identity, version=DecisionVersion(2), branch_id="branch-x")
    state = _verification_state(identity)
    result = _challenged_result(proposal_ref)

    handoff = handoff_verification_result(state=state, result=result)

    assert handoff.verification_result is result
    assert handoff.verification_result.proposal_ref is proposal_ref
    assert handoff.verification_result.proposal_ref.lineage_ref.branch_id == "branch-x"


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.parametrize(
    "disposition",
    [VerificationDisposition.PASSED, VerificationDisposition.CHALLENGED],
)
def test_handoff_does_not_mutate_lifecycle_state(
    disposition: VerificationDisposition,
) -> None:
    identity = _identity()
    proposal_ref = _proposal_ref(identity=identity)
    state = _verification_state(identity, transition_index=2)
    before_stage = state.stage
    before_index = state.transition_index
    before_version = state.identity.version

    result = (
        _passed_result(proposal_ref)
        if disposition is VerificationDisposition.PASSED
        else _challenged_result(proposal_ref)
    )
    handoff = handoff_verification_result(state=state, result=result)

    assert handoff.lifecycle_state is state
    assert state.stage is before_stage
    assert state.stage is DecisionLifecycleStage.VERIFICATION
    assert state.transition_index == before_index == 2
    assert state.identity.version == before_version


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.parametrize("invalid_stage", _NON_VERIFICATION_STAGES)
def test_handoff_rejects_non_verification_stage(
    invalid_stage: DecisionLifecycleStage,
) -> None:
    identity = _identity()
    proposal_ref = _proposal_ref(identity=identity)
    state = DecisionLifecycleState(
        identity=identity,
        stage=invalid_stage,
        transition_index=3,
    )
    with pytest.raises(ValueError, match="requires lifecycle stage verification"):
        handoff_verification_result(state=state, result=_passed_result(proposal_ref))


@pytest.mark.unit
@pytest.mark.gate
def test_handoff_rejects_version_mismatch() -> None:
    decision_id = mint_decision_id()
    lineage = _execution_lineage()
    lifecycle_identity = _identity(
        decision_id=decision_id,
        version=initial_decision_version(),
        execution=lineage,
    )
    verification_identity = DecisionIdentity(
        decision_id=decision_id,
        version=next_decision_version(initial_decision_version()),
        scope=lifecycle_identity.scope,
        tenant_id=lifecycle_identity.tenant_id,
        execution=lineage,
    )
    state = _verification_state(lifecycle_identity)
    result = _passed_result(_proposal_ref(identity=verification_identity))

    with pytest.raises(ValueError, match="requires lifecycle identity to match"):
        handoff_verification_result(state=state, result=result)


@pytest.mark.unit
@pytest.mark.gate
def test_handoff_rejects_decision_id_mismatch() -> None:
    lineage = _execution_lineage()
    lifecycle_identity = _identity(execution=lineage)
    verification_identity = _identity(execution=lineage)
    state = _verification_state(lifecycle_identity)
    result = _passed_result(_proposal_ref(identity=verification_identity))

    with pytest.raises(ValueError, match="requires lifecycle identity to match"):
        handoff_verification_result(state=state, result=result)


@pytest.mark.unit
@pytest.mark.gate
def test_handoff_rejects_tenant_scope_mismatch() -> None:
    decision_id = mint_decision_id()
    lineage = _execution_lineage()
    lifecycle_identity = _identity(
        decision_id=decision_id,
        tenant_id="tenant-a",
        namespace="incident",
        subject="incident-1",
        execution=lineage,
    )
    verification_identity = DecisionIdentity(
        decision_id=decision_id,
        version=lifecycle_identity.version,
        scope=DecisionScope(namespace="incident", subject="incident-2"),
        tenant_id="tenant-b",
        execution=lineage,
    )
    state = _verification_state(lifecycle_identity)
    result = _passed_result(_proposal_ref(identity=verification_identity))

    with pytest.raises(ValueError, match="requires lifecycle identity to match"):
        handoff_verification_result(state=state, result=result)


@pytest.mark.unit
@pytest.mark.gate
def test_handoff_rejects_execution_lineage_mismatch() -> None:
    decision_id = mint_decision_id()
    version = initial_decision_version()
    scope = DecisionScope(namespace="scope", subject="subject-1")
    lifecycle_identity = DecisionIdentity(
        decision_id=decision_id,
        version=version,
        scope=scope,
        tenant_id="tenant-a",
        execution=_execution_lineage(),
    )
    verification_identity = DecisionIdentity(
        decision_id=decision_id,
        version=version,
        scope=scope,
        tenant_id="tenant-a",
        execution=_execution_lineage(),
    )
    state = _verification_state(lifecycle_identity)
    result = _passed_result(_proposal_ref(identity=verification_identity))

    with pytest.raises(ValueError, match="requires lifecycle identity to match"):
        handoff_verification_result(state=state, result=result)


@pytest.mark.unit
@pytest.mark.gate
def test_sibling_branch_identity_preserved_without_lifecycle_branch_validation() -> None:
    identity = _identity(version=DecisionVersion(2))
    ref_a = _proposal_ref(identity=identity, version=DecisionVersion(2), branch_id="A")
    ref_b = _proposal_ref(identity=identity, version=DecisionVersion(2), branch_id="B")
    state = _verification_state(identity)
    result_a = _challenged_result(ref_a)

    handoff = handoff_verification_result(state=state, result=result_a)

    assert handoff.verification_result.proposal_ref.lineage_ref.branch_id == "A"
    assert ref_b.lineage_ref.branch_id == "B"
    assert ref_a.identity == ref_b.identity == state.identity
    handoff_for_b = handoff_verification_result(state=state, result=_passed_result(ref_b))
    assert handoff_for_b.verification_result.proposal_ref.lineage_ref.branch_id == "B"


@pytest.mark.unit
@pytest.mark.gate
def test_challenged_handoff_does_not_auto_transition_to_revision() -> None:
    identity = _identity()
    proposal_ref = _proposal_ref(identity=identity)
    state = _verification_state(identity, transition_index=2)

    handoff_verification_result(state=state, result=_challenged_result(proposal_ref))

    assert state.stage is DecisionLifecycleStage.VERIFICATION
    assert state.transition_index == 2
    assert state.identity is identity


@pytest.mark.unit
@pytest.mark.gate
def test_passed_handoff_does_not_auto_transition_to_resolution() -> None:
    identity = _identity()
    proposal_ref = _proposal_ref(identity=identity)
    state = _verification_state(identity, transition_index=2)

    handoff_verification_result(state=state, result=_passed_result(proposal_ref))

    assert state.stage is DecisionLifecycleStage.VERIFICATION
    assert state.transition_index == 2


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.parametrize(
    "to_stage",
    [
        DecisionLifecycleStage.REVISION,
        DecisionLifecycleStage.ADJUDICATION,
        DecisionLifecycleStage.RESOLUTION,
    ],
)
def test_handoff_does_not_remove_canonical_verification_exit_choices(
    to_stage: DecisionLifecycleStage,
) -> None:
    identity = _identity()
    proposal_ref = _proposal_ref(identity=identity)
    state = _verification_state(identity, transition_index=2)

    handoff_verification_result(state=state, result=_passed_result(proposal_ref))

    transitioned = transition_decision_lifecycle(state, to_stage)
    assert transitioned.stage is to_stage
    assert transitioned.transition_index == state.transition_index + 1


@pytest.mark.unit
@pytest.mark.gate
def test_decision_lifecycle_state_does_not_carry_branch_identity() -> None:
    source = _MODULE_PATH.read_text(encoding="utf-8")
    assert "DecisionLifecycleState`` binds ``DecisionIdentity`` only" in source
    field_names = frozenset(field.name for field in fields(DecisionLifecycleState))
    assert "branch_id" not in field_names
    assert "lineage_ref" not in field_names


@pytest.mark.unit
@pytest.mark.gate
def test_forbidden_production_patterns_absent() -> None:
    source = _MODULE_PATH.read_text(encoding="utf-8")
    hits = [
        pattern
        for pattern in _FORBIDDEN_PRODUCTION_PATTERNS
        if re.search(pattern, source)
    ]
    assert hits == [], f"forbidden production patterns found: {hits}"
