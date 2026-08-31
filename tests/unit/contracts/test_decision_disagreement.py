# © Artur Czarnecki. All rights reserved.

import dataclasses

import pytest

from intergrax.contracts.decision_disagreement import (
    DecisionDisagreementArtifact,
    DisagreementConflict,
    DisagreementPosition,
    UnresolvedQuestion,
    decision_disagreement_artifact,
    disagreement_conflict,
    disagreement_position,
    unresolved_question,
)
from intergrax.contracts.decision_identity import (
    DecisionExecutionLineage,
    DecisionIdentity,
    DecisionScope,
    DecisionVersion,
    initial_decision_version,
    mint_decision_id,
)
from intergrax.contracts.decision_record import (
    DecisionBranchId,
    DecisionProposalRef,
    decision_lineage_ref,
    decision_proposal_ref,
    validate_decision_branch_id,
)
from intergrax.contracts.evidence_claims import (
    EvidenceReferenceId,
    validate_evidence_reference_id,
)
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)

_FORBIDDEN_FIELD_FRAGMENTS = (
    "winner",
    "resolution",
    "majority",
    "adjudication",
    "hitl",
    "verification",
    "retry",
    "metadata",
    "payload",
)

_PRIVATE_COT_FRAGMENTS = (
    "chain_of_thought",
    "reasoning_trace",
    "scratchpad",
    "internal_reasoning",
)

_CONTRACT_TYPES = (
    DisagreementPosition,
    DisagreementConflict,
    UnresolvedQuestion,
    DecisionDisagreementArtifact,
)


def _execution_lineage() -> DecisionExecutionLineage:
    return DecisionExecutionLineage(
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )


def _identity(
    *,
    tenant_id: str = "tenant-a",
    namespace: str = "incident",
    subject: str = "incident-123",
    version: DecisionVersion | None = None,
) -> DecisionIdentity:
    return DecisionIdentity(
        decision_id=mint_decision_id(),
        version=version or initial_decision_version(),
        scope=DecisionScope(namespace=namespace, subject=subject),
        tenant_id=tenant_id,
        execution=_execution_lineage(),
    )


def _branch(branch_id: str) -> DecisionBranchId:
    return validate_decision_branch_id(branch_id)


def _proposal(
    identity: DecisionIdentity,
    *,
    version: int | None = None,
    branch_id: str = "analysis-a",
) -> DecisionProposalRef:
    resolved_version = DecisionVersion(version or identity.version.value)
    return decision_proposal_ref(
        identity=identity,
        lineage_ref=decision_lineage_ref(resolved_version, _branch(branch_id)),
    )


def _evidence(identifier: str) -> EvidenceReferenceId:
    return validate_evidence_reference_id(identifier)


def _position(
    identity: DecisionIdentity,
    *,
    version: int | None = None,
    branch_id: str = "analysis-a",
    summary: str = "Prefer escalation path A",
    evidence: tuple[EvidenceReferenceId, ...] = (),
) -> DisagreementPosition:
    return disagreement_position(
        proposal_ref=_proposal(identity, version=version, branch_id=branch_id),
        summary=summary,
        evidence_refs=evidence,
    )


def _conflict(
    identity: DecisionIdentity,
    *,
    dimension: str = "response strategy",
    proposal_refs: tuple[DecisionProposalRef, ...] | None = None,
    summary: str = "Branches disagree on escalation timing",
) -> DisagreementConflict:
    resolved_refs = proposal_refs or (
        _proposal(identity, branch_id="analysis-a"),
        _proposal(identity, branch_id="analysis-b"),
    )
    return disagreement_conflict(
        dimension=dimension,
        proposal_refs=resolved_refs,
        summary=summary,
    )


def _artifact(
    *,
    identity: DecisionIdentity | None = None,
    proposal_refs: tuple[DecisionProposalRef, ...] | None = None,
    positions: tuple[DisagreementPosition, ...] | None = None,
    conflicts: tuple[DisagreementConflict, ...] | None = None,
    unresolved_questions: tuple[UnresolvedQuestion, ...] = (),
) -> DecisionDisagreementArtifact:
    resolved_identity = identity or _identity()
    resolved_proposals = proposal_refs or (
        _proposal(resolved_identity, branch_id="analysis-a"),
        _proposal(resolved_identity, branch_id="analysis-b"),
    )
    resolved_positions = positions or (
        _position(resolved_identity, branch_id="analysis-a"),
        _position(
            resolved_identity,
            branch_id="analysis-b",
            summary="Prefer containment path B",
        ),
    )
    resolved_conflicts = conflicts or (_conflict(resolved_identity),)
    return decision_disagreement_artifact(
        proposal_refs=resolved_proposals,
        positions=resolved_positions,
        conflicts=resolved_conflicts,
        unresolved_questions=unresolved_questions,
    )


@pytest.mark.unit
@pytest.mark.gate
def test_valid_disagreement_between_two_refs() -> None:
    artifact = _artifact()
    assert len(artifact.proposal_refs) == 2
    assert len(artifact.positions) == 2
    assert len(artifact.conflicts) == 1


@pytest.mark.unit
@pytest.mark.gate
def test_sibling_same_version_different_branch_works() -> None:
    identity = _identity(version=DecisionVersion(2))
    artifact = _artifact(
        identity=identity,
        proposal_refs=(
            _proposal(identity, version=2, branch_id="analysis-a"),
            _proposal(identity, version=2, branch_id="analysis-b"),
        ),
        positions=(
            _position(identity, version=2, branch_id="analysis-a"),
            _position(
                identity,
                version=2,
                branch_id="analysis-b",
                summary="Hold for review",
            ),
        ),
        conflicts=(
            _conflict(
                identity,
                proposal_refs=(
                    _proposal(identity, version=2, branch_id="analysis-a"),
                    _proposal(identity, version=2, branch_id="analysis-b"),
                ),
            ),
        ),
    )
    assert artifact.proposal_refs[0].lineage_ref.version.value == 2
    assert artifact.proposal_refs[0].lineage_ref.branch_id == "analysis-a"
    assert artifact.proposal_refs[1].lineage_ref.branch_id == "analysis-b"


@pytest.mark.unit
@pytest.mark.gate
def test_duplicate_proposal_refs_rejected() -> None:
    identity = _identity()
    duplicate = _proposal(identity, branch_id="analysis-a")
    with pytest.raises(ValueError, match="duplicates"):
        _artifact(identity=identity, proposal_refs=(duplicate, duplicate))


@pytest.mark.unit
@pytest.mark.gate
def test_fewer_than_two_proposals_rejected() -> None:
    identity = _identity()
    with pytest.raises(ValueError, match="at least 2"):
        _artifact(
            identity=identity,
            proposal_refs=(_proposal(identity, branch_id="analysis-a"),),
        )


@pytest.mark.unit
@pytest.mark.gate
def test_position_unknown_proposal_rejected() -> None:
    identity = _identity()
    with pytest.raises(ValueError, match="known proposal refs"):
        _artifact(
            identity=identity,
            positions=(
                _position(identity, branch_id="analysis-a"),
                _position(
                    identity,
                    branch_id="unknown-branch",
                    summary="Unknown stance",
                ),
            ),
        )


@pytest.mark.unit
@pytest.mark.gate
def test_duplicate_position_for_same_proposal_rejected() -> None:
    identity = _identity()
    with pytest.raises(ValueError, match="duplicate proposal_ref"):
        _artifact(
            identity=identity,
            positions=(
                _position(identity, branch_id="analysis-a"),
                _position(
                    identity,
                    branch_id="analysis-a",
                    summary="Duplicate stance",
                ),
            ),
        )


@pytest.mark.unit
@pytest.mark.gate
def test_conflict_with_only_one_proposal_rejected() -> None:
    identity = _identity()
    with pytest.raises(ValueError, match="at least 2"):
        DisagreementConflict(
            dimension="scope",
            proposal_refs=(_proposal(identity, branch_id="analysis-a"),),
            summary="Only one side",
        )


@pytest.mark.unit
@pytest.mark.gate
def test_conflict_unknown_proposal_rejected() -> None:
    identity = _identity()
    with pytest.raises(ValueError, match="known proposal refs"):
        _artifact(
            identity=identity,
            conflicts=(
                _conflict(
                    identity,
                    proposal_refs=(
                        _proposal(identity, branch_id="analysis-a"),
                        _proposal(identity, branch_id="foreign-branch"),
                    ),
                ),
            ),
        )


@pytest.mark.unit
@pytest.mark.gate
def test_duplicate_refs_in_conflict_rejected() -> None:
    identity = _identity()
    duplicate = _proposal(identity, branch_id="analysis-a")
    with pytest.raises(ValueError, match="duplicates"):
        DisagreementConflict(
            dimension="scope",
            proposal_refs=(duplicate, duplicate),
            summary="Duplicate refs",
        )


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.parametrize("summary", ["", "   "])
def test_blank_position_summary_rejected(summary: str) -> None:
    identity = _identity()
    with pytest.raises(ValueError):
        DisagreementPosition(
            proposal_ref=_proposal(identity, branch_id="analysis-a"),
            summary=summary,
        )


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.parametrize("value", ["", "   "])
def test_blank_conflict_topic_or_summary_rejected(value: str) -> None:
    identity = _identity()
    with pytest.raises(ValueError):
        DisagreementConflict(
            dimension=value,
            proposal_refs=(
                _proposal(identity, branch_id="analysis-a"),
                _proposal(identity, branch_id="analysis-b"),
            ),
            summary="Valid summary",
        )
    with pytest.raises(ValueError):
        DisagreementConflict(
            dimension="Valid dimension",
            proposal_refs=(
                _proposal(identity, branch_id="analysis-a"),
                _proposal(identity, branch_id="analysis-b"),
            ),
            summary=value,
        )


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.parametrize("question", ["", "   "])
def test_blank_unresolved_question_rejected(question: str) -> None:
    with pytest.raises(ValueError):
        UnresolvedQuestion(question=question)


@pytest.mark.unit
@pytest.mark.gate
def test_unresolved_question_unknown_proposal_rejected() -> None:
    identity = _identity()
    with pytest.raises(ValueError, match="known proposal refs"):
        _artifact(
            identity=identity,
            unresolved_questions=(
                UnresolvedQuestion(
                    question="Which evidence is authoritative?",
                    related_proposal_refs=(
                        _proposal(identity, branch_id="foreign-branch"),
                    ),
                ),
            ),
        )


@pytest.mark.unit
@pytest.mark.gate
def test_duplicate_unresolved_question_rejected() -> None:
    identity = _identity()
    with pytest.raises(ValueError, match="duplicate question"):
        _artifact(
            identity=identity,
            unresolved_questions=(
                UnresolvedQuestion(question="Need more logs?"),
                UnresolvedQuestion(question="Need more logs?"),
            ),
        )


@pytest.mark.unit
@pytest.mark.gate
def test_duplicate_related_proposal_refs_in_question_rejected() -> None:
    identity = _identity()
    duplicate = _proposal(identity, branch_id="analysis-a")
    with pytest.raises(ValueError, match="duplicates"):
        UnresolvedQuestion(
            question="Which branch is better supported?",
            related_proposal_refs=(duplicate, duplicate),
        )


@pytest.mark.unit
@pytest.mark.gate
def test_evidence_duplicate_rejected_on_position() -> None:
    identity = _identity()
    evidence = _evidence("trace.record.alpha")
    with pytest.raises(ValueError, match="duplicates"):
        DisagreementPosition(
            proposal_ref=_proposal(identity, branch_id="analysis-a"),
            summary="Evidence-backed stance",
            evidence_refs=(evidence, evidence),
        )


@pytest.mark.unit
@pytest.mark.gate
def test_deterministic_ordering() -> None:
    identity = _identity(version=DecisionVersion(2))
    artifact = decision_disagreement_artifact(
        proposal_refs=(
            _proposal(identity, version=2, branch_id="analysis-b"),
            _proposal(identity, version=2, branch_id="analysis-a"),
        ),
        positions=(
            _position(
                identity,
                version=2,
                branch_id="analysis-b",
                summary="B stance",
            ),
            _position(
                identity,
                version=2,
                branch_id="analysis-a",
                summary="A stance",
            ),
        ),
        conflicts=(
            _conflict(
                identity,
                proposal_refs=(
                    _proposal(identity, version=2, branch_id="analysis-b"),
                    _proposal(identity, version=2, branch_id="analysis-a"),
                ),
            ),
        ),
    )
    assert [ref.lineage_ref.branch_id for ref in artifact.proposal_refs] == [
        "analysis-a",
        "analysis-b",
    ]
    assert [
        position.proposal_ref.lineage_ref.branch_id for position in artifact.positions
    ] == [
        "analysis-a",
        "analysis-b",
    ]
    position = disagreement_position(
        proposal_ref=_proposal(identity, branch_id="analysis-a"),
        summary="Ordered evidence",
        evidence_refs=(_evidence("z.ref"), _evidence("a.ref")),
    )
    assert [str(item) for item in position.evidence_refs] == ["a.ref", "z.ref"]


@pytest.mark.unit
@pytest.mark.gate
def test_direct_constructor_protects_invariants() -> None:
    identity = _identity()
    with pytest.raises(ValueError):
        DecisionDisagreementArtifact(
            proposal_refs=(_proposal(identity, branch_id="analysis-a"),),
            positions=(_position(identity, branch_id="analysis-a"),),
            conflicts=(_conflict(identity),),
        )


@pytest.mark.unit
@pytest.mark.gate
def test_artifact_immutable_value_semantics() -> None:
    artifact = _artifact()
    assert dataclasses.is_dataclass(artifact)
    replaced = dataclasses.replace(artifact)
    assert replaced == artifact
    assert replaced is not artifact


@pytest.mark.unit
@pytest.mark.gate
def test_structural_field_audit_forbidden_architecture_concepts() -> None:
    for contract_type in _CONTRACT_TYPES:
        field_names = {field.name for field in dataclasses.fields(contract_type)}
        for field_name in field_names:
            lowered = field_name.lower()
            for fragment in _FORBIDDEN_FIELD_FRAGMENTS:
                assert fragment not in lowered, (
                    f"{contract_type.__name__}.{field_name} contains forbidden "
                    f"fragment {fragment!r}"
                )
            for fragment in _PRIVATE_COT_FRAGMENTS:
                assert fragment not in lowered, (
                    f"{contract_type.__name__}.{field_name} contains private CoT "
                    f"fragment {fragment!r}"
                )


@pytest.mark.unit
@pytest.mark.gate
def test_unresolved_question_with_valid_refs_and_evidence() -> None:
    identity = _identity()
    artifact = _artifact(
        identity=identity,
        unresolved_questions=(
            unresolved_question(
                question="Is latency evidence sufficient?",
                related_proposal_refs=(
                    _proposal(identity, branch_id="analysis-b"),
                    _proposal(identity, branch_id="analysis-a"),
                ),
                evidence_refs=(_evidence("metrics.latency"),),
            ),
        ),
    )
    assert artifact.unresolved_questions[0].question.startswith("Is latency")
    assert [
        ref.lineage_ref.branch_id
        for ref in artifact.unresolved_questions[0].related_proposal_refs
    ] == [
        "analysis-a",
        "analysis-b",
    ]


@pytest.mark.unit
@pytest.mark.gate
def test_different_decision_id_rejected() -> None:
    identity_a = _identity()
    identity_b = _identity()
    with pytest.raises(ValueError, match="identity boundary"):
        decision_disagreement_artifact(
            proposal_refs=(
                _proposal(identity_a, branch_id="analysis-a"),
                _proposal(identity_b, branch_id="analysis-b"),
            ),
            positions=(
                _position(identity_a, branch_id="analysis-a"),
                _position(identity_b, branch_id="analysis-b", summary="Foreign"),
            ),
            conflicts=(_conflict(identity_a),),
        )


@pytest.mark.unit
@pytest.mark.gate
def test_different_tenant_rejected() -> None:
    identity_a = _identity(tenant_id="tenant-a")
    identity_b = _identity(tenant_id="tenant-b")
    with pytest.raises(ValueError, match="identity boundary"):
        decision_disagreement_artifact(
            proposal_refs=(
                _proposal(identity_a, branch_id="analysis-a"),
                _proposal(identity_b, branch_id="analysis-b"),
            ),
            positions=(
                _position(identity_a, branch_id="analysis-a"),
                _position(identity_b, branch_id="analysis-b", summary="Foreign"),
            ),
            conflicts=(_conflict(identity_a),),
        )


@pytest.mark.unit
@pytest.mark.gate
def test_different_scope_rejected() -> None:
    identity_a = _identity(namespace="incident", subject="incident-123")
    identity_b = _identity(namespace="incident", subject="incident-999")
    with pytest.raises(ValueError, match="identity boundary"):
        decision_disagreement_artifact(
            proposal_refs=(
                _proposal(identity_a, branch_id="analysis-a"),
                _proposal(identity_b, branch_id="analysis-b"),
            ),
            positions=(
                _position(identity_a, branch_id="analysis-a"),
                _position(identity_b, branch_id="analysis-b", summary="Foreign"),
            ),
            conflicts=(_conflict(identity_a),),
        )


@pytest.mark.unit
@pytest.mark.gate
def test_same_lineage_different_decision_distinct() -> None:
    identity_a = _identity()
    identity_b = _identity()
    ref_a = _proposal(identity_a, branch_id="analysis-a")
    ref_b = _proposal(identity_b, branch_id="analysis-a")
    assert ref_a.lineage_ref == ref_b.lineage_ref
    assert ref_a != ref_b


@pytest.mark.unit
@pytest.mark.gate
def test_position_from_foreign_decision_rejected() -> None:
    identity_a = _identity()
    identity_b = _identity()
    with pytest.raises(ValueError, match="identity boundary"):
        decision_disagreement_artifact(
            proposal_refs=(
                _proposal(identity_a, branch_id="analysis-a"),
                _proposal(identity_a, branch_id="analysis-b"),
            ),
            positions=(
                _position(identity_a, branch_id="analysis-a"),
                _position(identity_b, branch_id="analysis-b", summary="Foreign"),
            ),
            conflicts=(_conflict(identity_a),),
        )


@pytest.mark.unit
@pytest.mark.gate
def test_conflict_with_foreign_decision_rejected() -> None:
    identity_a = _identity()
    identity_b = _identity()
    with pytest.raises(ValueError, match="identity boundary"):
        decision_disagreement_artifact(
            proposal_refs=(
                _proposal(identity_a, branch_id="analysis-a"),
                _proposal(identity_a, branch_id="analysis-b"),
            ),
            positions=(
                _position(identity_a, branch_id="analysis-a"),
                _position(
                    identity_a,
                    branch_id="analysis-b",
                    summary="Local stance",
                ),
            ),
            conflicts=(
                _conflict(
                    identity_a,
                    proposal_refs=(
                        _proposal(identity_a, branch_id="analysis-a"),
                        _proposal(identity_b, branch_id="analysis-b"),
                    ),
                ),
            ),
        )


@pytest.mark.unit
@pytest.mark.gate
def test_unresolved_question_foreign_decision_rejected() -> None:
    identity_a = _identity()
    identity_b = _identity()
    with pytest.raises(ValueError, match="identity boundary"):
        decision_disagreement_artifact(
            proposal_refs=(
                _proposal(identity_a, branch_id="analysis-a"),
                _proposal(identity_a, branch_id="analysis-b"),
            ),
            positions=(
                _position(identity_a, branch_id="analysis-a"),
                _position(
                    identity_a,
                    branch_id="analysis-b",
                    summary="Local stance",
                ),
            ),
            conflicts=(_conflict(identity_a),),
            unresolved_questions=(
                UnresolvedQuestion(
                    question="Which evidence applies?",
                    related_proposal_refs=(
                        _proposal(identity_b, branch_id="analysis-a"),
                    ),
                ),
            ),
        )


@pytest.mark.unit
@pytest.mark.gate
def test_direct_constructor_rejects_cross_decision_mixture() -> None:
    identity_a = _identity()
    identity_b = _identity()
    with pytest.raises(ValueError, match="identity boundary"):
        DecisionDisagreementArtifact(
            proposal_refs=(
                _proposal(identity_a, branch_id="analysis-a"),
                _proposal(identity_b, branch_id="analysis-b"),
            ),
            positions=(
                _position(identity_a, branch_id="analysis-a"),
                _position(identity_b, branch_id="analysis-b", summary="Foreign"),
            ),
            conflicts=(_conflict(identity_a),),
        )
