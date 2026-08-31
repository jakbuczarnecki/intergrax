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
from intergrax.contracts.decision_identity import DecisionVersion
from intergrax.contracts.decision_record import (
    DecisionBranchId,
    DecisionLineageRef,
    decision_lineage_ref,
    validate_decision_branch_id,
)
from intergrax.contracts.evidence_claims import (
    EvidenceReferenceId,
    validate_evidence_reference_id,
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


def _branch(branch_id: str) -> DecisionBranchId:
    return validate_decision_branch_id(branch_id)


def _ref(version: int, branch_id: str) -> DecisionLineageRef:
    return decision_lineage_ref(DecisionVersion(version), _branch(branch_id))


def _evidence(identifier: str) -> EvidenceReferenceId:
    return validate_evidence_reference_id(identifier)


def _position(
    *,
    version: int = 1,
    branch_id: str = "analysis-a",
    summary: str = "Prefer escalation path A",
    evidence: tuple[EvidenceReferenceId, ...] = (),
) -> DisagreementPosition:
    return disagreement_position(
        proposal_ref=_ref(version, branch_id),
        summary=summary,
        evidence_refs=evidence,
    )


def _conflict(
    *,
    dimension: str = "response strategy",
    proposal_refs: tuple[DecisionLineageRef, ...] | None = None,
    summary: str = "Branches disagree on escalation timing",
) -> DisagreementConflict:
    resolved_refs = proposal_refs or (_ref(1, "analysis-a"), _ref(1, "analysis-b"))
    return disagreement_conflict(
        dimension=dimension,
        proposal_refs=resolved_refs,
        summary=summary,
    )


def _artifact(
    *,
    proposal_refs: tuple[DecisionLineageRef, ...] | None = None,
    positions: tuple[DisagreementPosition, ...] | None = None,
    conflicts: tuple[DisagreementConflict, ...] | None = None,
    unresolved_questions: tuple[UnresolvedQuestion, ...] = (),
) -> DecisionDisagreementArtifact:
    resolved_proposals = proposal_refs or (
        _ref(1, "analysis-a"),
        _ref(1, "analysis-b"),
    )
    resolved_positions = positions or (
        _position(branch_id="analysis-a"),
        _position(branch_id="analysis-b", summary="Prefer containment path B"),
    )
    resolved_conflicts = conflicts or (_conflict(),)
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
    artifact = _artifact(
        proposal_refs=(_ref(2, "analysis-a"), _ref(2, "analysis-b")),
        positions=(
            _position(version=2, branch_id="analysis-a"),
            _position(version=2, branch_id="analysis-b", summary="Hold for review"),
        ),
        conflicts=(
            _conflict(
                proposal_refs=(_ref(2, "analysis-a"), _ref(2, "analysis-b")),
            ),
        ),
    )
    assert artifact.proposal_refs[0].version.value == 2
    assert artifact.proposal_refs[0].branch_id == "analysis-a"
    assert artifact.proposal_refs[1].branch_id == "analysis-b"


@pytest.mark.unit
@pytest.mark.gate
def test_duplicate_proposal_refs_rejected() -> None:
    duplicate = _ref(1, "analysis-a")
    with pytest.raises(ValueError, match="duplicates"):
        _artifact(proposal_refs=(duplicate, duplicate))


@pytest.mark.unit
@pytest.mark.gate
def test_fewer_than_two_proposals_rejected() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        _artifact(proposal_refs=(_ref(1, "analysis-a"),))


@pytest.mark.unit
@pytest.mark.gate
def test_position_unknown_proposal_rejected() -> None:
    with pytest.raises(ValueError, match="known proposal refs"):
        _artifact(
            positions=(
                _position(branch_id="analysis-a"),
                _position(branch_id="unknown-branch", summary="Unknown stance"),
            ),
        )


@pytest.mark.unit
@pytest.mark.gate
def test_duplicate_position_for_same_proposal_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate proposal_ref"):
        _artifact(
            positions=(
                _position(branch_id="analysis-a"),
                _position(branch_id="analysis-a", summary="Duplicate stance"),
            ),
        )


@pytest.mark.unit
@pytest.mark.gate
def test_conflict_with_only_one_proposal_rejected() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        DisagreementConflict(
            dimension="scope",
            proposal_refs=(_ref(1, "analysis-a"),),
            summary="Only one side",
        )


@pytest.mark.unit
@pytest.mark.gate
def test_conflict_unknown_proposal_rejected() -> None:
    with pytest.raises(ValueError, match="known proposal refs"):
        _artifact(
            conflicts=(
                _conflict(
                    proposal_refs=(_ref(1, "analysis-a"), _ref(1, "foreign-branch")),
                ),
            ),
        )


@pytest.mark.unit
@pytest.mark.gate
def test_duplicate_refs_in_conflict_rejected() -> None:
    duplicate = _ref(1, "analysis-a")
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
    with pytest.raises(ValueError):
        DisagreementPosition(
            proposal_ref=_ref(1, "analysis-a"),
            summary=summary,
        )


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.parametrize("value", ["", "   "])
def test_blank_conflict_topic_or_summary_rejected(value: str) -> None:
    with pytest.raises(ValueError):
        DisagreementConflict(
            dimension=value,
            proposal_refs=(_ref(1, "analysis-a"), _ref(1, "analysis-b")),
            summary="Valid summary",
        )
    with pytest.raises(ValueError):
        DisagreementConflict(
            dimension="Valid dimension",
            proposal_refs=(_ref(1, "analysis-a"), _ref(1, "analysis-b")),
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
    with pytest.raises(ValueError, match="known proposal refs"):
        _artifact(
            unresolved_questions=(
                UnresolvedQuestion(
                    question="Which evidence is authoritative?",
                    related_proposal_refs=(_ref(1, "foreign-branch"),),
                ),
            ),
        )


@pytest.mark.unit
@pytest.mark.gate
def test_duplicate_unresolved_question_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate question"):
        _artifact(
            unresolved_questions=(
                UnresolvedQuestion(question="Need more logs?"),
                UnresolvedQuestion(question="Need more logs?"),
            ),
        )


@pytest.mark.unit
@pytest.mark.gate
def test_duplicate_related_proposal_refs_in_question_rejected() -> None:
    duplicate = _ref(1, "analysis-a")
    with pytest.raises(ValueError, match="duplicates"):
        UnresolvedQuestion(
            question="Which branch is better supported?",
            related_proposal_refs=(duplicate, duplicate),
        )


@pytest.mark.unit
@pytest.mark.gate
def test_evidence_duplicate_rejected_on_position() -> None:
    evidence = _evidence("trace.record.alpha")
    with pytest.raises(ValueError, match="duplicates"):
        DisagreementPosition(
            proposal_ref=_ref(1, "analysis-a"),
            summary="Evidence-backed stance",
            evidence_refs=(evidence, evidence),
        )


@pytest.mark.unit
@pytest.mark.gate
def test_deterministic_ordering() -> None:
    artifact = decision_disagreement_artifact(
        proposal_refs=(
            _ref(2, "analysis-b"),
            _ref(2, "analysis-a"),
        ),
        positions=(
            _position(version=2, branch_id="analysis-b", summary="B stance"),
            _position(version=2, branch_id="analysis-a", summary="A stance"),
        ),
        conflicts=(
            _conflict(
                proposal_refs=(_ref(2, "analysis-b"), _ref(2, "analysis-a")),
            ),
        ),
    )
    assert [ref.branch_id for ref in artifact.proposal_refs] == [
        "analysis-a",
        "analysis-b",
    ]
    assert [position.proposal_ref.branch_id for position in artifact.positions] == [
        "analysis-a",
        "analysis-b",
    ]
    position = disagreement_position(
        proposal_ref=_ref(1, "analysis-a"),
        summary="Ordered evidence",
        evidence_refs=(_evidence("z.ref"), _evidence("a.ref")),
    )
    assert [str(item) for item in position.evidence_refs] == ["a.ref", "z.ref"]


@pytest.mark.unit
@pytest.mark.gate
def test_direct_constructor_protects_invariants() -> None:
    with pytest.raises(ValueError):
        DecisionDisagreementArtifact(
            proposal_refs=(_ref(1, "analysis-a"),),
            positions=(_position(branch_id="analysis-a"),),
            conflicts=(_conflict(),),
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
    artifact = _artifact(
            unresolved_questions=(
                unresolved_question(
                    question="Is latency evidence sufficient?",
                    related_proposal_refs=(_ref(1, "analysis-b"), _ref(1, "analysis-a")),
                    evidence_refs=(_evidence("metrics.latency"),),
                ),
            ),
    )
    assert artifact.unresolved_questions[0].question.startswith("Is latency")
    assert [ref.branch_id for ref in artifact.unresolved_questions[0].related_proposal_refs] == [
        "analysis-a",
        "analysis-b",
    ]
