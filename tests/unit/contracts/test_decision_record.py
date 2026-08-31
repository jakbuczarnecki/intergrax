# © Artur Czarnecki. All rights reserved.

from dataclasses import dataclass, replace

import pytest

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
    DecisionArtifactKind,
    DecisionBranchId,
    DecisionLineageRef,
    DecisionProposalRef,
    DecisionVersionLineage,
    candidate_decision_ref,
    decision_lineage_ref,
    decision_proposal_ref,
    decision_proposal_ref_sort_key,
    decision_version_lineage,
    initial_decision_branch_id,
    validate_decision_artifact_kind,
    validate_decision_branch_id,
)
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)


@dataclass(frozen=True, slots=True)
class IncidentDecisionPayload:
    recommendation: str


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


def _artifact(
    *,
    kind: str = "incident_resolution",
    recommendation: str = "escalate",
) -> DecisionArtifact[IncidentDecisionPayload]:
    return DecisionArtifact(
        kind=validate_decision_artifact_kind(kind),
        content=IncidentDecisionPayload(recommendation=recommendation),
    )


def _root_ref() -> DecisionLineageRef:
    return decision_lineage_ref(initial_decision_version())


def _root_lineage() -> DecisionVersionLineage:
    return DecisionVersionLineage(current=_root_ref())


def _linear_lineage(
    current: DecisionVersion,
    parent: DecisionLineageRef,
    *,
    branch_id: DecisionBranchId | None = None,
) -> DecisionVersionLineage:
    return DecisionVersionLineage(
        current=decision_lineage_ref(current, branch_id),
        parents=(parent,),
    )


def _candidate(
    *,
    identity: DecisionIdentity | None = None,
    artifact: DecisionArtifact[IncidentDecisionPayload] | None = None,
    lineage: DecisionVersionLineage | None = None,
) -> CandidateDecision[IncidentDecisionPayload]:
    resolved_identity = identity or _identity()
    resolved_lineage = lineage or DecisionVersionLineage(
        current=decision_lineage_ref(resolved_identity.version),
    )
    return CandidateDecision(
        identity=resolved_identity,
        artifact=artifact or _artifact(),
        lineage=resolved_lineage,
    )


@pytest.mark.unit
@pytest.mark.gate
def test_decision_branch_id_valid() -> None:
    branch_id = validate_decision_branch_id("branch-A")
    assert branch_id == "branch-A"


@pytest.mark.unit
@pytest.mark.gate
def test_decision_branch_id_rejects_wrong_type() -> None:
    with pytest.raises(TypeError):
        validate_decision_branch_id(42)


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.parametrize("branch_id", ["", "   "])
def test_decision_branch_id_rejects_blank(branch_id: str) -> None:
    with pytest.raises(ValueError):
        validate_decision_branch_id(branch_id)


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.parametrize("branch_id", [" branch-A", "branch-A "])
def test_decision_branch_id_rejects_surrounding_whitespace(branch_id: str) -> None:
    with pytest.raises(ValueError):
        validate_decision_branch_id(branch_id)


@pytest.mark.unit
@pytest.mark.gate
def test_initial_decision_branch_id_is_deterministic_main() -> None:
    assert initial_decision_branch_id() == "main"
    assert initial_decision_branch_id() == initial_decision_branch_id()


@pytest.mark.unit
@pytest.mark.gate
def test_decision_lineage_ref_defaults_branch_to_main() -> None:
    ref = decision_lineage_ref(initial_decision_version())
    assert ref.branch_id == initial_decision_branch_id()


@pytest.mark.unit
@pytest.mark.gate
def test_decision_lineage_ref_rejects_explicit_empty_branch() -> None:
    with pytest.raises(ValueError):
        decision_lineage_ref(initial_decision_version(), DecisionBranchId(""))


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.parametrize("branch_id", ["   ", " branch "])
def test_decision_lineage_ref_rejects_explicit_whitespace_branch(
    branch_id: str,
) -> None:
    with pytest.raises(ValueError):
        decision_lineage_ref(initial_decision_version(), DecisionBranchId(branch_id))


@pytest.mark.unit
@pytest.mark.gate
def test_decision_artifact_accepts_valid_typed_payload() -> None:
    artifact = _artifact(recommendation="contain")
    assert artifact.content.recommendation == "contain"
    assert artifact.kind == "incident_resolution"


@pytest.mark.unit
@pytest.mark.gate
def test_decision_artifact_kind_valid() -> None:
    kind = validate_decision_artifact_kind("contract_review")
    assert kind == "contract_review"


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.parametrize("kind", ["", "   "])
def test_decision_artifact_kind_rejects_empty(kind: str) -> None:
    with pytest.raises(ValueError):
        validate_decision_artifact_kind(kind)


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.parametrize("kind", [" incident_resolution", "incident_resolution "])
def test_decision_artifact_kind_rejects_whitespace(kind: str) -> None:
    with pytest.raises(ValueError):
        validate_decision_artifact_kind(kind)


@pytest.mark.unit
@pytest.mark.gate
def test_decision_artifact_is_immutable() -> None:
    artifact = _artifact()
    with pytest.raises(AttributeError):
        artifact.kind = validate_decision_artifact_kind("other")


@pytest.mark.unit
@pytest.mark.gate
def test_decision_artifact_kind_not_silently_normalized() -> None:
    with pytest.raises(ValueError):
        validate_decision_artifact_kind(" incident_resolution")


@pytest.mark.unit
@pytest.mark.gate
def test_lineage_v1_main_root_valid() -> None:
    lineage = _root_lineage()
    assert lineage.current.version.value == 1
    assert lineage.current.branch_id == "main"
    assert lineage.parents == ()


@pytest.mark.unit
@pytest.mark.gate
def test_lineage_non_v1_without_parents_rejected() -> None:
    with pytest.raises(ValueError, match="without parents requires current version 1"):
        DecisionVersionLineage(current=decision_lineage_ref(DecisionVersion(2)))


@pytest.mark.unit
@pytest.mark.gate
def test_lineage_v1_with_parents_rejected() -> None:
    with pytest.raises(ValueError, match="requires current version > 1"):
        DecisionVersionLineage(
            current=_root_ref(),
            parents=(decision_lineage_ref(DecisionVersion(1), validate_decision_branch_id("A")),),
        )


@pytest.mark.unit
@pytest.mark.gate
def test_lineage_v2_main_parent_v1_main_valid() -> None:
    parent = _root_ref()
    lineage = _linear_lineage(DecisionVersion(2), parent)
    assert lineage.current.version.value == 2
    assert lineage.current.branch_id == "main"
    assert lineage.parents == (parent,)


@pytest.mark.unit
@pytest.mark.gate
def test_lineage_v3_main_parent_v2_main_valid() -> None:
    parent = decision_lineage_ref(DecisionVersion(2))
    lineage = _linear_lineage(DecisionVersion(3), parent)
    assert lineage.current.version.value == 3
    assert lineage.parents == (parent,)


@pytest.mark.unit
@pytest.mark.gate
def test_lineage_parallel_sibling_refs_are_distinct() -> None:
    root = _root_ref()
    branch_a = validate_decision_branch_id("A")
    branch_b = validate_decision_branch_id("B")
    lineage_a = _linear_lineage(DecisionVersion(2), root, branch_id=branch_a)
    lineage_b = _linear_lineage(DecisionVersion(2), root, branch_id=branch_b)

    assert lineage_a.current.version == lineage_b.current.version
    assert lineage_a.current != lineage_b.current
    assert lineage_a.parents == lineage_b.parents == (root,)


@pytest.mark.unit
@pytest.mark.gate
def test_lineage_two_sibling_candidates_share_version_distinct_branch() -> None:
    root = _root_ref()
    branch_a = validate_decision_branch_id("A")
    branch_b = validate_decision_branch_id("B")
    identity_a = _identity(version=DecisionVersion(2))
    identity_b = _identity(version=DecisionVersion(2))
    candidate_a = _candidate(
        identity=identity_a,
        lineage=_linear_lineage(DecisionVersion(2), root, branch_id=branch_a),
    )
    candidate_b = _candidate(
        identity=identity_b,
        lineage=_linear_lineage(DecisionVersion(2), root, branch_id=branch_b),
    )

    assert candidate_a.identity.version == candidate_b.identity.version
    assert candidate_a.lineage.current != candidate_b.lineage.current
    assert candidate_a.lineage.parents == candidate_b.lineage.parents


@pytest.mark.unit
@pytest.mark.gate
def test_lineage_synthesis_multi_parent_valid() -> None:
    root = _root_ref()
    v2a = decision_lineage_ref(DecisionVersion(2), validate_decision_branch_id("A"))
    v2b = decision_lineage_ref(DecisionVersion(2), validate_decision_branch_id("B"))
    _linear_lineage(DecisionVersion(2), root, branch_id=validate_decision_branch_id("A"))
    _linear_lineage(DecisionVersion(2), root, branch_id=validate_decision_branch_id("B"))
    synthesis = decision_version_lineage(
        current=decision_lineage_ref(DecisionVersion(3)),
        parents=(v2a, v2b),
    )

    assert synthesis.current.version.value == 3
    assert synthesis.current.branch_id == "main"
    assert synthesis.parents == (v2a, v2b)


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.parametrize(
    ("current", "parent_version"),
    [
        (2, 2),
        (2, 3),
        (3, 5),
    ],
)
def test_lineage_rejects_parent_not_earlier_than_current(
    current: int,
    parent_version: int,
) -> None:
    parent = decision_lineage_ref(
        DecisionVersion(parent_version),
        validate_decision_branch_id("parent"),
    )
    with pytest.raises(ValueError):
        DecisionVersionLineage(
            current=decision_lineage_ref(
                DecisionVersion(current),
                validate_decision_branch_id("current"),
            ),
            parents=(parent,),
        )


@pytest.mark.unit
@pytest.mark.gate
def test_lineage_rejects_duplicate_parent_ref() -> None:
    parent = decision_lineage_ref(DecisionVersion(2), validate_decision_branch_id("A"))
    with pytest.raises(ValueError, match="duplicates"):
        DecisionVersionLineage(
            current=decision_lineage_ref(DecisionVersion(3)),
            parents=(parent, parent),
        )


@pytest.mark.unit
@pytest.mark.gate
def test_lineage_rejects_current_as_parent() -> None:
    current = decision_lineage_ref(DecisionVersion(2), validate_decision_branch_id("A"))
    with pytest.raises(ValueError, match="cannot appear in parents"):
        DecisionVersionLineage(current=current, parents=(current,))


@pytest.mark.unit
@pytest.mark.gate
def test_lineage_is_immutable() -> None:
    lineage = _root_lineage()
    with pytest.raises(AttributeError):
        lineage.current = decision_lineage_ref(DecisionVersion(2))


@pytest.mark.unit
@pytest.mark.gate
def test_candidate_valid_accepted() -> None:
    candidate = _candidate()
    assert candidate.identity.tenant_id == "tenant-a"
    assert candidate.artifact.content.recommendation == "escalate"


@pytest.mark.unit
@pytest.mark.gate
def test_candidate_requires_identity_version_matches_lineage_current() -> None:
    identity = _identity(version=DecisionVersion(3))
    lineage = _linear_lineage(DecisionVersion(2), _root_ref())
    with pytest.raises(ValueError, match="current.version"):
        _candidate(identity=identity, lineage=lineage)


@pytest.mark.unit
@pytest.mark.gate
def test_candidate_is_immutable() -> None:
    candidate = _candidate()
    with pytest.raises(AttributeError):
        candidate.identity = _identity()


@pytest.mark.unit
@pytest.mark.gate
def test_candidate_preserves_artifact() -> None:
    artifact = _artifact(recommendation="isolate")
    candidate = _candidate(artifact=artifact)
    assert candidate.artifact is artifact
    assert candidate.artifact.content.recommendation == "isolate"


@pytest.mark.unit
@pytest.mark.gate
def test_candidate_equality_semantics() -> None:
    identity = _identity()
    artifact = _artifact()
    lineage = DecisionVersionLineage(current=decision_lineage_ref(identity.version))
    first = CandidateDecision(identity=identity, artifact=artifact, lineage=lineage)
    second = CandidateDecision(identity=identity, artifact=artifact, lineage=lineage)
    assert first == second

    other_version = _candidate(
        identity=_identity(version=DecisionVersion(2)),
        lineage=_linear_lineage(DecisionVersion(2), _root_ref()),
    )
    other_tenant = _candidate(identity=_identity(tenant_id="tenant-b"))
    other_scope = _candidate(
        identity=_identity(namespace="routing", subject="route-1"),
    )
    other_artifact = _candidate(artifact=_artifact(recommendation="rollback"))
    assert first != other_version
    assert first != other_tenant
    assert first != other_scope
    assert first != other_artifact


@pytest.mark.unit
@pytest.mark.gate
def test_authoritative_accepted_valid_contract() -> None:
    identity = _identity()
    artifact = _artifact()
    lineage = DecisionVersionLineage(current=decision_lineage_ref(identity.version))
    accepted = AuthoritativeAcceptedDecision(
        identity=identity,
        artifact=artifact,
        lineage=lineage,
    )
    assert accepted.identity is identity
    assert accepted.artifact is artifact
    assert accepted.lineage is lineage


@pytest.mark.unit
@pytest.mark.gate
def test_authoritative_accepted_preserves_multi_parent_lineage() -> None:
    v2a = decision_lineage_ref(DecisionVersion(2), validate_decision_branch_id("A"))
    v2b = decision_lineage_ref(DecisionVersion(2), validate_decision_branch_id("B"))
    lineage = decision_version_lineage(
        current=decision_lineage_ref(DecisionVersion(3)),
        parents=(v2a, v2b),
    )
    identity = _identity(version=DecisionVersion(3))
    accepted = AuthoritativeAcceptedDecision(
        identity=identity,
        artifact=_artifact(),
        lineage=lineage,
    )
    assert accepted.lineage.parents == (v2a, v2b)


@pytest.mark.unit
@pytest.mark.gate
def test_authoritative_accepted_enforces_identity_lineage_consistency() -> None:
    identity = _identity(version=DecisionVersion(2))
    lineage = _root_lineage()
    with pytest.raises(ValueError, match="current.version"):
        AuthoritativeAcceptedDecision(
            identity=identity,
            artifact=_artifact(),
            lineage=lineage,
        )


@pytest.mark.unit
@pytest.mark.gate
def test_authoritative_accepted_is_immutable() -> None:
    identity = _identity()
    accepted = AuthoritativeAcceptedDecision(
        identity=identity,
        artifact=_artifact(),
        lineage=DecisionVersionLineage(current=decision_lineage_ref(identity.version)),
    )
    with pytest.raises(AttributeError):
        accepted.artifact = _artifact(recommendation="other")


@pytest.mark.unit
@pytest.mark.gate
def test_authoritative_accepted_retains_artifact() -> None:
    artifact = _artifact(recommendation="patch")
    identity = _identity()
    accepted = AuthoritativeAcceptedDecision(
        identity=identity,
        artifact=artifact,
        lineage=DecisionVersionLineage(current=decision_lineage_ref(identity.version)),
    )
    assert accepted.artifact.content.recommendation == "patch"


@pytest.mark.unit
@pytest.mark.gate
def test_authoritative_accepted_differs_by_version_tenant_scope() -> None:
    base_identity = _identity()
    base = AuthoritativeAcceptedDecision(
        identity=base_identity,
        artifact=_artifact(),
        lineage=DecisionVersionLineage(
            current=decision_lineage_ref(base_identity.version),
        ),
    )
    other_version = AuthoritativeAcceptedDecision(
        identity=_identity(version=DecisionVersion(2)),
        artifact=_artifact(),
        lineage=_linear_lineage(DecisionVersion(2), _root_ref()),
    )
    other_tenant = AuthoritativeAcceptedDecision(
        identity=_identity(tenant_id="tenant-b"),
        artifact=_artifact(),
        lineage=_root_lineage(),
    )
    other_scope = AuthoritativeAcceptedDecision(
        identity=_identity(namespace="routing", subject="route-9"),
        artifact=_artifact(),
        lineage=_root_lineage(),
    )
    assert base != other_version
    assert base != other_tenant
    assert base != other_scope


@pytest.mark.unit
@pytest.mark.gate
def test_adversarial_identity_v2_lineage_current_v3_rejected() -> None:
    identity = _identity(version=DecisionVersion(2))
    lineage = _linear_lineage(
        DecisionVersion(3),
        decision_lineage_ref(DecisionVersion(2)),
    )
    with pytest.raises(ValueError, match="current.version"):
        CandidateDecision(
            identity=identity,
            artifact=_artifact(),
            lineage=lineage,
        )


@pytest.mark.unit
@pytest.mark.gate
def test_adversarial_lineage_escalation_parent_v99_rejected() -> None:
    with pytest.raises(ValueError):
        DecisionVersionLineage(
            current=decision_lineage_ref(DecisionVersion(2)),
            parents=(decision_lineage_ref(DecisionVersion(99)),),
        )


@pytest.mark.unit
@pytest.mark.gate
def test_adversarial_tenant_substitution_records_not_equal() -> None:
    trusted = _candidate(identity=_identity(tenant_id="tenant-trusted"))
    substituted = _candidate(identity=_identity(tenant_id="tenant-attacker"))
    assert trusted != substituted


@pytest.mark.unit
@pytest.mark.gate
def test_adversarial_artifact_substitution_records_not_equal() -> None:
    trusted = _candidate(artifact=_artifact(recommendation="trusted"))
    substituted = _candidate(artifact=_artifact(recommendation="malicious"))
    assert trusted != substituted


@pytest.mark.unit
@pytest.mark.gate
def test_adversarial_scope_substitution_records_not_equal() -> None:
    trusted = _candidate(identity=_identity(namespace="incident", subject="incident-123"))
    substituted = _candidate(
        identity=_identity(namespace="incident", subject="incident-999"),
    )
    assert trusted != substituted


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
        lineage_ref=decision_lineage_ref(resolved_version, DecisionBranchId(branch_id)),
    )


@pytest.mark.unit
@pytest.mark.gate
def test_decision_proposal_ref_valid() -> None:
    ref = _proposal_ref()
    assert ref.identity.version == ref.lineage_ref.version


@pytest.mark.unit
@pytest.mark.gate
def test_decision_proposal_ref_identity_version_mismatch_rejected() -> None:
    identity = _identity(version=DecisionVersion(2))
    with pytest.raises(ValueError, match="identity.version must match lineage_ref.version"):
        DecisionProposalRef(
            identity=identity,
            lineage_ref=decision_lineage_ref(DecisionVersion(3), DecisionBranchId("analysis-a")),
        )


@pytest.mark.unit
@pytest.mark.gate
def test_sibling_same_decision_distinct_proposal_refs() -> None:
    identity = _identity(version=DecisionVersion(2))
    ref_a = _proposal_ref(identity=identity, branch_id="analysis-a", version=DecisionVersion(2))
    ref_b = _proposal_ref(identity=identity, branch_id="analysis-b", version=DecisionVersion(2))
    assert ref_a != ref_b
    assert ref_a.identity == ref_b.identity


@pytest.mark.unit
@pytest.mark.gate
def test_same_lineage_different_decision_distinct_proposal_refs() -> None:
    identity_a = _identity()
    identity_b = _identity()
    ref_a = _proposal_ref(identity=identity_a, branch_id="analysis-a")
    ref_b = _proposal_ref(identity=identity_b, branch_id="analysis-a")
    assert ref_a.lineage_ref == ref_b.lineage_ref
    assert ref_a != ref_b


@pytest.mark.unit
@pytest.mark.gate
def test_decision_proposal_ref_is_immutable() -> None:
    ref = _proposal_ref()
    replaced = replace(ref)
    assert replaced == ref
    assert replaced is not ref


@pytest.mark.unit
@pytest.mark.gate
def test_decision_proposal_ref_direct_constructor_safe() -> None:
    identity = _identity(version=DecisionVersion(1))
    ref = DecisionProposalRef(
        identity=identity,
        lineage_ref=decision_lineage_ref(DecisionVersion(1), DecisionBranchId("analysis-a")),
    )
    assert ref.identity.decision_id == identity.decision_id


@pytest.mark.unit
@pytest.mark.gate
def test_decision_proposal_ref_sort_key_optional_execution_id_comparable() -> None:
    execution_with_id = _execution_lineage()
    execution_without_id = DecisionExecutionLineage(
        task_id=execution_with_id.task_id,
        run_id=execution_with_id.run_id,
        attempt_id=execution_with_id.attempt_id,
        execution_id=None,
    )
    identity = _identity(version=DecisionVersion(2))
    ref_with_execution = _proposal_ref(
        identity=replace(identity, execution=execution_with_id),
        branch_id="analysis-a",
        version=DecisionVersion(2),
    )
    ref_without_execution = _proposal_ref(
        identity=replace(identity, execution=execution_without_id),
        branch_id="analysis-a",
        version=DecisionVersion(2),
    )
    sorted_refs = sorted(
        (ref_with_execution, ref_without_execution),
        key=decision_proposal_ref_sort_key,
    )
    assert len(sorted_refs) == 2


@pytest.mark.unit
@pytest.mark.gate
def test_decision_proposal_ref_sort_key_reverse_input_deterministic() -> None:
    execution_with_id = _execution_lineage()
    execution_without_id = DecisionExecutionLineage(
        task_id=execution_with_id.task_id,
        run_id=execution_with_id.run_id,
        attempt_id=execution_with_id.attempt_id,
        execution_id=None,
    )
    identity = _identity(version=DecisionVersion(2))
    ref_with_execution = _proposal_ref(
        identity=replace(identity, execution=execution_with_id),
        branch_id="analysis-a",
        version=DecisionVersion(2),
    )
    ref_without_execution = _proposal_ref(
        identity=replace(identity, execution=execution_without_id),
        branch_id="analysis-a",
        version=DecisionVersion(2),
    )
    order_ab = sorted(
        (ref_with_execution, ref_without_execution),
        key=decision_proposal_ref_sort_key,
    )
    order_ba = sorted(
        (ref_without_execution, ref_with_execution),
        key=decision_proposal_ref_sort_key,
    )
    assert order_ab == order_ba


@pytest.mark.unit
@pytest.mark.gate
def test_decision_proposal_ref_sort_key_same_decision_siblings() -> None:
    identity = _identity(version=DecisionVersion(2))
    ref_a = _proposal_ref(identity=identity, branch_id="analysis-a", version=DecisionVersion(2))
    ref_b = _proposal_ref(identity=identity, branch_id="analysis-b", version=DecisionVersion(2))
    sorted_refs = sorted((ref_b, ref_a), key=decision_proposal_ref_sort_key)
    assert sorted_refs[0].lineage_ref.branch_id == "analysis-a"
    assert sorted_refs[1].lineage_ref.branch_id == "analysis-b"


@pytest.mark.unit
@pytest.mark.gate
def test_decision_proposal_ref_sort_key_exact_duplicate_same_key() -> None:
    identity = _identity(version=DecisionVersion(2))
    ref = _proposal_ref(identity=identity, branch_id="analysis-a", version=DecisionVersion(2))
    duplicate = _proposal_ref(identity=identity, branch_id="analysis-a", version=DecisionVersion(2))
    assert decision_proposal_ref_sort_key(ref) == decision_proposal_ref_sort_key(duplicate)


@pytest.mark.unit
@pytest.mark.gate
def test_candidate_decision_ref_derivation() -> None:
    identity = _identity(version=DecisionVersion(2))
    parent = decision_lineage_ref(DecisionVersion(1))
    lineage = _linear_lineage(DecisionVersion(2), parent, branch_id=DecisionBranchId("analysis-a"))
    candidate = CandidateDecision(
        identity=identity,
        artifact=_artifact(),
        lineage=lineage,
    )
    derived = candidate_decision_ref(candidate)
    expected = decision_proposal_ref(identity=identity, lineage_ref=lineage.current)
    assert derived == expected
