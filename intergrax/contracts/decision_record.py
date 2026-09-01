# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Decision System record contracts (DS-CORE-02, DS-CORE-07).

Candidate proposals and authoritative accepted decisions share typed identity,
artifact, and immutable version lineage with optional parallel branch identity.
Authority-grade payload types supplied to ``DecisionArtifact`` must themselves
be immutable typed value contracts; this module does not enforce deep
immutability of generic payloads at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, NewType, TypeVar

from intergrax.contracts.decision_identity import (
    DecisionIdentity,
    DecisionVersion,
)

DecisionArtifactKind = NewType("DecisionArtifactKind", str)
DecisionBranchId = NewType("DecisionBranchId", str)

_ROOT_BRANCH_ID = "main"

T = TypeVar("T")


def validate_decision_artifact_kind(value: object) -> DecisionArtifactKind:
    if type(value) is not str:
        raise TypeError(
            f"DecisionArtifactKind must be str, got {type(value).__name__}",
        )
    if not value or not value.strip():
        raise ValueError(
            "DecisionArtifactKind must be non-empty and not whitespace-only",
        )
    if value != value.strip():
        raise ValueError(
            "DecisionArtifactKind must not contain leading or trailing whitespace",
        )
    return DecisionArtifactKind(value)


def validate_decision_branch_id(value: object) -> DecisionBranchId:
    if type(value) is not str:
        raise TypeError(
            f"DecisionBranchId must be str, got {type(value).__name__}",
        )
    if not value or not value.strip():
        raise ValueError(
            "DecisionBranchId must be non-empty and not whitespace-only",
        )
    if value != value.strip():
        raise ValueError(
            "DecisionBranchId must not contain leading or trailing whitespace",
        )
    return DecisionBranchId(value)


def initial_decision_branch_id() -> DecisionBranchId:
    """Deterministic root branch identity for initial and main-line revisions."""
    return DecisionBranchId(_ROOT_BRANCH_ID)


@dataclass(frozen=True, slots=True)
class DecisionArtifact(Generic[T]):
    """Typed decision artifact carrier: explicit kind plus domain-owned payload."""

    kind: DecisionArtifactKind
    content: T

    def __post_init__(self) -> None:
        validate_decision_artifact_kind(self.kind)


@dataclass(frozen=True, slots=True)
class DecisionLineageRef:
    """One typed node in decision version lineage: generation plus branch identity."""

    version: DecisionVersion
    branch_id: DecisionBranchId

    def __post_init__(self) -> None:
        if type(self.version) is not DecisionVersion:
            raise TypeError("DecisionLineageRef.version must be DecisionVersion")
        validate_decision_branch_id(self.branch_id)


def decision_lineage_ref(
    version: DecisionVersion,
    branch_id: DecisionBranchId | None = None,
) -> DecisionLineageRef:
    """Build one lineage node; default branch is the deterministic root branch."""
    if type(version) is not DecisionVersion:
        raise TypeError("version must be DecisionVersion")
    if branch_id is None:
        resolved_branch = initial_decision_branch_id()
    else:
        resolved_branch = validate_decision_branch_id(branch_id)
    return DecisionLineageRef(version=version, branch_id=resolved_branch)


def _canonicalize_lineage_parents(
    parents: tuple[DecisionLineageRef, ...],
) -> tuple[DecisionLineageRef, ...]:
    return tuple(sorted(parents, key=lambda ref: (ref.version.value, ref.branch_id)))


@dataclass(frozen=True, slots=True)
class DecisionVersionLineage:
    """Immutable local lineage relation: current node plus parent refs.

    Parent tuple order is preserved but semantically non-ranking — parents form
    a set of lineage causes, not a winner ranking. Use ``decision_version_lineage``
    when deterministic parent ordering is required.
    """

    current: DecisionLineageRef
    parents: tuple[DecisionLineageRef, ...] = ()

    def __post_init__(self) -> None:
        if type(self.current) is not DecisionLineageRef:
            raise TypeError("DecisionVersionLineage.current must be DecisionLineageRef")
        for parent in self.parents:
            if type(parent) is not DecisionLineageRef:
                raise TypeError(
                    "DecisionVersionLineage.parents must contain DecisionLineageRef",
                )
        _validate_decision_version_lineage(self.current, self.parents)


def decision_version_lineage(
    *,
    current: DecisionLineageRef,
    parents: tuple[DecisionLineageRef, ...] = (),
) -> DecisionVersionLineage:
    """Construct lineage with parents canonicalized deterministically."""
    return DecisionVersionLineage(
        current=current,
        parents=_canonicalize_lineage_parents(parents),
    )


def _validate_decision_version_lineage(
    current: DecisionLineageRef,
    parents: tuple[DecisionLineageRef, ...],
) -> None:
    current_version = current.version.value
    if not parents:
        if current_version != 1:
            raise ValueError(
                "DecisionVersionLineage without parents requires current version 1",
            )
        return
    if current_version <= 1:
        raise ValueError(
            "DecisionVersionLineage with parents requires current version > 1",
        )
    seen: set[tuple[int, str]] = set()
    for parent in parents:
        if parent == current:
            raise ValueError(
                "DecisionVersionLineage.current cannot appear in parents",
            )
        parent_key = (parent.version.value, parent.branch_id)
        if parent_key in seen:
            raise ValueError("DecisionVersionLineage.parents must not contain duplicates")
        seen.add(parent_key)
        if parent.version.value >= current_version:
            raise ValueError(
                "DecisionVersionLineage parent version must be earlier than current version",
            )


def _validate_identity_lineage_alignment(
    identity: DecisionIdentity,
    lineage: DecisionVersionLineage,
) -> None:
    if identity.version != lineage.current.version:
        raise ValueError(
            "DecisionIdentity.version must match DecisionVersionLineage.current.version",
        )


def _validate_proposal_ref_identity_lineage_alignment(
    identity: DecisionIdentity,
    lineage_ref: DecisionLineageRef,
) -> None:
    if identity.version != lineage_ref.version:
        raise ValueError(
            "DecisionProposalRef identity.version must match lineage_ref.version",
        )


def decision_proposal_ref_sort_key(
    ref: DecisionProposalRef,
) -> tuple[str, str, str, str, str, str, str, int, str, int, str]:
    """Deterministic ordering key using typed Decision identity and lineage fields."""
    execution = ref.identity.execution
    execution_id_present = 0 if execution.execution_id is None else 1
    execution_id_value = (
        "" if execution.execution_id is None else str(execution.execution_id)
    )
    return (
        str(ref.identity.decision_id),
        ref.identity.tenant_id,
        ref.identity.scope.namespace,
        ref.identity.scope.subject,
        str(ref.identity.execution.task_id),
        str(ref.identity.execution.run_id),
        str(ref.identity.execution.attempt_id),
        execution_id_present,
        execution_id_value,
        ref.lineage_ref.version.value,
        str(ref.lineage_ref.branch_id),
    )


@dataclass(frozen=True, slots=True)
class DecisionProposalRef:
    """Exact proposal reference within one canonical Decision identity boundary."""

    identity: DecisionIdentity
    lineage_ref: DecisionLineageRef

    def __post_init__(self) -> None:
        if type(self.identity) is not DecisionIdentity:
            raise TypeError("DecisionProposalRef.identity must be DecisionIdentity")
        if type(self.lineage_ref) is not DecisionLineageRef:
            raise TypeError("DecisionProposalRef.lineage_ref must be DecisionLineageRef")
        _validate_proposal_ref_identity_lineage_alignment(self.identity, self.lineage_ref)


def decision_proposal_ref(
    *,
    identity: DecisionIdentity,
    lineage_ref: DecisionLineageRef,
) -> DecisionProposalRef:
    """Build one exact proposal reference with identity/lineage alignment enforced."""
    return DecisionProposalRef(identity=identity, lineage_ref=lineage_ref)


@dataclass(frozen=True, slots=True)
class CandidateDecision(Generic[T]):
    """Proposed decision version subject to verification, revision, or adjudication."""

    identity: DecisionIdentity
    artifact: DecisionArtifact[T]
    lineage: DecisionVersionLineage

    def __post_init__(self) -> None:
        if type(self.identity) is not DecisionIdentity:
            raise TypeError("CandidateDecision.identity must be DecisionIdentity")
        if type(self.artifact) is not DecisionArtifact:
            raise TypeError("CandidateDecision.artifact must be DecisionArtifact")
        if type(self.lineage) is not DecisionVersionLineage:
            raise TypeError("CandidateDecision.lineage must be DecisionVersionLineage")
        _validate_identity_lineage_alignment(self.identity, self.lineage)


def candidate_decision(
    *,
    identity: DecisionIdentity,
    artifact_kind: DecisionArtifactKind,
    payload: T,
    lineage: DecisionVersionLineage | None = None,
) -> CandidateDecision[T]:
    """Assemble a typed candidate from deliberation output."""
    if lineage is None:
        resolved_lineage = decision_version_lineage(
            current=decision_lineage_ref(identity.version),
        )
    else:
        resolved_lineage = lineage
    artifact = DecisionArtifact(kind=artifact_kind, content=payload)
    return CandidateDecision(
        identity=identity,
        artifact=artifact,
        lineage=resolved_lineage,
    )


def candidate_decision_ref(candidate: CandidateDecision[T]) -> DecisionProposalRef:
    """Derive the exact proposal reference for one candidate decision."""
    if type(candidate) is not CandidateDecision:
        raise TypeError("candidate must be CandidateDecision")
    return DecisionProposalRef(
        identity=candidate.identity,
        lineage_ref=candidate.lineage.current,
    )


@dataclass(frozen=True, slots=True)
class AuthoritativeAcceptedDecision(Generic[T]):
    """Terminal accepted decision version — authoritative outcome, not execution authorization."""

    identity: DecisionIdentity
    artifact: DecisionArtifact[T]
    lineage: DecisionVersionLineage

    def __post_init__(self) -> None:
        if type(self.identity) is not DecisionIdentity:
            raise TypeError(
                "AuthoritativeAcceptedDecision.identity must be DecisionIdentity",
            )
        if type(self.artifact) is not DecisionArtifact:
            raise TypeError(
                "AuthoritativeAcceptedDecision.artifact must be DecisionArtifact",
            )
        if type(self.lineage) is not DecisionVersionLineage:
            raise TypeError(
                "AuthoritativeAcceptedDecision.lineage must be DecisionVersionLineage",
            )
        _validate_identity_lineage_alignment(self.identity, self.lineage)
