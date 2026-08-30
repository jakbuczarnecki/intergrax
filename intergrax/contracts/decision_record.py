# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Decision System record contracts (DS-CORE-02).

Candidate proposals and authoritative accepted decisions share typed identity,
artifact, and immutable version lineage. Authority-grade payload types supplied
to ``DecisionArtifact`` must themselves be immutable typed value contracts;
this module does not enforce deep immutability of generic payloads at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, NewType, TypeVar

from intergrax.contracts.decision_identity import (
    DecisionIdentity,
    DecisionVersion,
)

DecisionArtifactKind = NewType("DecisionArtifactKind", str)

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


@dataclass(frozen=True, slots=True)
class DecisionArtifact(Generic[T]):
    """Typed decision artifact carrier: explicit kind plus domain-owned payload."""

    kind: DecisionArtifactKind
    content: T

    def __post_init__(self) -> None:
        validate_decision_artifact_kind(self.kind)


@dataclass(frozen=True, slots=True)
class DecisionVersionLineage:
    """Immutable parent lineage for one decision version (no branch graph)."""

    current_version: DecisionVersion
    parent_version: DecisionVersion | None = None

    def __post_init__(self) -> None:
        if type(self.current_version) is not DecisionVersion:
            raise TypeError(
                "DecisionVersionLineage.current_version must be DecisionVersion",
            )
        if self.parent_version is not None and type(self.parent_version) is not DecisionVersion:
            raise TypeError(
                "DecisionVersionLineage.parent_version must be DecisionVersion or None",
            )
        current = self.current_version.value
        parent = self.parent_version
        if parent is None:
            if current != 1:
                raise ValueError(
                    "DecisionVersionLineage without parent requires current_version 1",
                )
            return
        if parent.value >= current:
            raise ValueError(
                "DecisionVersionLineage.parent_version must be earlier than current_version",
            )


def _validate_identity_lineage_alignment(
    identity: DecisionIdentity,
    lineage: DecisionVersionLineage,
) -> None:
    if identity.version != lineage.current_version:
        raise ValueError(
            "DecisionIdentity.version must match DecisionVersionLineage.current_version",
        )


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
