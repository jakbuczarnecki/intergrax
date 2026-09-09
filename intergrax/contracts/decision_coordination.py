# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed coordination semantic contracts (DS-NPSC-01).

Decision-owned description of *what* coordinated specialist work is required.
Expresses capability requirements and semantic payloads without physical agent
identity, lease ownership, or runtime scheduling topology.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Generic, NewType, TypeVar

from intergrax.contracts.decision_record import (
    DecisionArtifact,
    DecisionArtifactKind,
    validate_decision_artifact_kind,
)

DECISION_COORDINATION_ARTIFACT_KIND: Final = "decision.coordination"

DecisionContributionId = NewType("DecisionContributionId", str)
DecisionCapabilityId = NewType("DecisionCapabilityId", str)

PayloadT = TypeVar("PayloadT")


class DecisionCoordinationShape(StrEnum):
    """Semantic coordination shape — not scheduler topology."""

    SINGLE = "single"
    FAN_OUT = "fan_out"


def decision_coordination_artifact_kind() -> DecisionArtifactKind:
    """Canonical artifact kind for coordination semantic payloads."""
    return validate_decision_artifact_kind(DECISION_COORDINATION_ARTIFACT_KIND)


def validate_decision_contribution_id(value: object) -> DecisionContributionId:
    if type(value) is not str:
        raise TypeError(
            f"DecisionContributionId must be str, got {type(value).__name__}",
        )
    if not value or not value.strip():
        raise ValueError(
            "DecisionContributionId must be non-empty and not whitespace-only",
        )
    if value != value.strip():
        raise ValueError(
            "DecisionContributionId must not contain leading or trailing whitespace",
        )
    return DecisionContributionId(value)


def validate_decision_capability_id(value: object) -> DecisionCapabilityId:
    if type(value) is not str:
        raise TypeError(
            f"DecisionCapabilityId must be str, got {type(value).__name__}",
        )
    if not value or not value.strip():
        raise ValueError(
            "DecisionCapabilityId must be non-empty and not whitespace-only",
        )
    if value != value.strip():
        raise ValueError(
            "DecisionCapabilityId must not contain leading or trailing whitespace",
        )
    return DecisionCapabilityId(value)


@dataclass(frozen=True, slots=True)
class DecisionCapabilityRequirement:
    """Typed required capability identity — expresses WHAT, not WHO."""

    capability_id: DecisionCapabilityId

    def __post_init__(self) -> None:
        validate_decision_capability_id(self.capability_id)


@dataclass(frozen=True, slots=True)
class DecisionCoordinationContribution(Generic[PayloadT]):
    """One semantic contribution within a coordination decision."""

    contribution_id: DecisionContributionId
    capability_requirement: DecisionCapabilityRequirement
    payload: PayloadT

    def __post_init__(self) -> None:
        validate_decision_contribution_id(self.contribution_id)
        if type(self.capability_requirement) is not DecisionCapabilityRequirement:
            raise TypeError(
                "DecisionCoordinationContribution.capability_requirement "
                "must be DecisionCapabilityRequirement",
            )


def _validate_coordination_contributions(
    contributions: tuple[DecisionCoordinationContribution[PayloadT], ...],
) -> None:
    if not contributions:
        raise ValueError("contributions must be non-empty")
    seen: set[DecisionContributionId] = set()
    for contribution in contributions:
        if type(contribution) is not DecisionCoordinationContribution:
            raise TypeError("contributions must contain DecisionCoordinationContribution")
        if contribution.contribution_id in seen:
            raise ValueError("contributions must not contain duplicate contribution_id")
        seen.add(contribution.contribution_id)


def _validate_coordination_shape_cardinality(
    shape: DecisionCoordinationShape,
    contributions: tuple[DecisionCoordinationContribution[PayloadT], ...],
) -> None:
    count = len(contributions)
    if shape is DecisionCoordinationShape.SINGLE:
        if count != 1:
            raise ValueError(
                "DecisionCoordinationShape.SINGLE requires exactly one contribution",
            )
        return
    if shape is DecisionCoordinationShape.FAN_OUT:
        if count < 2:
            raise ValueError(
                "DecisionCoordinationShape.FAN_OUT requires at least two contributions",
            )
        return
    raise ValueError(f"unsupported DecisionCoordinationShape: {shape!r}")


@dataclass(frozen=True, slots=True)
class DecisionCoordinationSemantic(Generic[PayloadT]):
    """Typed coordination semantic payload hosted by DecisionArtifact."""

    shape: DecisionCoordinationShape
    contributions: tuple[DecisionCoordinationContribution[PayloadT], ...]

    def __post_init__(self) -> None:
        if type(self.shape) is not DecisionCoordinationShape:
            raise TypeError(
                "DecisionCoordinationSemantic.shape must be DecisionCoordinationShape",
            )
        _validate_coordination_contributions(self.contributions)
        _validate_coordination_shape_cardinality(self.shape, self.contributions)


def decision_coordination_artifact(
    payload: DecisionCoordinationSemantic[PayloadT],
) -> DecisionArtifact[DecisionCoordinationSemantic[PayloadT]]:
    """Wrap one coordination semantic payload in the canonical artifact carrier."""
    if type(payload) is not DecisionCoordinationSemantic:
        raise TypeError("payload must be DecisionCoordinationSemantic")
    return DecisionArtifact(
        kind=decision_coordination_artifact_kind(),
        content=payload,
    )
