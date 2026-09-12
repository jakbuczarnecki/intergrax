# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Autonomy policy plugin port — enterprise action posture (SELF-HEALING R6.1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from intergrax.contracts.self_healing.autonomy.level import AutonomyLevel
from intergrax.contracts.self_healing.autonomy.request import AutonomyControlRequest


@dataclass(frozen=True, slots=True)
class AutonomyConstraintDescriptor:
    """Immutable limit label — descriptive only in R6.1."""

    code: str
    description: str

    def __post_init__(self) -> None:
        if not self.code.strip():
            raise ValueError("code required")
        if not self.description.strip():
            raise ValueError("description required")


@dataclass(frozen=True, slots=True)
class AutonomyPolicyOutcome:
    policy_id: str
    policy_version: str
    suggested_level: AutonomyLevel
    constraint_descriptors: tuple[AutonomyConstraintDescriptor, ...]
    rationale: str

    def __post_init__(self) -> None:
        if not self.policy_id.strip():
            raise ValueError("policy_id required")
        if not self.policy_version.strip():
            raise ValueError("policy_version required")
        if not self.rationale.strip():
            raise ValueError("rationale required")
        self.suggested_level.ensure_runtime_activatable()


@runtime_checkable
class AutonomyPolicy(Protocol):
    @property
    def policy_id(self) -> str: ...

    def evaluate(self, request: AutonomyControlRequest) -> AutonomyPolicyOutcome:
        """Determine suggested autonomy posture — must not execute or select strategies."""
        ...


__all__ = [
    "AutonomyConstraintDescriptor",
    "AutonomyPolicy",
    "AutonomyPolicyOutcome",
]
