# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Explainable autonomy decision evaluation (SELF-HEALING R6.2)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.self_healing.autonomy.policy import AutonomyConstraintDescriptor


@dataclass(frozen=True, slots=True)
class AutonomyExplanationBullet:
    code: str
    message: str

    def __post_init__(self) -> None:
        if not self.code.strip():
            raise ValueError("code required")
        if not self.message.strip():
            raise ValueError("message required")


@dataclass(frozen=True, slots=True)
class AutonomyRuleInfluence:
    rule_id: str
    influence_summary: str

    def __post_init__(self) -> None:
        if not self.rule_id.strip():
            raise ValueError("rule_id required")
        if not self.influence_summary.strip():
            raise ValueError("influence_summary required")


@dataclass(frozen=True, slots=True)
class AutonomyDecisionExplanation:
    """Why the evaluation reached its posture — not an execution directive."""

    summary: str
    because: tuple[AutonomyExplanationBullet, ...]
    rules_influenced: tuple[AutonomyRuleInfluence, ...]
    constraints_active: tuple[AutonomyConstraintDescriptor, ...]

    def __post_init__(self) -> None:
        if not self.summary.strip():
            raise ValueError("summary required")


__all__ = [
    "AutonomyDecisionExplanation",
    "AutonomyExplanationBullet",
    "AutonomyRuleInfluence",
]
