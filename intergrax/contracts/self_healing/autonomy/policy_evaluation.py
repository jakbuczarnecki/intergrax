# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Policy evaluation result — separate from final autonomy evaluation (SELF-HEALING R6.2)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from intergrax.contracts.self_healing.autonomy.level import AutonomyLevel
from intergrax.contracts.self_healing.autonomy.policy import AutonomyConstraintDescriptor
from intergrax.contracts.self_healing.autonomy.request import AutonomyControlRequest


class AutonomyPolicyEvaluationVerdict(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"


@dataclass(frozen=True, slots=True)
class AutonomyPolicyEvaluationResult:
    evaluator_id: str
    policy_id: str
    policy_version: str
    verdict: AutonomyPolicyEvaluationVerdict
    suggested_level: AutonomyLevel
    constraint_descriptors: tuple[AutonomyConstraintDescriptor, ...]
    rationale: str

    def __post_init__(self) -> None:
        if not self.evaluator_id.strip():
            raise ValueError("evaluator_id required")
        if not self.policy_id.strip():
            raise ValueError("policy_id required")
        if not self.policy_version.strip():
            raise ValueError("policy_version required")
        if not self.rationale.strip():
            raise ValueError("rationale required")
        self.suggested_level.ensure_runtime_activatable()


@runtime_checkable
class AutonomyPolicyEvaluator(Protocol):
    @property
    def evaluator_id(self) -> str: ...

    def evaluate(self, request: AutonomyControlRequest) -> AutonomyPolicyEvaluationResult:
        """Evaluate enterprise policy posture — must not execute strategies."""
        ...


__all__ = [
    "AutonomyPolicyEvaluationResult",
    "AutonomyPolicyEvaluationVerdict",
    "AutonomyPolicyEvaluator",
]
