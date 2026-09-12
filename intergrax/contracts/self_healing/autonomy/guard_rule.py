# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Optional guard rule plugins — policy/risk/approval extensions (SELF-HEALING R6.3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from intergrax.contracts.self_healing.autonomy.evaluation_result import AutonomyEvaluationResult
from intergrax.contracts.self_healing.autonomy.guard import AutonomyExecutionAdmissionContext


@dataclass(frozen=True, slots=True)
class AutonomyExecutionGuardRuleVerdict:
    permitted: bool
    rationale: str

    def __post_init__(self) -> None:
        if not self.rationale.strip():
            raise ValueError("rationale required")


@runtime_checkable
class AutonomyExecutionGuardRule(Protocol):
    @property
    def rule_id(self) -> str: ...

    def assess(
        self,
        admission: AutonomyExecutionAdmissionContext,
        evaluation: AutonomyEvaluationResult,
    ) -> AutonomyExecutionGuardRuleVerdict:
        """Supplemental guard rule — must not execute or mutate admission."""
        ...


__all__ = [
    "AutonomyExecutionGuardRule",
    "AutonomyExecutionGuardRuleVerdict",
]
