# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Pre-execution autonomy guard contract — not wired to spine in R6.1."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from intergrax.contracts.self_healing.autonomy.decision import AutonomyControlDecision


class AutonomyGuardVerdict(StrEnum):
    ALLOWED = "ALLOWED"
    DENIED = "DENIED"
    DEFERRED = "DEFERRED"


@dataclass(frozen=True, slots=True)
class AutonomyExecutionAdmissionContext:
    """
    Admission check input — correlation and prior classification only.

    No executor references, commands, or workflow handles.
    """

    tenant_id: str
    recommendation_correlation_id: str
    decision_id: str
    prior_decision: AutonomyControlDecision
    approval_token_ref: str | None = None

    def __post_init__(self) -> None:
        if not self.tenant_id.strip():
            raise ValueError("tenant_id required")
        if not self.recommendation_correlation_id.strip():
            raise ValueError("recommendation_correlation_id required")
        if not self.decision_id.strip():
            raise ValueError("decision_id required")
        if self.prior_decision.decision_id != self.decision_id:
            raise ValueError("decision_id mismatch with prior_decision")
        if self.prior_decision.recommendation_correlation_id != self.recommendation_correlation_id:
            raise ValueError("recommendation_correlation_id mismatch with prior_decision")


@dataclass(frozen=True, slots=True)
class AutonomyGuardCheckResult:
    verdict: AutonomyGuardVerdict
    rationale: str
    audit_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.rationale.strip():
            raise ValueError("rationale required")


@runtime_checkable
class AutonomyExecutionGuard(Protocol):
    @property
    def guard_id(self) -> str: ...

    def check(self, admission: AutonomyExecutionAdmissionContext) -> AutonomyGuardCheckResult:
        """Pre-execution autonomy boundary — must not invoke executors."""
        ...


__all__ = [
    "AutonomyExecutionAdmissionContext",
    "AutonomyExecutionGuard",
    "AutonomyGuardCheckResult",
    "AutonomyGuardVerdict",
]
