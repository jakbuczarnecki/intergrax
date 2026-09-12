# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Execution authorization outcome — auditable guard result (SELF-HEALING R6.3)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from intergrax.contracts.self_healing.autonomy.level import AutonomyLevel
from intergrax.contracts.self_healing.autonomy.policy_evaluation import AutonomyPolicyEvaluationResult


class AutonomyExecutionAuthorizationStatus(StrEnum):
    AUTHORIZED = "AUTHORIZED"
    CONDITIONAL = "CONDITIONAL"
    DENIED = "DENIED"


@dataclass(frozen=True, slots=True)
class AutonomyExecutionAuthorization:
    """
    Immutable authorization snapshot for the execution spine boundary.

    Not a substitute for ``execute=true`` — status and rationale are authoritative.
    """

    authorization_id: str
    status: AutonomyExecutionAuthorizationStatus
    decision_id: str
    evaluation_id: str | None
    autonomy_level: AutonomyLevel
    policy_result: AutonomyPolicyEvaluationResult | None
    recorded_at: datetime
    reasons: tuple[str, ...]
    guard_id: str
    recommendation_correlation_id: str

    def __post_init__(self) -> None:
        if not self.authorization_id.strip():
            raise ValueError("authorization_id required")
        if not self.decision_id.strip():
            raise ValueError("decision_id required")
        if not self.guard_id.strip():
            raise ValueError("guard_id required")
        if not self.recommendation_correlation_id.strip():
            raise ValueError("recommendation_correlation_id required")
        if not self.reasons:
            raise ValueError("reasons required")
        self.autonomy_level.ensure_runtime_activatable()

    def permits_execution(self) -> bool:
        """Fail-safe: only explicit AUTHORIZED may proceed to the existing executor."""
        return self.status is AutonomyExecutionAuthorizationStatus.AUTHORIZED


__all__ = [
    "AutonomyExecutionAuthorization",
    "AutonomyExecutionAuthorizationStatus",
]
