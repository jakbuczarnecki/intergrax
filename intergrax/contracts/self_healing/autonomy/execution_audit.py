# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Persistence port for autonomy execution guard audits (SELF-HEALING R6.3)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

from intergrax.contracts.self_healing.autonomy.execution_authorization import (
    AutonomyExecutionAuthorization,
    AutonomyExecutionAuthorizationStatus,
)
from intergrax.contracts.self_healing.autonomy.level import AutonomyLevel


@dataclass(frozen=True, slots=True)
class AutonomyExecutionAuditRecord:
    record_id: str
    authorization_id: str
    decision_id: str
    evaluation_id: str | None
    autonomy_level: AutonomyLevel
    policy_id: str | None
    policy_version: str | None
    status: AutonomyExecutionAuthorizationStatus
    guard_id: str
    recommendation_correlation_id: str
    recorded_at: datetime
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.record_id.strip():
            raise ValueError("record_id required")
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

    @classmethod
    def from_authorization(
        cls,
        record_id: str,
        authorization: AutonomyExecutionAuthorization,
    ) -> AutonomyExecutionAuditRecord:
        policy_id: str | None = None
        policy_version: str | None = None
        if authorization.policy_result is not None:
            policy_id = authorization.policy_result.policy_id
            policy_version = authorization.policy_result.policy_version
        return cls(
            record_id=record_id,
            authorization_id=authorization.authorization_id,
            decision_id=authorization.decision_id,
            evaluation_id=authorization.evaluation_id,
            autonomy_level=authorization.autonomy_level,
            policy_id=policy_id,
            policy_version=policy_version,
            status=authorization.status,
            guard_id=authorization.guard_id,
            recommendation_correlation_id=authorization.recommendation_correlation_id,
            recorded_at=authorization.recorded_at,
            reasons=authorization.reasons,
        )


@runtime_checkable
class AutonomyExecutionAuditRepository(Protocol):
    def append(self, record: AutonomyExecutionAuditRecord) -> AutonomyExecutionAuditRecord: ...


__all__ = [
    "AutonomyExecutionAuditRecord",
    "AutonomyExecutionAuditRepository",
]
