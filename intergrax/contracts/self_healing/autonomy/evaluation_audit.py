# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Audit trail contract for autonomy decision evaluation (SELF-HEALING R6.2)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

from intergrax.contracts.self_healing.autonomy.evaluation_result import AutonomyEvaluationResult


@dataclass(frozen=True, slots=True)
class AutonomyEvaluationAuditTrailEntry:
    evaluation_id: str
    evaluator_id: str
    policy_id: str
    policy_version: str
    risk_evaluator_id: str
    approval_evaluator_id: str
    contract_version: str
    recorded_at: datetime
    principal_id: str | None = None

    def __post_init__(self) -> None:
        if not self.evaluation_id.strip():
            raise ValueError("evaluation_id required")
        if not self.evaluator_id.strip():
            raise ValueError("evaluator_id required")
        if not self.policy_id.strip():
            raise ValueError("policy_id required")
        if not self.policy_version.strip():
            raise ValueError("policy_version required")
        if not self.risk_evaluator_id.strip():
            raise ValueError("risk_evaluator_id required")
        if not self.approval_evaluator_id.strip():
            raise ValueError("approval_evaluator_id required")
        if not self.contract_version.strip():
            raise ValueError("contract_version required")

    @classmethod
    def from_evaluation_result(
        cls,
        result: AutonomyEvaluationResult,
        recorded_at: datetime,
    ) -> AutonomyEvaluationAuditTrailEntry:
        return cls(
            evaluation_id=result.evaluation_id,
            evaluator_id=result.evaluator_id,
            policy_id=result.policy_result.policy_id,
            policy_version=result.policy_result.policy_version,
            risk_evaluator_id=result.risk_result.evaluator_id,
            approval_evaluator_id=result.approval_result.evaluator_id,
            contract_version=result.contract_version,
            recorded_at=recorded_at,
            principal_id=result.audit_bundle.principal_id,
        )


@runtime_checkable
class AutonomyEvaluationAuditRecorder(Protocol):
    def record(self, entry: AutonomyEvaluationAuditTrailEntry) -> None:
        """Persist evaluation audit metadata — storage adapter supplied by host."""
        ...


__all__ = [
    "AutonomyEvaluationAuditRecorder",
    "AutonomyEvaluationAuditTrailEntry",
]
