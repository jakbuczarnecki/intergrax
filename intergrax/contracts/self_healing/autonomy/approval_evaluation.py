# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Human approval evaluation result — no approval workflow (SELF-HEALING R6.2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from intergrax.contracts.self_healing.autonomy.approval import HumanApprovalRequirement
from intergrax.contracts.self_healing.autonomy.level import AutonomyLevel
from intergrax.contracts.self_healing.autonomy.policy_evaluation import AutonomyPolicyEvaluationResult
from intergrax.contracts.self_healing.autonomy.request import AutonomyControlRequest
from intergrax.contracts.self_healing.autonomy.risk import AutonomyRiskEvaluationResult


@dataclass(frozen=True, slots=True)
class HumanApprovalEvaluationResult:
    evaluator_id: str
    approval_required: bool
    reason_code: str
    rationale: str
    escalation_hint: str | None = None

    def __post_init__(self) -> None:
        if not self.evaluator_id.strip():
            raise ValueError("evaluator_id required")
        if not self.reason_code.strip():
            raise ValueError("reason_code required")
        if not self.rationale.strip():
            raise ValueError("rationale required")

    @classmethod
    def from_requirement(
        cls,
        evaluator_id: str,
        requirement: HumanApprovalRequirement,
    ) -> HumanApprovalEvaluationResult:
        return cls(
            evaluator_id=evaluator_id,
            approval_required=requirement.required,
            reason_code=requirement.reason_code,
            rationale=requirement.rationale,
            escalation_hint=requirement.escalation_hint,
        )


@runtime_checkable
class HumanApprovalEvaluator(Protocol):
    @property
    def evaluator_id(self) -> str: ...

    def evaluate(
        self,
        request: AutonomyControlRequest,
        policy_result: AutonomyPolicyEvaluationResult,
        risk_result: AutonomyRiskEvaluationResult,
        effective_level: AutonomyLevel,
    ) -> HumanApprovalEvaluationResult:
        """Determine whether human gate applies — read-only classification."""
        ...


__all__ = [
    "HumanApprovalEvaluationResult",
    "HumanApprovalEvaluator",
]
