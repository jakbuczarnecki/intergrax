# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Self-healing plan contract — declarative only (SELF-HEALING R2)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from uuid import uuid4

from intergrax.contracts.self_healing.workflow.step import SelfHealingStep


class SelfHealingRiskLevel(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


def mint_self_healing_plan_id() -> str:
    return f"sh_plan_{uuid4().hex}"


@dataclass(frozen=True, slots=True)
class SelfHealingPlan:
    """
    Immutable healing process description.

    Does not execute — ``SelfHealingWorkflowOrchestrator`` coordinates lifecycle only.
    """

    plan_id: str
    plan_version: str
    strategy_id: str
    tenant_id: str
    scope: str
    steps: tuple[SelfHealingStep, ...]
    validation_policy_id: str
    rollback_policy_id: str
    evidence_refs: tuple[str, ...]
    risk_level: SelfHealingRiskLevel

    def __post_init__(self) -> None:
        if not self.plan_id.startswith("sh_plan_"):
            raise ValueError("plan_id must be sh_plan_*")
        if not self.plan_version.strip():
            raise ValueError("plan_version required")
        if not self.strategy_id.strip():
            raise ValueError("strategy_id required")
        if not self.tenant_id.strip():
            raise ValueError("tenant_id required")
        if not self.scope.strip():
            raise ValueError("scope required")
        if not self.steps:
            raise ValueError("steps must be non-empty")
        if not self.validation_policy_id.strip():
            raise ValueError("validation_policy_id required")
        if not self.rollback_policy_id.strip():
            raise ValueError("rollback_policy_id required")
        if not self.evidence_refs:
            raise ValueError("evidence_refs must be non-empty")
        ordered = sorted(self.steps, key=lambda s: s.sequence_number)
        if tuple(ordered) != self.steps:
            raise ValueError("steps must be sorted by sequence_number")
        seqs = [s.sequence_number for s in self.steps]
        if len(set(seqs)) != len(seqs):
            raise ValueError("duplicate sequence_number in steps")


__all__ = [
    "SelfHealingPlan",
    "SelfHealingRiskLevel",
    "mint_self_healing_plan_id",
]
