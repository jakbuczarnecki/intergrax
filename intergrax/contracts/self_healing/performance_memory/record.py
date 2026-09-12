# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Immutable strategy execution experience — factual history only (SELF-HEALING R5.1)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from uuid import uuid4


class SelfHealingStrategyExecutionOutcome(StrEnum):
    """Terminal repair outcome for one workflow run — not a quality score."""

    REPAIR_SUCCEEDED = "REPAIR_SUCCEEDED"
    REPAIR_FAILED = "REPAIR_FAILED"
    ROLLED_BACK = "ROLLED_BACK"
    INCONCLUSIVE = "INCONCLUSIVE"


def mint_self_healing_strategy_performance_experience_id() -> str:
    return f"sh_spm_{uuid4().hex}"


@dataclass(frozen=True, slots=True)
class SelfHealingStrategyPerformanceExperience:
    """
    One observed self-healing strategy run.

    Stores historical facts only — no ranking weights, scores, or selection hints.
    """

    experience_id: str
    tenant_id: str
    strategy_id: str
    workflow_id: str
    plan_id: str
    execution_ids: tuple[str, ...]
    diagnostic_investigation_id: str
    execution_outcome: SelfHealingStrategyExecutionOutcome
    rollback_executed: bool
    recovery_time_seconds: float
    evidence_refs: tuple[str, ...]
    recorded_at: datetime

    def __post_init__(self) -> None:
        if not self.experience_id.startswith("sh_spm_"):
            raise ValueError("experience_id must be sh_spm_*")
        if not self.tenant_id.strip():
            raise ValueError("tenant_id required")
        if not self.strategy_id.strip():
            raise ValueError("strategy_id required")
        if not self.workflow_id.startswith("sh_wf_"):
            raise ValueError("workflow_id must be sh_wf_*")
        if not self.plan_id.startswith("sh_plan_"):
            raise ValueError("plan_id must be sh_plan_*")
        if not self.diagnostic_investigation_id.strip():
            raise ValueError("diagnostic_investigation_id required")
        if self.recovery_time_seconds < 0.0:
            raise ValueError("recovery_time_seconds must be >= 0")
        if not self.evidence_refs:
            raise ValueError("evidence_refs must be non-empty")


__all__ = [
    "SelfHealingStrategyExecutionOutcome",
    "SelfHealingStrategyPerformanceExperience",
    "mint_self_healing_strategy_performance_experience_id",
]
