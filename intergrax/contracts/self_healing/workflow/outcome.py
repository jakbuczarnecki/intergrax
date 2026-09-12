# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Self-healing workflow outcome — learning loop input (SELF-HEALING R2)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from intergrax.contracts.self_healing.workflow.validation import ValidationResult


@dataclass(frozen=True, slots=True)
class SelfHealingWorkflowOutcome:
    workflow_id: str
    strategy_id: str
    tenant_id: str
    successful_steps: tuple[str, ...]
    failed_steps: tuple[str, ...]
    rollback_executed: bool
    validation_result: ValidationResult | None
    recovery_time: timedelta
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.workflow_id.startswith("sh_wf_"):
            raise ValueError("workflow_id must be sh_wf_*")
        if not self.strategy_id.strip():
            raise ValueError("strategy_id required")
        if not self.tenant_id.strip():
            raise ValueError("tenant_id required")
        if not self.evidence_refs:
            raise ValueError("evidence_refs must be non-empty")


__all__ = ["SelfHealingWorkflowOutcome"]
