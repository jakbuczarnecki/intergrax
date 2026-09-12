# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Self-healing workflow execution context (SELF-HEALING R2)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from intergrax.contracts.self_healing.context import SelfHealingContext
from intergrax.contracts.self_healing.decision import SelfHealingDecision
from intergrax.contracts.self_healing.workflow.lifecycle import SelfHealingWorkflowState
from intergrax.contracts.self_healing.workflow.plan import SelfHealingPlan
from intergrax.contracts.self_healing.workflow.validation import ValidationResult


@dataclass(frozen=True, slots=True)
class SelfHealingWorkflowContext:
    workflow_id: str
    tenant_id: str
    plan: SelfHealingPlan
    decision: SelfHealingDecision
    healing_context: SelfHealingContext
    state: SelfHealingWorkflowState
    successful_step_ids: tuple[str, ...]
    failed_step_ids: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    validation_result: ValidationResult | None
    started_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        if not self.workflow_id.startswith("sh_wf_"):
            raise ValueError("workflow_id must be sh_wf_*")
        if self.tenant_id != self.plan.tenant_id:
            raise ValueError("tenant_id mismatch with plan")
        if self.tenant_id != self.healing_context.tenant_id:
            raise ValueError("tenant_id mismatch with healing context")


__all__ = ["SelfHealingWorkflowContext"]
