# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Healing execution timeline read projection (SELF-HEALING R3)."""

from __future__ import annotations

from intergrax.contracts.self_healing.execution.lifecycle import SelfHealingExecutionLifecycleAuditEntry
from intergrax.contracts.self_healing_investigation_read import (
    HealingExecutionTimelineStageView,
    HealingExecutionTimelineView,
)

def _map_state_to_stage(state: str) -> str:
    mapping = {
        "CREATED": "Healing Plan",
        "APPROVAL_PENDING": "Approval",
        "APPROVED": "Approval",
        "EXECUTION_REQUESTED": "Execution",
        "EXECUTING": "Execution",
        "OBSERVING": "Evidence",
        "VALIDATING": "Validation",
        "COMPLETED": "Outcome",
        "ROLLBACK_PENDING": "Validation",
        "ROLLED_BACK": "Outcome",
        "FAILED": "Outcome",
    }
    return mapping.get(state, "Outcome")


def project_healing_execution_timeline(
    audit_entries: tuple[SelfHealingExecutionLifecycleAuditEntry, ...],
    *,
    workflow_id: str,
    strategy_id: str,
    plan_id: str,
    problem_label: str = "Problem",
    prediction_label: str = "Prediction",
    recommendation_label: str = "Recommendation",
) -> HealingExecutionTimelineView:
    stages: list[HealingExecutionTimelineStageView] = [
        HealingExecutionTimelineStageView(phase="Problem", label=problem_label, recorded_at=None),
        HealingExecutionTimelineStageView(phase="Prediction", label=prediction_label, recorded_at=None),
        HealingExecutionTimelineStageView(
            phase="Recommendation",
            label=recommendation_label,
            recorded_at=None,
        ),
    ]
    for entry in audit_entries:
        if entry.workflow_id != workflow_id:
            continue
        stage = _map_state_to_stage(entry.to_state.value)
        stages.append(
            HealingExecutionTimelineStageView(
                phase=stage,
                label=f"{entry.from_state.value} → {entry.to_state.value}",
                recorded_at=entry.recorded_at,
                evidence_refs=entry.evidence_refs,
            ),
        )
    outcome_at = audit_entries[-1].recorded_at if audit_entries else None
    stages.append(
        HealingExecutionTimelineStageView(
            phase="Outcome",
            label=f"workflow={workflow_id} plan={plan_id} strategy={strategy_id}",
            recorded_at=outcome_at,
        ),
    )
    return HealingExecutionTimelineView(
        workflow_id=workflow_id,
        strategy_id=strategy_id,
        plan_id=plan_id,
        stages=tuple(stages),
    )


__all__ = ["project_healing_execution_timeline"]
