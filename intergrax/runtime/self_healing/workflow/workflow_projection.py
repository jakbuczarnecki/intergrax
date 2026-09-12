# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Operator read projection for healing workflows (SELF-HEALING R2)."""

from __future__ import annotations

from intergrax.contracts.self_healing.workflow.lifecycle import SelfHealingWorkflowAuditEntry
from intergrax.contracts.self_healing_investigation_read import RelatedSelfHealingWorkflowHistoryEntryView


def project_healing_workflow_history(
    audit_trail: tuple[SelfHealingWorkflowAuditEntry, ...],
    *,
    workflow_id: str,
    strategy_id: str,
    plan_id: str,
) -> tuple[RelatedSelfHealingWorkflowHistoryEntryView, ...]:
    views: list[RelatedSelfHealingWorkflowHistoryEntryView] = []
    for entry in audit_trail:
        if entry.workflow_id != workflow_id:
            continue
        views.append(
            RelatedSelfHealingWorkflowHistoryEntryView(
                workflow_id=workflow_id,
                strategy_id=strategy_id,
                plan_id=plan_id,
                tenant_id=entry.tenant_id,
                from_state=entry.from_state.value,
                to_state=entry.to_state.value,
                actor=entry.actor,
                reason=entry.reason,
                recorded_at=entry.recorded_at,
            ),
        )
    return tuple(views)


__all__ = ["project_healing_workflow_history"]
