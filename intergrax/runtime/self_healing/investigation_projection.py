# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Operator read projection for self-healing history (SELF-HEALING R1)."""

from __future__ import annotations

from intergrax.contracts.self_healing.audit import SelfHealingAuditRecord
from intergrax.contracts.self_healing_investigation_read import RelatedSelfHealingHistoryEntryView


def project_self_healing_history(
    records: tuple[SelfHealingAuditRecord, ...],
    *,
    justification_by_decision: dict[str, str] | None = None,
) -> tuple[RelatedSelfHealingHistoryEntryView, ...]:
    justifications = justification_by_decision or {}
    views: list[RelatedSelfHealingHistoryEntryView] = []
    for record in records:
        views.append(
            RelatedSelfHealingHistoryEntryView(
                decision_id=record.decision_id,
                strategy_id=record.strategy_id,
                tenant_id=record.tenant_id,
                evidence_refs=record.evidence_refs,
                approval_refs=record.approval_refs,
                external_operation_id=record.external_operation_id,
                execution_id=record.execution_id,
                outcome=record.outcome.value,
                recorded_at=record.recorded_at,
                action_type=record.action_type,
                admission_reason=record.admission_reason,
                justification=justifications.get(record.decision_id, ""),
            ),
        )
    return tuple(views)


__all__ = ["project_self_healing_history"]
