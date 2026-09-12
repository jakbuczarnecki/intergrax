# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Operator read projection for preventive action history (PREVENTIVE R7)."""

from __future__ import annotations

from intergrax.contracts.preventive.actions.audit import PreventiveActionAuditRecord
from intergrax.contracts.preventive_investigation_read import (
    RelatedPreventiveActionHistoryEntryView,
)


def project_preventive_action_history(
    records: tuple[PreventiveActionAuditRecord, ...],
) -> tuple[RelatedPreventiveActionHistoryEntryView, ...]:
    views: list[RelatedPreventiveActionHistoryEntryView] = []
    for record in records:
        views.append(
            RelatedPreventiveActionHistoryEntryView(
                proposal_id=record.proposal_id,
                tenant_id=record.tenant_id,
                action_type=record.action_type,
                risk_signal_refs=record.risk_signal_refs,
                recommendation_refs=record.recommendation_refs,
                approval_refs=record.approval_refs,
                external_operation_id=record.external_operation_id,
                execution_id=record.execution_id,
                outcome=record.outcome.value,
                recorded_at=record.recorded_at,
                admission_reason=record.admission_reason,
            ),
        )
    return tuple(views)


__all__ = ["project_preventive_action_history"]
