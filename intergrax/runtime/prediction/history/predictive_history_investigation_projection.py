# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Project prediction history into operator investigation read model (PREDICTIVE R2)."""

from __future__ import annotations

from intergrax.contracts.predictive_history import PredictiveRiskHistoryRecord
from intergrax.contracts.predictive_investigation_read import RelatedPredictiveHistoryEntryView


def project_prediction_history(
    records: tuple[PredictiveRiskHistoryRecord, ...],
) -> tuple[RelatedPredictiveHistoryEntryView, ...]:
    views = tuple(
        RelatedPredictiveHistoryEntryView(
            risk_signal_id=record.risk_signal_id,
            tenant_id=record.tenant_id,
            generated_at=record.generated_at,
            risk_type=record.risk_type,
            subject_identity=record.subject_identity,
            confidence=record.confidence,
            outcome_status=record.outcome_status,
            analyzer_id=record.analyzer_id,
            summary=record.summary,
            evidence_refs=record.evidence_refs,
        )
        for record in records
    )
    return views


__all__ = ["project_prediction_history"]
