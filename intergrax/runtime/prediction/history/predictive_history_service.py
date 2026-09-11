# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Record prediction runs into history — no Problem authority (PREDICTIVE R2)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.predictive_history import (
    PredictiveHistoryOutcomeStatus,
    PredictiveRiskHistoryRecord,
)
from intergrax.contracts.predictive_risk import PredictiveRiskSignal
from intergrax.runtime.prediction.history.predictive_history_persistence import (
    PredictiveHistoryPersistence,
)
from intergrax.runtime.prediction.outcome.prediction_outcome_resolver import (
    PredictionFutureEvidenceSnapshot,
    PredictionOutcomeResolver,
)


@dataclass(slots=True)
class PredictiveHistoryService:
    persistence: PredictiveHistoryPersistence
    outcome_resolver: PredictionOutcomeResolver = PredictionOutcomeResolver()

    def record_signals(
        self,
        signals: tuple[PredictiveRiskSignal, ...],
    ) -> tuple[PredictiveRiskHistoryRecord, ...]:
        stored: list[PredictiveRiskHistoryRecord] = []
        for signal in signals:
            record = PredictiveRiskHistoryRecord(
                prediction_run_id=signal.prediction_run_id,
                risk_signal_id=signal.signal_id,
                tenant_id=signal.tenant_id,
                analyzer_id=signal.analyzer_metadata.analyzer_id,
                analyzer_version=signal.analyzer_metadata.analyzer_version,
                generated_at=signal.generated_at,
                prediction_window=signal.prediction_window,
                confidence=signal.confidence,
                evidence_refs=signal.evidence_refs,
                outcome_status=PredictiveHistoryOutcomeStatus.CREATED,
                subject_identity=signal.subject_identity,
                risk_type=signal.risk_type,
                summary=signal.summary,
            )
            stored.append(self.persistence.append(record))
        return tuple(stored)

    def observe(self, *, tenant_id: str, risk_signal_id: str) -> PredictiveRiskHistoryRecord:
        return self.persistence.update_outcome(
            tenant_id=tenant_id,
            risk_signal_id=risk_signal_id,
            outcome_status=PredictiveHistoryOutcomeStatus.OBSERVED,
        )

    def resolve_and_persist(
        self,
        record: PredictiveRiskHistoryRecord,
        future: PredictionFutureEvidenceSnapshot,
    ) -> PredictiveRiskHistoryRecord:
        if record.outcome_status is PredictiveHistoryOutcomeStatus.CREATED:
            record = self.observe(tenant_id=record.tenant_id, risk_signal_id=record.risk_signal_id)
        resolution = self.outcome_resolver.resolve(record, future)
        updated = self.persistence.update_outcome(
            tenant_id=record.tenant_id,
            risk_signal_id=record.risk_signal_id,
            outcome_status=resolution.outcome_status,
        )
        if resolution.outcome_status in (
            PredictiveHistoryOutcomeStatus.CONFIRMED,
            PredictiveHistoryOutcomeStatus.FALSE_POSITIVE,
            PredictiveHistoryOutcomeStatus.EXPIRED,
            PredictiveHistoryOutcomeStatus.UNKNOWN,
            PredictiveHistoryOutcomeStatus.INSUFFICIENT_EVIDENCE,
        ):
            return self.persistence.update_outcome(
                tenant_id=updated.tenant_id,
                risk_signal_id=updated.risk_signal_id,
                outcome_status=PredictiveHistoryOutcomeStatus.EVALUATED,
            )
        return updated


__all__ = ["PredictiveHistoryService"]
