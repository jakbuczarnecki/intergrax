# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Default evidence-backed outcome resolver (PREDICTIVE R5)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from intergrax.contracts.predictive.outcome.evaluation import PredictionOutcomeEvaluation
from intergrax.contracts.predictive.outcome.resolver import PredictiveOutcomeResolverContext
from intergrax.contracts.predictive.outcome.types import (
    PredictionOutcomeEvaluationStatus,
    PredictionOutcomeType,
)
from intergrax.contracts.predictive_risk import PredictiveRiskSignal
from intergrax.runtime.prediction.outcome.prediction_outcome_resolver import (
    PredictionFutureEvidenceSnapshot,
    PredictionOutcomeResolver as HistoryOutcomeResolver,
)


@dataclass(frozen=True, slots=True)
class EvidenceBackedPredictiveOutcomeResolver:
    """SPI adapter over R2 history resolver — evidence only, no Problem mint."""

    resolver_id: str = "evidence_backed_default"
    _history: HistoryOutcomeResolver = HistoryOutcomeResolver()

    def evaluate(
        self,
        prediction: PredictiveRiskSignal,
        context: PredictiveOutcomeResolverContext,
    ) -> PredictionOutcomeEvaluation:
        if context.tenant_id != prediction.tenant_id:
            raise ValueError("tenant_id mismatch between prediction and context")

        from intergrax.contracts.predictive_history import (
            PredictiveHistoryOutcomeStatus,
            PredictiveRiskHistoryRecord,
        )

        record = PredictiveRiskHistoryRecord(
            prediction_run_id=prediction.prediction_run_id,
            risk_signal_id=prediction.signal_id,
            tenant_id=prediction.tenant_id,
            analyzer_id=prediction.analyzer_metadata.analyzer_id,
            analyzer_version=prediction.analyzer_metadata.analyzer_version,
            generated_at=prediction.generated_at,
            prediction_window=prediction.prediction_window,
            confidence=prediction.confidence,
            evidence_refs=prediction.evidence_refs,
            outcome_status=PredictiveHistoryOutcomeStatus.CREATED,
            subject_identity=prediction.subject_identity,
            risk_type=prediction.risk_type,
            summary=prediction.summary,
        )
        future = PredictionFutureEvidenceSnapshot(
            observed_at=context.observed_at,
            evidence_refs=context.evidence_refs,
            execution_failed=context.execution_failed,
            problem_created_for_subject=context.problem_created_for_subject,
            matching_risk_keywords=context.matching_risk_keywords,
        )
        resolution = self._history.resolve(record, future)
        outcome_type, status = _map_history_status(resolution.outcome_status)
        evidence = _merge_evidence(context)
        return PredictionOutcomeEvaluation(
            prediction_run_id=prediction.prediction_run_id,
            prediction_signal_id=prediction.signal_id,
            tenant_id=prediction.tenant_id,
            analyzer_id=prediction.analyzer_metadata.analyzer_id,
            risk_type=prediction.risk_type,
            predicted_at=prediction.generated_at,
            outcome_type=outcome_type,
            evaluation_status=status,
            evaluated_at=datetime.now(tz=UTC),
            evidence_refs=evidence,
            rationale=resolution.rationale,
            resolver_id=self.resolver_id,
        )


def _merge_evidence(context: PredictiveOutcomeResolverContext) -> tuple[str, ...]:
    merged = list(context.evidence_refs)
    for ref in context.incident_evidence_refs:
        if ref not in merged:
            merged.append(ref)
    return tuple(merged)


def _map_history_status(
    status: object,
) -> tuple[PredictionOutcomeType, PredictionOutcomeEvaluationStatus]:
    from intergrax.contracts.predictive_history import PredictiveHistoryOutcomeStatus

    if status is PredictiveHistoryOutcomeStatus.CONFIRMED:
        return (
            PredictionOutcomeType.TRUE_POSITIVE,
            PredictionOutcomeEvaluationStatus.EVALUATED,
        )
    if status is PredictiveHistoryOutcomeStatus.FALSE_POSITIVE:
        return (
            PredictionOutcomeType.FALSE_POSITIVE,
            PredictionOutcomeEvaluationStatus.EVALUATED,
        )
    if status is PredictiveHistoryOutcomeStatus.INSUFFICIENT_EVIDENCE:
        return (
            PredictionOutcomeType.UNKNOWN,
            PredictionOutcomeEvaluationStatus.INSUFFICIENT_EVIDENCE,
        )
    return (PredictionOutcomeType.UNKNOWN, PredictionOutcomeEvaluationStatus.UNKNOWN)


__all__ = ["EvidenceBackedPredictiveOutcomeResolver"]
