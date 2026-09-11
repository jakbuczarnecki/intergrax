# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Orchestrate outcome resolvers, persistence, audit, and quality updates (PREDICTIVE R5)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime

from intergrax.contracts.predictive.analyzer_quality_profile import PredictiveAnalyzerQualityProfile
from intergrax.contracts.predictive.outcome.audit import PredictionOutcomeAuditRecord
from intergrax.contracts.predictive.outcome.evaluation import PredictionOutcomeEvaluation
from intergrax.contracts.predictive.outcome.resolver import (
    PredictiveOutcomeResolver,
    PredictiveOutcomeResolverContext,
)
from intergrax.contracts.predictive.outcome.types import (
    PredictionOutcomeEvaluationStatus,
    PredictionOutcomeType,
)
from intergrax.contracts.predictive_risk import PredictiveRiskSignal
from intergrax.runtime.prediction.governance.analyzer_quality_store import (
    InMemoryPredictiveAnalyzerQualityStore,
    apply_outcome_evaluation,
)
from intergrax.runtime.prediction.outcome.prediction_outcome_persistence import (
    InMemoryPredictionOutcomePersistence,
    PredictionOutcomePersistence,
)


@dataclass(frozen=True, slots=True)
class PredictionOutcomeEngineResult:
    evaluation: PredictionOutcomeEvaluation
    audit: PredictionOutcomeAuditRecord
    quality_profile: PredictiveAnalyzerQualityProfile | None


@dataclass
class PredictionOutcomeEngine:
    """
    Prediction → resolver plugins → evaluation → audit → optional quality learning.

    Does not create Problems or write diagnostic truth.
    """

    persistence: PredictionOutcomePersistence = field(
        default_factory=InMemoryPredictionOutcomePersistence,
    )
    quality_store: InMemoryPredictiveAnalyzerQualityStore | None = None
    resolvers: tuple[PredictiveOutcomeResolver, ...] = ()
    resolver_timeout_ms: int = 200
    audit_records: list[PredictionOutcomeAuditRecord] = field(default_factory=list)

    def evaluate(
        self,
        prediction: PredictiveRiskSignal,
        context: PredictiveOutcomeResolverContext,
        *,
        update_quality: bool = True,
    ) -> PredictionOutcomeEngineResult:
        if prediction.tenant_id != context.tenant_id:
            raise ValueError("tenant isolation violation: prediction/context tenant mismatch")

        ordered = sorted(self.resolvers, key=lambda r: r.resolver_id)
        resolver_outcomes: list[str] = []
        evaluation: PredictionOutcomeEvaluation | None = None

        for resolver in ordered:
            deadline = time.monotonic() + (self.resolver_timeout_ms / 1000.0)
            try:
                candidate = resolver.evaluate(prediction, context)
            except Exception as exc:
                resolver_outcomes.append(f"{resolver.resolver_id}:failed:{type(exc).__name__}")
                continue
            if time.monotonic() > deadline:
                resolver_outcomes.append(f"{resolver.resolver_id}:timeout")
                evaluation = _unknown_evaluation(
                    prediction,
                    resolver_id=resolver.resolver_id,
                    status=PredictionOutcomeEvaluationStatus.RESOLVER_TIMEOUT,
                    rationale="resolver exceeded time budget",
                )
                break
            resolver_outcomes.append(f"{resolver.resolver_id}:ok")
            if candidate.tenant_id != prediction.tenant_id:
                resolver_outcomes.append(f"{resolver.resolver_id}:tenant_rejected")
                continue
            evaluation = candidate
            break

        if evaluation is None:
            evaluation = _unknown_evaluation(
                prediction,
                resolver_id="none",
                status=PredictionOutcomeEvaluationStatus.RESOLVER_FAILED,
                rationale="no resolver produced an evaluation",
            )

        stored = self.persistence.append(evaluation)
        audit = PredictionOutcomeAuditRecord(
            prediction_run_id=stored.prediction_run_id,
            prediction_signal_id=stored.prediction_signal_id,
            tenant_id=stored.tenant_id,
            analyzer_id=stored.analyzer_id,
            outcome_type=stored.outcome_type,
            evaluation_status=stored.evaluation_status,
            evaluated_at=stored.evaluated_at,
            evidence_refs=stored.evidence_refs,
            resolver_id=stored.resolver_id,
            resolver_outcomes=tuple(resolver_outcomes),
            rationale=stored.rationale,
        )
        self.audit_records.append(audit)

        profile: PredictiveAnalyzerQualityProfile | None = None
        if (
            update_quality
            and self.quality_store is not None
            and stored.evaluation_status is PredictionOutcomeEvaluationStatus.EVALUATED
            and stored.outcome_type
            not in (PredictionOutcomeType.UNKNOWN,)
        ):
            profile = apply_outcome_evaluation(self.quality_store, stored)

        return PredictionOutcomeEngineResult(
            evaluation=stored,
            audit=audit,
            quality_profile=profile,
        )


def _unknown_evaluation(
    prediction: PredictiveRiskSignal,
    *,
    resolver_id: str,
    status: PredictionOutcomeEvaluationStatus,
    rationale: str,
) -> PredictionOutcomeEvaluation:
    return PredictionOutcomeEvaluation(
        prediction_run_id=prediction.prediction_run_id,
        prediction_signal_id=prediction.signal_id,
        tenant_id=prediction.tenant_id,
        analyzer_id=prediction.analyzer_metadata.analyzer_id,
        risk_type=prediction.risk_type,
        predicted_at=prediction.generated_at,
        outcome_type=PredictionOutcomeType.UNKNOWN,
        evaluation_status=status,
        evaluated_at=datetime.now(tz=UTC),
        evidence_refs=prediction.evidence_refs,
        rationale=rationale,
        resolver_id=resolver_id,
    )


__all__ = ["PredictionOutcomeEngine", "PredictionOutcomeEngineResult"]
