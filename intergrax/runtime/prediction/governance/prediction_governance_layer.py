# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Prediction governance layer — snapshots, quality, audit (PREDICTIVE R4)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from intergrax.contracts.predictive import PredictiveContext
from intergrax.contracts.predictive.audit import PredictionAuditRecord
from intergrax.contracts.predictive.context_quality import PredictiveContextQualityReport
from intergrax.contracts.predictive.quality import PredictiveQualityAssessment
from intergrax.contracts.predictive.snapshot import PredictiveContextSnapshot
from intergrax.contracts.predictive_risk import PredictiveRiskSignal
from intergrax.runtime.prediction.governance.analyzer_quality_store import (
    InMemoryPredictiveAnalyzerQualityStore,
)
from intergrax.runtime.prediction.governance.predictive_confidence_governance import (
    govern_risk_signal_confidence,
)
from intergrax.runtime.prediction.governance.predictive_context_quality_evaluator import (
    PredictiveContextQualityEvaluator,
)
from intergrax.runtime.prediction.governance.predictive_quality_assessment import (
    assess_prediction_quality,
)


@dataclass(frozen=True, slots=True)
class PredictionGovernanceResult:
    signals: tuple[PredictiveRiskSignal, ...]
    audit: PredictionAuditRecord
    context_snapshot: PredictiveContextSnapshot
    quality_by_signal_id: dict[str, PredictiveQualityAssessment]


@dataclass(frozen=True, slots=True)
class PredictionGovernanceLayer:
    """Wraps analyzer output with enterprise quality and audit chain."""

    quality_store: InMemoryPredictiveAnalyzerQualityStore
    context_evaluator: PredictiveContextQualityEvaluator = PredictiveContextQualityEvaluator()

    def snapshot_context(self, context: PredictiveContext) -> PredictiveContextSnapshot:
        return PredictiveContextSnapshot(
            snapshot_id=context.context_snapshot_id,
            captured_at=context.as_of,
            context=context,
        )

    def govern_run(
        self,
        *,
        context: PredictiveContext,
        prediction_run_id: str,
        raw_signals: tuple[PredictiveRiskSignal, ...],
        analyzer_outcomes: tuple[str, ...],
        degraded: bool,
        generated_at: datetime | None = None,
    ) -> PredictionGovernanceResult:
        when = generated_at or datetime.now(tz=UTC)
        snapshot = self.snapshot_context(context)
        context_quality = self.context_evaluator.evaluate(context)
        governed_signals: list[PredictiveRiskSignal] = []
        quality_by_id: dict[str, PredictiveQualityAssessment] = {}
        analyzer_ids: list[str] = []
        analyzer_versions: list[str] = []

        aggregate_quality: PredictiveQualityAssessment | None = None

        for signal in raw_signals:
            analyzer_id = signal.analyzer_metadata.analyzer_id
            profile = self.quality_store.get_profile(
                tenant_id=signal.tenant_id,
                analyzer_id=analyzer_id,
            )
            governed = govern_risk_signal_confidence(
                signal,
                context_quality=context_quality,
                analyzer_profile=profile,
            )
            assessment = assess_prediction_quality(
                context_quality=context_quality,
                analyzer_profile=profile,
                signal=governed,
            )
            governed_signals.append(governed)
            quality_by_id[governed.signal_id] = assessment
            analyzer_ids.append(analyzer_id)
            analyzer_versions.append(signal.model_version)
            aggregate_quality = assessment

        if aggregate_quality is None:
            from intergrax.contracts.predictive.quality import PredictiveQualityDimension

            aggregate_quality = PredictiveQualityAssessment(
                context_quality=PredictiveQualityDimension(
                    label="context_reliability",
                    score=context_quality.reliability,
                    rationale="no signals emitted",
                ),
                analyzer_quality=PredictiveQualityDimension(
                    label="analyzer_precision",
                    score=1.0,
                    rationale="no analyzers produced signals",
                ),
                evidence_quality=PredictiveQualityDimension(
                    label="evidence_completeness",
                    score=context_quality.coverage,
                    rationale="context-only run",
                ),
                confidence_quality=PredictiveQualityDimension(
                    label="governed_confidence",
                    score=0.0,
                    rationale="no signals",
                ),
                completeness=context_quality.completeness,
                governed_confidence=0.0,
            )

        provider_versions = tuple(
            f"{p.source}@{p.version}" for p in context.provenance
        )
        audit = PredictionAuditRecord(
            prediction_run_id=prediction_run_id,
            tenant_id=context.tenant_id,
            context_snapshot_id=snapshot.snapshot_id,
            input_snapshot_id=context.input_snapshot_id,
            generated_at=when,
            analyzer_ids=tuple(dict.fromkeys(analyzer_ids)),
            analyzer_versions=tuple(dict.fromkeys(analyzer_versions)),
            provider_versions=provider_versions,
            quality_assessment=aggregate_quality,
            signal_ids=tuple(s.signal_id for s in governed_signals),
            analyzer_outcomes=analyzer_outcomes,
            degraded=degraded,
        )
        return PredictionGovernanceResult(
            signals=tuple(governed_signals),
            audit=audit,
            context_snapshot=snapshot,
            quality_by_signal_id=quality_by_id,
        )

    def reconstruct_context_quality(
        self,
        snapshot: PredictiveContextSnapshot,
    ) -> PredictiveContextQualityReport:
        return self.context_evaluator.evaluate(snapshot.context)


__all__ = [
    "PredictionGovernanceLayer",
    "PredictionGovernanceResult",
]
