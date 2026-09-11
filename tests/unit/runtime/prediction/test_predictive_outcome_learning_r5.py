# © Artur Czarnecki. All rights reserved.

"""PREDICTIVE R5 — outcome learning loop qualification."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pytest

from intergrax.contracts.predictive.outcome.resolver import PredictiveOutcomeResolverContext
from intergrax.contracts.predictive.outcome.types import (
    PredictionOutcomeEvaluationStatus,
    PredictionOutcomeType,
)
from intergrax.contracts.predictive_risk import (
    PredictiveAnalyzerMetadata,
    PredictiveRiskScope,
    PredictiveRiskSeverity,
    PredictiveRiskSignal,
    PredictiveWindow,
)
from intergrax.runtime.prediction import LatencyTrendAnalyzer, PredictionEngine, PredictiveAnalyzerRegistry
from intergrax.runtime.prediction.governance import (
    InMemoryPredictiveAnalyzerQualityStore,
    PredictiveConfidenceCalibrator,
    PredictiveContextQualityEvaluator,
)
from intergrax.runtime.prediction.outcome import (
    EvidenceBackedPredictiveOutcomeResolver,
    InMemoryPredictionOutcomePersistence,
    PredictionOutcomeEngine,
)
from intergrax.runtime.prediction.predictive_outcome_investigation_projection import (
    project_prediction_outcome_history,
)
from tests.unit.runtime.prediction.conftest import (
    crm_agent_day2_future_evidence,
    crm_agent_incident_prevention_context,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_AS_OF = datetime(2026, 9, 11, 12, 0, tzinfo=UTC)


def _risk_signal(
    *,
    tenant_id: str = "tenant-demo",
    signal_id: str = "prsig_r5",
    confidence: float = 0.82,
    run_id: str = "prun_r5",
) -> PredictiveRiskSignal:
    return PredictiveRiskSignal(
        signal_id=signal_id,
        prediction_run_id=run_id,
        tenant_id=tenant_id,
        scope=PredictiveRiskScope.COMPONENT,
        subject_identity="crm_agent",
        risk_type="HIGH_LATENCY_RISK",
        severity=PredictiveRiskSeverity.HIGH,
        confidence=confidence,
        evidence_refs=("feature:latency", "subject:crm_agent"),
        prediction_window=PredictiveWindow(duration_seconds=3600, label="60m"),
        generated_at=_AS_OF - timedelta(minutes=60),
        analyzer_metadata=PredictiveAnalyzerMetadata(
            analyzer_id="latency_trend",
            analyzer_version="latency_trend@1.0.0",
        ),
        model_version="latency_trend@1.0.0",
        summary="Customer API latency rising",
    )


def _resolver_context(*, tenant_id: str = "tenant-demo") -> PredictiveOutcomeResolverContext:
    future = crm_agent_day2_future_evidence()
    return PredictiveOutcomeResolverContext(
        tenant_id=tenant_id,
        observed_at=future.observed_at,
        evidence_refs=future.evidence_refs + ("INC-4521",),
        execution_failed=future.execution_failed,
        problem_created_for_subject=future.problem_created_for_subject,
        matching_risk_keywords=future.matching_risk_keywords,
        incident_evidence_refs=("INC-4521",),
    )


def test_true_positive_evaluation() -> None:
    engine = PredictionOutcomeEngine(
        resolvers=(EvidenceBackedPredictiveOutcomeResolver(),),
        quality_store=InMemoryPredictiveAnalyzerQualityStore(),
    )
    result = engine.evaluate(_risk_signal(), _resolver_context())
    assert result.evaluation.outcome_type is PredictionOutcomeType.TRUE_POSITIVE
    assert result.evaluation.evaluation_status is PredictionOutcomeEvaluationStatus.EVALUATED
    assert "INC-4521" in result.evaluation.evidence_refs
    assert result.quality_profile is not None
    assert result.quality_profile.true_positive >= 1


def test_false_positive_evaluation() -> None:
    engine = PredictionOutcomeEngine(resolvers=(EvidenceBackedPredictiveOutcomeResolver(),))
    ctx = PredictiveOutcomeResolverContext(
        tenant_id="tenant-demo",
        observed_at=_AS_OF + timedelta(hours=2),
        evidence_refs=("unrelated:ref",),
        execution_failed=False,
        problem_created_for_subject=False,
        matching_risk_keywords=("other_service",),
    )
    result = engine.evaluate(_risk_signal(), ctx)
    assert result.evaluation.outcome_type is PredictionOutcomeType.FALSE_POSITIVE


def test_unknown_outcome_preserves_uncertainty() -> None:
    engine = PredictionOutcomeEngine(resolvers=(EvidenceBackedPredictiveOutcomeResolver(),))
    ctx = PredictiveOutcomeResolverContext(
        tenant_id="tenant-demo",
        observed_at=_AS_OF - timedelta(minutes=30),
        evidence_refs=(),
        execution_failed=False,
        problem_created_for_subject=False,
    )
    result = engine.evaluate(_risk_signal(), ctx)
    assert result.evaluation.outcome_type is PredictionOutcomeType.UNKNOWN
    assert result.evaluation.evaluation_status in (
        PredictionOutcomeEvaluationStatus.UNKNOWN,
        PredictionOutcomeEvaluationStatus.INSUFFICIENT_EVIDENCE,
    )


@dataclass(frozen=True, slots=True)
class _ExplodingResolver:
    resolver_id: str = "exploding"

    def evaluate(self, prediction: PredictiveRiskSignal, context: PredictiveOutcomeResolverContext):
        raise RuntimeError("boom")


def test_failed_outcome_resolver_is_contained() -> None:
    engine = PredictionOutcomeEngine(
        resolvers=(_ExplodingResolver(), EvidenceBackedPredictiveOutcomeResolver(resolver_id="fallback")),
    )
    result = engine.evaluate(_risk_signal(), _resolver_context())
    assert result.evaluation.outcome_type is PredictionOutcomeType.TRUE_POSITIVE
    assert any("exploding:failed" in o for o in result.audit.resolver_outcomes)


def test_outcome_evaluation_tenant_isolated() -> None:
    engine = PredictionOutcomeEngine(resolvers=(EvidenceBackedPredictiveOutcomeResolver(),))
    with pytest.raises(ValueError, match="tenant"):
        engine.evaluate(
            _risk_signal(tenant_id="tenant-a"),
            _resolver_context(tenant_id="tenant-b"),
        )


def test_prediction_outcome_audit_complete() -> None:
    engine = PredictionOutcomeEngine(resolvers=(EvidenceBackedPredictiveOutcomeResolver(),))
    result = engine.evaluate(_risk_signal(), _resolver_context())
    audit = result.audit
    assert audit.prediction_run_id == "prun_r5"
    assert audit.prediction_signal_id == "prsig_r5"
    assert audit.resolver_id
    assert audit.resolver_outcomes
    assert audit.evidence_refs


def test_confidence_calibration_uses_quality_profile() -> None:
    store = InMemoryPredictiveAnalyzerQualityStore()
    engine = PredictionOutcomeEngine(
        resolvers=(EvidenceBackedPredictiveOutcomeResolver(),),
        quality_store=store,
    )
    for idx in range(8):
        engine.evaluate(_risk_signal(signal_id=f"sig_{idx}"), _resolver_context())
    updated = store.get_profile(tenant_id="tenant-demo", analyzer_id="latency_trend")
    context = crm_agent_incident_prevention_context(tenant_id="tenant-demo")
    ctx_quality = PredictiveContextQualityEvaluator().evaluate(context)
    signal = _risk_signal(confidence=0.95)
    calibrated = PredictiveConfidenceCalibrator().calibrate_signal(
        signal,
        context_quality=ctx_quality,
        analyzer_profile=updated,
    )
    assert calibrated < signal.confidence


def test_prediction_without_outcome_unchanged() -> None:
    context = crm_agent_incident_prevention_context()
    engine = PredictionEngine(registry=PredictiveAnalyzerRegistry((LatencyTrendAnalyzer(),)))
    before = engine.analyze(context)
    after = engine.analyze(context)
    assert before.signals[0].risk_type == after.signals[0].risk_type
    assert before.audit.prediction_run_id != after.audit.prediction_run_id


def test_prediction_cannot_create_problem() -> None:
    assert not hasattr(PredictionOutcomeEngine, "create_problem")
    assert not hasattr(PredictionOutcomeEngine, "commit_problem")
    PredictionOutcomeEngine(
        resolvers=(EvidenceBackedPredictiveOutcomeResolver(),),
    ).evaluate(_risk_signal(), _resolver_context())


def test_read_model_prediction_outcome_history() -> None:
    engine = PredictionOutcomeEngine(resolvers=(EvidenceBackedPredictiveOutcomeResolver(),))
    evaluation = engine.evaluate(_risk_signal(), _resolver_context()).evaluation
    views = project_prediction_outcome_history(
        (evaluation,),
        precision_delta_by_signal={evaluation.prediction_signal_id: "+0.02 precision"},
    )
    assert views[0].outcome_type is PredictionOutcomeType.TRUE_POSITIVE
    assert views[0].precision_delta_label == "+0.02 precision"


def test_crm_showcase_outcome_learning_phases() -> None:
    store = InMemoryPredictiveAnalyzerQualityStore()
    engine = PredictionOutcomeEngine(
        resolvers=(EvidenceBackedPredictiveOutcomeResolver(),),
        quality_store=store,
        persistence=InMemoryPredictionOutcomePersistence(),
    )
    context = crm_agent_incident_prevention_context()
    pred_engine = PredictionEngine(
        registry=PredictiveAnalyzerRegistry((LatencyTrendAnalyzer(),)),
    )
    phase1 = pred_engine.analyze(context)
    assert phase1.signals
    phase1_conf = phase1.signals[0].confidence
    phase2 = pred_engine.analyze(context)
    assert phase2.signals[0].confidence >= phase1_conf
    signal = _risk_signal(
        signal_id=phase2.signals[0].signal_id,
        confidence=phase2.signals[0].confidence,
        run_id=phase2.audit.prediction_run_id,
    )
    outcome = engine.evaluate(signal, _resolver_context())
    assert outcome.evaluation.outcome_type is PredictionOutcomeType.TRUE_POSITIVE
    profile = store.get_profile(
        tenant_id="tenant-demo",
        analyzer_id=signal.analyzer_metadata.analyzer_id,
    )
    assert profile.predictions >= 1
