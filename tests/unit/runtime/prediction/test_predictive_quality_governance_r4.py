# © Artur Czarnecki. All rights reserved.

"""PREDICTIVE R4 — quality governance layer qualification."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from intergrax.contracts.predictive import (
    PredictiveAnalyzerQualityProfile,
    PredictiveContextCompleteness,
    PredictiveContextProviderFragment,
    PredictiveContextProviderStatus,
    PredictiveScope,
)
from intergrax.contracts.predictive.outcome_evaluation import (
    PredictionOutcomeEvaluationResult,
    build_outcome_evaluation,
)
from intergrax.contracts.predictive_risk import (
    PREDICTION_RUN_ID_ENGINE_STAMP,
    PredictiveAnalyzerMetadata,
    PredictiveRiskScope,
    PredictiveRiskSeverity,
    PredictiveRiskSignal,
    PredictiveWindow,
)
from intergrax.runtime.prediction.context import (
    BusinessSignalProvider,
    ExecutionHistoryProvider,
    PerformanceHistoryProvider,
    PredictiveContextAggregator,
    PredictiveContextBuilder,
)
from intergrax.runtime.prediction.governance import (
    InMemoryPredictiveAnalyzerQualityStore,
    PredictionGovernanceLayer,
    PredictiveContextQualityEvaluator,
    apply_outcome_evaluation,
    compose_governed_confidence,
)
from intergrax.runtime.prediction import LatencyTrendAnalyzer, PredictionEngine, PredictiveAnalyzerRegistry
from tests.unit.runtime.prediction.conftest import (
    crm_agent_incident_prevention_context,
    crm_agent_showcase_context,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_AS_OF = datetime(2026, 9, 11, 12, 0, tzinfo=UTC)


def _sample_signal(*, tenant_id: str = "tenant-a", confidence: float = 0.9) -> PredictiveRiskSignal:
    return PredictiveRiskSignal(
        signal_id="prsig_test",
        prediction_run_id=PREDICTION_RUN_ID_ENGINE_STAMP,
        tenant_id=tenant_id,
        scope=PredictiveRiskScope.COMPONENT,
        subject_identity="crm_agent",
        risk_type="LATENCY_DEGRADATION",
        severity=PredictiveRiskSeverity.HIGH,
        confidence=confidence,
        evidence_refs=("feature:latency", "subject:crm_agent"),
        prediction_window=PredictiveWindow(duration_seconds=1800, label="30m"),
        generated_at=_AS_OF,
        analyzer_metadata=PredictiveAnalyzerMetadata(
            analyzer_id="latency_degradation_forecast",
            analyzer_version="latency_degradation_forecast@1.0.0",
        ),
        model_version="latency_degradation_forecast@1.0.0",
        summary="latency increasing",
    )


def test_predictive_quality_calculation() -> None:
    governed = compose_governed_confidence(
        raw_confidence=0.95,
        context_reliability=0.8,
        analyzer_precision=0.82,
        evidence_completeness=0.85,
    )
    assert governed < 0.95
    assert abs(governed - (0.95 * 0.8 * 0.82 * 0.85)) < 1e-9


def test_context_contains_provenance() -> None:
    builder = PredictiveContextBuilder(
        aggregator=PredictiveContextAggregator(
            providers=(
                PerformanceHistoryProvider(),
                BusinessSignalProvider(current_state=("workflow:crm",)),
            ),
        ),
    )
    context = builder.build(PredictiveScope(tenant_id="tenant-a"))
    assert context.provenance
    assert any(p.source == "performance_history" for p in context.provenance)
    assert context.context_snapshot_id.startswith("csnap_")


def test_prediction_reconstruction_from_snapshot() -> None:
    context = crm_agent_incident_prevention_context()
    layer = PredictionGovernanceLayer(quality_store=InMemoryPredictiveAnalyzerQualityStore())
    snapshot = layer.snapshot_context(context)
    before = layer.context_evaluator.evaluate(context)
    after = layer.reconstruct_context_quality(snapshot)
    assert after.completeness == before.completeness
    assert after.coverage == before.coverage


def test_prediction_audit_chain_complete() -> None:
    result = PredictionEngine(
        registry=PredictiveAnalyzerRegistry((LatencyTrendAnalyzer(),)),
    ).analyze(crm_agent_showcase_context())
    audit = result.audit
    assert audit.prediction_run_id.startswith("prun_")
    assert audit.context_snapshot_id
    assert audit.analyzer_ids
    assert audit.analyzer_versions
    assert audit.quality_assessment.governed_confidence >= 0.0
    assert result.context_snapshot.snapshot_id == audit.context_snapshot_id


def test_low_quality_analyzer_reduces_confidence() -> None:
    store = InMemoryPredictiveAnalyzerQualityStore()
    store.put_profile(
        PredictiveAnalyzerQualityProfile(
            analyzer_id="latency_degradation_forecast",
            tenant_id="tenant-a",
            predictions=100,
            true_positive=30,
            false_positive=70,
            precision=0.3,
        ),
    )
    layer = PredictionGovernanceLayer(quality_store=store)
    context = crm_agent_incident_prevention_context(tenant_id="tenant-a")
    raw = _sample_signal(tenant_id="tenant-a", confidence=0.9)
    governed = layer.govern_run(
        context=context,
        prediction_run_id="prun_test",
        raw_signals=(raw,),
        analyzer_outcomes=("latency_degradation_forecast:ok:1",),
        degraded=False,
        generated_at=_AS_OF,
    )
    assert governed.signals[0].confidence < raw.confidence


def test_failed_provider_marks_context_partial() -> None:
    class _FailProvider:
        provider_id = "business_metrics"
        provider_version = "v1"

        def build(self, scope: PredictiveScope) -> PredictiveContextProviderFragment:
            raise RuntimeError("missing business metrics")

    builder = PredictiveContextBuilder(
        aggregator=PredictiveContextAggregator(
            providers=(
                ExecutionHistoryProvider(),
                PerformanceHistoryProvider(),
                _FailProvider(),
            ),
        ),
    )
    context = builder.build(PredictiveScope(tenant_id="tenant-a"))
    report = PredictiveContextQualityEvaluator().evaluate(context)
    assert context.completeness in (
        PredictiveContextCompleteness.PARTIAL,
        PredictiveContextCompleteness.LIMITED,
    )
    assert (
        "business_metrics" in report.missing_sections
        or "business_metrics" in context.metadata.missing_providers
    )


def test_prediction_quality_tenant_isolated() -> None:
    store = InMemoryPredictiveAnalyzerQualityStore()
    store.put_profile(
        PredictiveAnalyzerQualityProfile(
            analyzer_id="latency_degradation_forecast",
            tenant_id="tenant-a",
            predictions=10,
            true_positive=9,
            false_positive=1,
            precision=0.9,
        ),
    )
    store.put_profile(
        PredictiveAnalyzerQualityProfile(
            analyzer_id="latency_degradation_forecast",
            tenant_id="tenant-b",
            predictions=10,
            true_positive=2,
            false_positive=8,
            precision=0.2,
        ),
    )
    layer = PredictionGovernanceLayer(quality_store=store)
    context_a = crm_agent_incident_prevention_context(tenant_id="tenant-a")
    context_b = crm_agent_incident_prevention_context(tenant_id="tenant-b")
    signal_a = _sample_signal(tenant_id="tenant-a")
    signal_b = _sample_signal(tenant_id="tenant-b")
    out_a = layer.govern_run(
        context=context_a,
        prediction_run_id="prun_a",
        raw_signals=(signal_a,),
        analyzer_outcomes=("ok",),
        degraded=False,
    )
    out_b = layer.govern_run(
        context=context_b,
        prediction_run_id="prun_b",
        raw_signals=(signal_b,),
        analyzer_outcomes=("ok",),
        degraded=False,
    )
    assert out_a.signals[0].confidence > out_b.signals[0].confidence


def test_prediction_outcome_updates_quality() -> None:
    store = InMemoryPredictiveAnalyzerQualityStore()
    before = store.get_profile(tenant_id="tenant-a", analyzer_id="latency_trend")
    evaluation = build_outcome_evaluation(
        prediction_run_id="prun_1",
        signal_id="prsig_1",
        tenant_id="tenant-a",
        analyzer_id="latency_trend",
        risk_type="HIGH_LATENCY_RISK",
        predicted_at=_AS_OF - timedelta(hours=1),
        evaluated_at=_AS_OF,
        result=PredictionOutcomeEvaluationResult.TRUE_POSITIVE,
        rationale="incident occurred",
        evidence_refs=("INC-4521",),
    )
    after = apply_outcome_evaluation(store, evaluation)
    assert after.predictions == before.predictions + 1
    assert after.true_positive == before.true_positive + 1
