# © Artur Czarnecki. All rights reserved.

"""R3 statistical forecasting qualification (PREDICTIVE R3)."""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta

import pytest

from intergrax.contracts.predictive_context import (
    ExecutionPatternSnapshot,
    PerformanceMetricPoint,
    PredictiveContext,
    predictive_context_from_legacy_fields,
)
from intergrax.contracts.predictive_historical_intelligence import (
    AnalyzerHistoricalReliability,
    HistoricalRiskIntelligence,
)
from intergrax.contracts.predictive_feature_set import PredictiveFeatureSet, SubjectPredictiveFeatures
from intergrax.runtime.prediction.forecasting.analyzers.latency_degradation import (
    LatencyDegradationForecastAnalyzer,
)
from intergrax.runtime.prediction.forecasting.default_feature_extractor import (
    DefaultPredictiveFeatureExtractor,
)
from intergrax.runtime.prediction.forecasting.forecast_registry import (
    PredictiveForecastAnalyzerRegistry,
)
from intergrax.runtime.prediction.forecasting.statistical_forecast_engine import (
    StatisticalForecastEngine,
)
pytestmark = [pytest.mark.unit, pytest.mark.gate]

_AS_OF = datetime(2026, 9, 11, 12, 0, tzinfo=UTC)
_SUBJECT = "svc_api"


def _base_context(
    *,
    tenant_id: str = "tenant-a",
    performance: tuple[PerformanceMetricPoint, ...] = (),
    failure: tuple[PerformanceMetricPoint, ...] = (),
    patterns: tuple[ExecutionPatternSnapshot, ...] = (),
    intelligence: HistoricalRiskIntelligence | None = None,
) -> PredictiveContext:
    return predictive_context_from_legacy_fields(
        tenant_id=tenant_id,
        current_state=(),
        historical_problems=(),
        execution_patterns=patterns,
        failure_history=failure,
        performance_history=performance,
        decision_history=(),
        lineage_patterns=(),
        input_snapshot_id="r3_test_snapshot",
        as_of=_AS_OF,
        historical_risk_intelligence=intelligence or HistoricalRiskIntelligence(),
    )


def _latency_series(values: tuple[float, ...]) -> tuple[PerformanceMetricPoint, ...]:
    base = _AS_OF - timedelta(minutes=30)
    return tuple(
        PerformanceMetricPoint(
            metric_name="latency_ms",
            value=value,
            observed_at=base + timedelta(minutes=10 * (i + 1)),
            component_id=_SUBJECT,
        )
        for i, value in enumerate(values)
    )


def _default_registry() -> PredictiveForecastAnalyzerRegistry:
    return PredictiveForecastAnalyzerRegistry.platform_default()


def test_r3_a1_latency_degradation() -> None:
    context = _base_context(
        performance=_latency_series((120.0, 180.0, 260.0)),
        patterns=(
            ExecutionPatternSnapshot(
                subject_identity=_SUBJECT,
                execution_count=100,
                failed_execution_count=5,
                avg_latency_ms=200.0,
            ),
        ),
        intelligence=HistoricalRiskIntelligence(
            analyzer_reliability=(
                AnalyzerHistoricalReliability(
                    analyzer_id="latency_degradation_forecast",
                    precision=0.88,
                    evaluated_count=100,
                ),
            ),
        ),
    )
    result = StatisticalForecastEngine(registry=_default_registry()).analyze(context)
    latency = [s for s in result.signals if s.risk_type == "LATENCY_DEGRADATION"]
    assert latency, "expected LATENCY_DEGRADATION signal"
    assert latency[0].confidence >= 0.5
    assert latency[0].prediction_window.label == "20-40 min"


def test_r3_a2_failure_acceleration() -> None:
    base = _AS_OF - timedelta(hours=3)
    failure = (
        PerformanceMetricPoint("failure_count", 5.0, base + timedelta(hours=1), _SUBJECT),
        PerformanceMetricPoint("failure_count", 20.0, base + timedelta(hours=2), _SUBJECT),
        PerformanceMetricPoint("failure_count", 60.0, base + timedelta(hours=3), _SUBJECT),
    )
    context = _base_context(
        failure=failure,
        patterns=(
            ExecutionPatternSnapshot(
                subject_identity=_SUBJECT,
                execution_count=50,
                failed_execution_count=25,
                avg_latency_ms=100.0,
            ),
        ),
    )
    result = StatisticalForecastEngine(registry=_default_registry()).analyze(context)
    accel = [s for s in result.signals if s.risk_type == "FAILURE_ACCELERATION_RISK"]
    assert accel


def test_r3_a3_retry_storm() -> None:
    base = _AS_OF - timedelta(minutes=20)
    retry = (
        PerformanceMetricPoint("retry_per_execution", 1.2, base, _SUBJECT),
        PerformanceMetricPoint("retry_per_execution", 4.8, base + timedelta(minutes=15), _SUBJECT),
    )
    context = _base_context(
        performance=retry,
        patterns=(
            ExecutionPatternSnapshot(
                subject_identity=_SUBJECT,
                execution_count=40,
                failed_execution_count=4,
                avg_latency_ms=90.0,
            ),
        ),
    )
    result = StatisticalForecastEngine(registry=_default_registry()).analyze(context)
    storm = [s for s in result.signals if s.risk_type == "RETRY_STORM_RISK"]
    assert storm


def test_r3_a4_false_positive_control_no_trend() -> None:
    context = _base_context(
        performance=_latency_series((120.0, 122.0, 121.0)),
        patterns=(
            ExecutionPatternSnapshot(
                subject_identity=_SUBJECT,
                execution_count=20,
                failed_execution_count=1,
                avg_latency_ms=121.0,
            ),
        ),
    )
    result = StatisticalForecastEngine(registry=_default_registry()).analyze(context)
    assert not result.signals


def test_r3_a5_historical_quality_influence() -> None:
    features = PredictiveFeatureSet(
        tenant_id="tenant-a",
        input_snapshot_id="snap",
        as_of=_AS_OF,
        subjects=(
            SubjectPredictiveFeatures(
                subject_identity=_SUBJECT,
                latency_growth_rate=0.8,
                data_completeness=0.98,
            ),
        ),
        global_data_completeness=0.98,
    )
    analyzer = LatencyDegradationForecastAnalyzer()
    good = analyzer.analyze(
        features,
        historical_intelligence=HistoricalRiskIntelligence(
            analyzer_reliability=(
                AnalyzerHistoricalReliability("latency_degradation_forecast", 0.95, 100),
            ),
        ),
    )
    poor = analyzer.analyze(
        features,
        historical_intelligence=HistoricalRiskIntelligence(
            analyzer_reliability=(
                AnalyzerHistoricalReliability("latency_degradation_forecast", 0.55, 100),
            ),
        ),
    )
    assert good and poor
    assert good[0].confidence > poor[0].confidence


def test_r3_a6_plugin_isolation() -> None:
    class _BoomAnalyzer:
        @property
        def descriptor(self):
            from intergrax.contracts.forecast_analyzer_descriptor import (
                ForecastAnalyzerDescriptor,
                ForecastResourceBudget,
            )

            return ForecastAnalyzerDescriptor(
                analyzer_id="boom",
                version="0",
                namespace="test",
                priority=200,
                supported_features=("latency_growth_rate",),
                resource_budget=ForecastResourceBudget(10, 100, 64),
                supported_risk_types=("BOOM",),
            )

        def analyze(self, features, *, historical_intelligence):
            raise RuntimeError("plugin fault")

    context = _base_context(
        performance=_latency_series((120.0, 180.0, 260.0)),
        patterns=(
            ExecutionPatternSnapshot(
                subject_identity=_SUBJECT,
                execution_count=30,
                failed_execution_count=2,
                avg_latency_ms=200.0,
            ),
        ),
    )
    registry = PredictiveForecastAnalyzerRegistry(
        (_BoomAnalyzer(), LatencyDegradationForecastAnalyzer()),
    )
    result = StatisticalForecastEngine(registry=registry).analyze(context)
    assert any("boom:PLUGIN_UNAVAILABLE" in o for o in result.analyzer_outcomes)
    assert any(s.risk_type == "LATENCY_DEGRADATION" for s in result.signals)


def test_r3_a7_tenant_isolation() -> None:
    from dataclasses import replace

    class _CrossTenantLeak:
        def __init__(self) -> None:
            self._inner = LatencyDegradationForecastAnalyzer()

        @property
        def descriptor(self):
            return self._inner.descriptor

        def analyze(self, features, *, historical_intelligence):
            batch = self._inner.analyze(
                features,
                historical_intelligence=historical_intelligence,
            )
            return tuple(replace(s, tenant_id="tenant-b") for s in batch)

    context = _base_context(
        tenant_id="tenant-a",
        performance=_latency_series((120.0, 180.0, 260.0)),
        patterns=(
            ExecutionPatternSnapshot(
                subject_identity=_SUBJECT,
                execution_count=30,
                failed_execution_count=2,
                avg_latency_ms=200.0,
            ),
        ),
    )
    registry = PredictiveForecastAnalyzerRegistry((_CrossTenantLeak(),))
    with pytest.raises(ValueError, match="cross-tenant"):
        StatisticalForecastEngine(registry=registry).analyze(context)


def test_r3_a8_bounded_execution() -> None:
    base = _AS_OF - timedelta(hours=2)
    points = tuple(
        PerformanceMetricPoint(
            metric_name="latency_ms",
            value=100.0 + (i % 7),
            observed_at=base + timedelta(seconds=i),
            component_id=_SUBJECT,
        )
        for i in range(1000)
    )
    context = _base_context(
        performance=points,
        patterns=(
            ExecutionPatternSnapshot(
                subject_identity=_SUBJECT,
                execution_count=1000,
                failed_execution_count=10,
                avg_latency_ms=103.0,
            ),
        ),
    )
    t0 = time.perf_counter()
    StatisticalForecastEngine(registry=_default_registry()).analyze(context)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert elapsed_ms < 500.0
