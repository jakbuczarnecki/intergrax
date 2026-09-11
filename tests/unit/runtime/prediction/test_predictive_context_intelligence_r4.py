# © Artur Czarnecki. All rights reserved.

"""PREDICTIVE R4 — predictive context intelligence layer qualification."""

from __future__ import annotations

import json
import time
from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta

import pytest

from intergrax.contracts.predictive import (
    PREDICTIVE_CONTEXT_VERSION,
    ExecutionPatternSnapshot,
    HistoricalProblemRef,
    PerformanceMetricPoint,
    PredictiveContext,
    PredictiveContextCompleteness,
    PredictiveContextDiagnostic,
    PredictiveContextHistory,
    PredictiveContextPerformance,
    PredictiveContextProviderFragment,
    PredictiveContextProviderStatus,
    PredictiveScope,
    predictive_context_from_legacy_fields,
)
from intergrax.contracts.predictive_risk import PredictiveRiskSeverity
from intergrax.runtime.prediction import LatencyTrendAnalyzer, PredictionEngine, PredictiveAnalyzerRegistry
from intergrax.runtime.prediction.context import (
    BusinessSignalProvider,
    ExecutionHistoryProvider,
    FailureHistoryProvider,
    PerformanceHistoryProvider,
    PredictiveContextAggregator,
    PredictiveContextBuilder,
)
from intergrax.runtime.prediction.predictive_investigation_projection import (
    project_related_risk_signals,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_AS_OF = datetime(2026, 9, 11, 12, 0, tzinfo=UTC)


def test_predictive_context_contract() -> None:
    scope = PredictiveScope(tenant_id="tenant-a", task_id="task-1")
    context = predictive_context_from_legacy_fields(
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        current_state=("workflow:crm",),
        historical_problems=(),
        execution_patterns=(),
        failure_history=(),
        performance_history=(),
        decision_history=(),
        lineage_patterns=(),
        input_snapshot_id="contract_snapshot",
        as_of=_AS_OF,
    )
    assert context.scope.task_id == "task-1"
    assert context.metadata.context_version == PREDICTIVE_CONTEXT_VERSION
    payload = context.to_serializable_mapping()
    json.dumps(payload)


def test_context_immutable() -> None:
    context = predictive_context_from_legacy_fields(
        tenant_id="tenant-a",
        current_state=(),
        historical_problems=(),
        execution_patterns=(),
        failure_history=(),
        performance_history=(),
        decision_history=(),
        lineage_patterns=(),
        input_snapshot_id="imm",
        as_of=_AS_OF,
    )
    with pytest.raises(FrozenInstanceError):
        context.scope = PredictiveScope(tenant_id="other")  # type: ignore[misc]


def test_context_versioning() -> None:
    context = predictive_context_from_legacy_fields(
        tenant_id="tenant-a",
        current_state=(),
        historical_problems=(),
        execution_patterns=(),
        failure_history=(),
        performance_history=(),
        decision_history=(),
        lineage_patterns=(),
        input_snapshot_id="ver",
        as_of=_AS_OF,
    )
    assert context.metadata.context_version == PREDICTIVE_CONTEXT_VERSION
    upgraded = replace(context.metadata, context_version="predictive.context@4.1.0")
    assert upgraded.context_version.startswith("predictive.context@")


def test_provider_failure_isolated() -> None:
    class _BrokenProvider:
        provider_id = "broken_provider"
        provider_version = "v1"

        def build(self, scope: PredictiveScope) -> PredictiveContextProviderFragment:
            raise RuntimeError("provider unavailable")

    class _OkProvider:
        provider_id = "execution_history"
        provider_version = "v1"

        def build(self, scope: PredictiveScope) -> PredictiveContextProviderFragment:
            return PredictiveContextProviderFragment(
                provider_id=self.provider_id,
                status=PredictiveContextProviderStatus.SUCCESS,
                history=PredictiveContextHistory(
                    execution_patterns=(
                        ExecutionPatternSnapshot(
                            subject_identity="svc",
                            execution_count=20,
                            failed_execution_count=1,
                            avg_latency_ms=120.0,
                        ),
                    ),
                ),
            )

    builder = PredictiveContextBuilder(
        aggregator=PredictiveContextAggregator(providers=(_BrokenProvider(), _OkProvider())),
    )
    context = builder.build(PredictiveScope(tenant_id="tenant-a"))
    assert context.execution_patterns
    assert "broken_provider" in context.metadata.missing_providers
    assert context.completeness is PredictiveContextCompleteness.PARTIAL


def test_timeout_does_not_break_prediction() -> None:
    class _SlowBusinessProvider:
        provider_id = "business_metrics"
        provider_version = "v1"

        def build(self, scope: PredictiveScope) -> PredictiveContextProviderFragment:
            time.sleep(0.05)
            return PredictiveContextProviderFragment(
                provider_id=self.provider_id,
                status=PredictiveContextProviderStatus.SUCCESS,
                current_state=("business:ignored",),
            )

    latency = (
        PerformanceMetricPoint(
            metric_name="latency_ms",
            value=100.0,
            observed_at=_AS_OF - timedelta(hours=2),
            component_id="crm_agent",
        ),
        PerformanceMetricPoint(
            metric_name="latency_ms",
            value=200.0,
            observed_at=_AS_OF - timedelta(hours=1),
            component_id="crm_agent",
        ),
    )
    builder = PredictiveContextBuilder(
        aggregator=PredictiveContextAggregator(
            providers=(
                ExecutionHistoryProvider(
                    history=PredictiveContextHistory(
                        execution_patterns=(
                            ExecutionPatternSnapshot(
                                subject_identity="crm_agent",
                                execution_count=50,
                                failed_execution_count=5,
                                avg_latency_ms=180.0,
                                timeout_count=4,
                            ),
                        ),
                    ),
                ),
                PerformanceHistoryProvider(
                    performance=PredictiveContextPerformance(latency_series=latency),
                ),
                _SlowBusinessProvider(),
            ),
            provider_timeout_ms=1,
        ),
    )
    engine = PredictionEngine(
        registry=PredictiveAnalyzerRegistry((LatencyTrendAnalyzer(),)),
        context_builder=builder,
    )
    result = engine.analyze_scope(PredictiveScope(tenant_id="tenant-demo"))
    assert result.signals
    context = builder.build(PredictiveScope(tenant_id="tenant-demo"))
    assert "business_metrics" in context.metadata.missing_providers
    assert context.completeness is PredictiveContextCompleteness.PARTIAL


def test_complete_context() -> None:
    builder = PredictiveContextBuilder(
        aggregator=PredictiveContextAggregator(
            providers=(
                ExecutionHistoryProvider(
                    history=PredictiveContextHistory(
                        execution_patterns=(
                            ExecutionPatternSnapshot(
                                subject_identity="svc",
                                execution_count=12,
                                failed_execution_count=0,
                                avg_latency_ms=50.0,
                            ),
                        ),
                    ),
                ),
                PerformanceHistoryProvider(
                    performance=PredictiveContextPerformance(
                        latency_series=(
                            PerformanceMetricPoint(
                                metric_name="latency_ms",
                                value=50.0,
                                observed_at=_AS_OF,
                                component_id="svc",
                            ),
                        ),
                    ),
                ),
                BusinessSignalProvider(current_state=("business:ok",)),
            ),
        ),
    )
    context = builder.build(PredictiveScope(tenant_id="tenant-a"))
    assert context.completeness is PredictiveContextCompleteness.COMPLETE
    assert context.metadata.missing_providers == ()


def test_partial_context() -> None:
    builder = PredictiveContextBuilder(
        aggregator=PredictiveContextAggregator(
            providers=(
                PerformanceHistoryProvider(
                    performance=PredictiveContextPerformance(
                        latency_series=(
                            PerformanceMetricPoint(
                                metric_name="latency_ms",
                                value=10.0,
                                observed_at=_AS_OF,
                                component_id="svc",
                            ),
                        ),
                    ),
                ),
                BusinessSignalProvider(current_state=()),
            ),
        ),
    )
    context = builder.build(PredictiveScope(tenant_id="tenant-a"))
    assert context.completeness in {
        PredictiveContextCompleteness.COMPLETE,
        PredictiveContextCompleteness.PARTIAL,
    }


def test_unavailable_context() -> None:
    class _AlwaysFail:
        provider_id = "execution_history"
        provider_version = "v1"

        def build(self, scope: PredictiveScope) -> PredictiveContextProviderFragment:
            raise RuntimeError("no data")

    context = PredictiveContextBuilder(
        aggregator=PredictiveContextAggregator(providers=(_AlwaysFail(),)),
    ).build(PredictiveScope(tenant_id="tenant-a"))
    assert context.completeness is PredictiveContextCompleteness.UNAVAILABLE


def test_context_tenant_isolation() -> None:
    builder = PredictiveContextBuilder(
        aggregator=PredictiveContextAggregator(
            providers=(
                ExecutionHistoryProvider(
                    history=PredictiveContextHistory(
                        execution_patterns=(
                            ExecutionPatternSnapshot(
                                subject_identity="crm_agent",
                                execution_count=40,
                                failed_execution_count=2,
                                avg_latency_ms=90.0,
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    ctx_a = builder.build(PredictiveScope(tenant_id="tenant-a"))
    ctx_b = builder.build(PredictiveScope(tenant_id="tenant-b"))
    assert ctx_a.tenant_id != ctx_b.tenant_id
    assert ctx_a.input_snapshot_id != ctx_b.input_snapshot_id


def test_execution_history_to_risk_signal_to_investigation_view() -> None:
    from tests.unit.runtime.prediction.conftest import crm_agent_incident_prevention_context

    engine = PredictionEngine(registry=PredictiveAnalyzerRegistry((LatencyTrendAnalyzer(),)))
    context = crm_agent_incident_prevention_context()
    result = engine.analyze(context)
    assert result.signals
    views = project_related_risk_signals(result.signals)
    assert views
    assert views[0].severity in {
        PredictiveRiskSeverity.HIGH.value,
        PredictiveRiskSeverity.MEDIUM.value,
        "HIGH",
        "MEDIUM",
    }


def test_crm_agent_incident_prevention_showcase() -> None:
    from tests.unit.runtime.prediction.conftest import crm_agent_incident_prevention_context

    context = crm_agent_incident_prevention_context()
    assert context.history.execution_patterns[0].execution_count >= 10_000
    engine = PredictionEngine(
        registry=PredictiveAnalyzerRegistry((LatencyTrendAnalyzer(),)),
    )
    signals = engine.analyze(context).signals
    high = [s for s in signals if s.severity is PredictiveRiskSeverity.HIGH]
    assert high, "expected HIGH degradation risk for CRM showcase"
    signal = high[0]
    assert signal.confidence >= 0.85
    assert "latency" in signal.summary.lower() or "degradation" in signal.summary.lower()
    assert any("execution_pattern" in ref for ref in signal.evidence_refs)
