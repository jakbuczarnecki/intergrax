# © Artur Czarnecki. All rights reserved.

"""PREVENTIVE R6-Q — enterprise governance qualification matrix."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pytest

from intergrax.contracts.predictive_risk import (
    PredictiveAnalyzerMetadata,
    PredictiveRiskScope,
    PredictiveRiskSeverity,
    PredictiveRiskSignal,
    PredictiveWindow,
)
from intergrax.contracts.preventive.analyzer_descriptor import (
    PreventiveAnalyzerDescriptor,
    PreventiveResourceBudget,
)
from intergrax.contracts.preventive.category import PreventiveRecommendationCategory
from intergrax.contracts.preventive.conflict import CONFLICTING_RECOMMENDATIONS
from intergrax.contracts.preventive.context import (
    DiagnosticEvidenceContext,
    HistoricalOutcome,
    PreventiveAnalysisInput,
)
from intergrax.contracts.preventive.evidence import RecommendationEvidenceReference
from intergrax.contracts.preventive.lifecycle import (
    PreventiveRecommendationLifecycleState,
    assert_lifecycle_transition,
)
from intergrax.contracts.preventive.recommendation import PreventiveRecommendationCandidate
from intergrax.contracts.preventive.safety import PreventiveSafetyAssessment
from intergrax.runtime.prediction import LatencyTrendAnalyzer, PredictionEngine, PredictiveAnalyzerRegistry
from intergrax.runtime.prevention import (
    CrmLatencyPreventiveAnalyzer,
    PreventiveAnalyzerRegistry,
    PreventiveIntelligenceEngine,
    PreventiveOutcomeEngine,
    project_preventive_recommendations,
)
from intergrax.contracts.preventive.outcome_evaluation import (
    OperatorRecommendationDecision,
    RecommendationEffectiveness,
    RecommendationOutcomeEvaluation,
)
from tests.unit.runtime.prediction.conftest import crm_agent_incident_prevention_context

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_AS_OF = datetime(2026, 9, 11, 12, 0, tzinfo=UTC)


def _descriptor(analyzer_id: str) -> PreventiveAnalyzerDescriptor:
    return PreventiveAnalyzerDescriptor(
        analyzer_id=analyzer_id,
        namespace="test.preventive",
        version=f"{analyzer_id}@1.0.0",
        owner="test-owner",
        capabilities=("test",),
        quality_profile_id=analyzer_id,
        resource_budget=PreventiveResourceBudget(
            max_execution_time_ms=100,
            max_candidates_per_run=1,
        ),
    )


def _risk_signal(tenant_id: str = "tenant-demo") -> PredictiveRiskSignal:
    return PredictiveRiskSignal(
        signal_id="prsig_gov",
        prediction_run_id="prun_gov",
        tenant_id=tenant_id,
        scope=PredictiveRiskScope.COMPONENT,
        subject_identity="crm_agent",
        risk_type="HIGH_LATENCY_RISK",
        severity=PredictiveRiskSeverity.HIGH,
        confidence=0.82,
        evidence_refs=("feature:latency",),
        prediction_window=PredictiveWindow(duration_seconds=3600, label="60m"),
        generated_at=_AS_OF - timedelta(minutes=60),
        analyzer_metadata=PredictiveAnalyzerMetadata(
            analyzer_id="latency_trend",
            analyzer_version="latency_trend@1.0.0",
        ),
        model_version="latency_trend@1.0.0",
        summary="Payment connector degradation",
    )


def _analysis_input(tenant_id: str = "tenant-demo") -> PreventiveAnalysisInput:
    return PreventiveAnalysisInput(
        predictive_context=crm_agent_incident_prevention_context(tenant_id=tenant_id),
        risk_signal=_risk_signal(tenant_id=tenant_id),
        historical_outcome=HistoricalOutcome(
            tenant_id=tenant_id,
            historical_success_rate=0.7,
            similar_incident_refs=("INC-4521", "INC-4400", "INC-4301", "INC-4200"),
        ),
        diagnostic_evidence=DiagnosticEvidenceContext(
            tenant_id=tenant_id,
            evidence_refs=("latency_trend:+240%", "INC-4521"),
        ),
    )


def test_execution_is_always_disabled() -> None:
    with pytest.raises(ValueError, match="execution_allowed"):
        PreventiveSafetyAssessment(
            execution_allowed=True,
            requires_human_review=True,
            risk_level="HIGH",
            governance_status="PENDING",
        )
    engine = PreventiveIntelligenceEngine(
        registry=PreventiveAnalyzerRegistry((CrmLatencyPreventiveAnalyzer(),)),
    )
    rec = engine.recommend(_analysis_input()).recommendations[0]
    assert rec.safety.execution_allowed is False
    assert rec.governance.execution_allowed is False


def test_recommendation_lifecycle_is_monotonic() -> None:
    assert_lifecycle_transition(
        PreventiveRecommendationLifecycleState.GENERATED,
        PreventiveRecommendationLifecycleState.VALIDATED,
    )
    assert_lifecycle_transition(
        PreventiveRecommendationLifecycleState.VALIDATED,
        PreventiveRecommendationLifecycleState.PRESENTED,
    )
    assert_lifecycle_transition(
        PreventiveRecommendationLifecycleState.PRESENTED,
        PreventiveRecommendationLifecycleState.ACCEPTED,
    )
    assert_lifecycle_transition(
        PreventiveRecommendationLifecycleState.ACCEPTED,
        PreventiveRecommendationLifecycleState.EVALUATED,
    )
    with pytest.raises(ValueError, match="illegal lifecycle"):
        assert_lifecycle_transition(
            PreventiveRecommendationLifecycleState.GENERATED,
            PreventiveRecommendationLifecycleState.EVALUATED,
        )


def test_recommendation_requires_evidence() -> None:
    with pytest.raises(ValueError, match="evidence_refs"):
        PreventiveRecommendationCandidate(
            category=PreventiveRecommendationCategory.OBSERVE,
            description="Observe",
            expected_impact="low",
            evidence_refs=(),
            analyzer_id="x",
            analyzer_version="x@1",
        )


def test_recommendation_is_reconstructable() -> None:
    engine = PreventiveIntelligenceEngine(
        registry=PreventiveAnalyzerRegistry((CrmLatencyPreventiveAnalyzer(),)),
    )
    result = engine.recommend(_analysis_input())
    rec = result.recommendations[0]
    audit = result.recommendation_audit[0]
    assert audit.recommendation_id == rec.recommendation_id
    assert audit.prediction_signal_id == rec.risk_signal_id
    assert audit.context_snapshot_id == rec.context_snapshot_id
    assert audit.analyzer_id == rec.analyzer_id
    assert audit.evidence_refs
    assert rec.reasoning_summary
    assert rec.known_limitations


@dataclass
class _CacheIncreaseAnalyzer:
    analyzer_namespace = "test.preventive"
    analyzer_id = "cache_increase"
    analyzer_version = "cache_increase@1.0.0"
    priority = 50

    @property
    def descriptor(self) -> PreventiveAnalyzerDescriptor:
        return _descriptor(self.analyzer_id)

    def analyze(self, analysis_input: PreventiveAnalysisInput) -> tuple[PreventiveRecommendationCandidate, ...]:
        ref = RecommendationEvidenceReference(
            source_type="risk_signal",
            source_id=analysis_input.risk_signal.signal_id,
            relation="supports",
        )
        return (
            PreventiveRecommendationCandidate(
                category=PreventiveRecommendationCategory.CONFIGURATION_REVIEW,
                description="Increase cache size for CRM connector",
                expected_impact="buffer spikes",
                evidence_refs=(ref,),
                analyzer_id=self.analyzer_id,
                analyzer_version=self.analyzer_version,
                reasoning_summary="Historical spikes suggest cache pressure",
                known_limitations=("no live cache metrics",),
            ),
        )


@dataclass
class _CacheReduceAnalyzer:
    analyzer_namespace = "test.preventive"
    analyzer_id = "cache_reduce"
    analyzer_version = "cache_reduce@1.0.0"
    priority = 40

    @property
    def descriptor(self) -> PreventiveAnalyzerDescriptor:
        return _descriptor(self.analyzer_id)

    def analyze(self, analysis_input: PreventiveAnalysisInput) -> tuple[PreventiveRecommendationCandidate, ...]:
        ref = RecommendationEvidenceReference(
            source_type="risk_signal",
            source_id=analysis_input.risk_signal.signal_id,
            relation="supports",
        )
        return (
            PreventiveRecommendationCandidate(
                category=PreventiveRecommendationCategory.CONFIGURATION_REVIEW,
                description="Reduce cache size to limit stale CRM payloads",
                expected_impact="fresher payloads",
                evidence_refs=(ref,),
                analyzer_id=self.analyzer_id,
                analyzer_version=self.analyzer_version,
                reasoning_summary="Stale cache correlated with incidents",
                known_limitations=("no live cache metrics",),
            ),
        )


def test_conflicting_recommendations_are_marked() -> None:
    engine = PreventiveIntelligenceEngine(
        registry=PreventiveAnalyzerRegistry(
            (_CacheIncreaseAnalyzer(), _CacheReduceAnalyzer()),
        ),
    )
    result = engine.recommend(_analysis_input())
    assert len(result.recommendations) == 2
    assert result.conflicts
    assert result.conflicts[0].marker == CONFLICTING_RECOMMENDATIONS


@dataclass
class _BrokenPreventiveAnalyzer:
    analyzer_namespace = "test.preventive"
    analyzer_id = "broken"
    analyzer_version = "broken@1.0.0"
    priority = 1

    @property
    def descriptor(self) -> PreventiveAnalyzerDescriptor:
        return _descriptor(self.analyzer_id)

    def analyze(self, analysis_input: PreventiveAnalysisInput) -> tuple[PreventiveRecommendationCandidate, ...]:
        raise RuntimeError("broken")


def test_failed_preventive_analyzer_isolated() -> None:
    engine = PreventiveIntelligenceEngine(
        registry=PreventiveAnalyzerRegistry(
            (_BrokenPreventiveAnalyzer(), CrmLatencyPreventiveAnalyzer()),
        ),
    )
    result = engine.recommend(_analysis_input())
    assert result.recommendations
    assert any("broken:PLUGIN_UNAVAILABLE" in item for item in result.audit.analyzer_outcomes)


def test_recommendation_cannot_cross_tenant() -> None:
    tenant_a = PreventiveIntelligenceEngine(
        registry=PreventiveAnalyzerRegistry((CrmLatencyPreventiveAnalyzer(),)),
    ).recommend(_analysis_input(tenant_id="tenant-a"))
    tenant_b = PreventiveIntelligenceEngine(
        registry=PreventiveAnalyzerRegistry((CrmLatencyPreventiveAnalyzer(),)),
    ).recommend(_analysis_input(tenant_id="tenant-b"))
    assert tenant_a.recommendations[0].tenant_id == "tenant-a"
    assert tenant_b.recommendations[0].tenant_id == "tenant-b"
    a_refs = {ref.source_id for ref in tenant_a.recommendations[0].evidence_refs}
    b_refs = {ref.source_id for ref in tenant_b.recommendations[0].evidence_refs}
    assert a_refs.isdisjoint({item for item in b_refs if item.startswith("tenant-b")})


def test_preventive_tenant_isolation() -> None:
    with pytest.raises(ValueError, match="tenant isolation"):
        PreventiveAnalysisInput(
            predictive_context=crm_agent_incident_prevention_context(tenant_id="tenant-a"),
            risk_signal=_risk_signal(tenant_id="tenant-b"),
            historical_outcome=HistoricalOutcome(tenant_id="tenant-a", historical_success_rate=0.5),
            diagnostic_evidence=DiagnosticEvidenceContext(tenant_id="tenant-a", evidence_refs=("x",)),
        )


def test_prediction_without_prevention_behavior_unchanged() -> None:
    context = crm_agent_incident_prevention_context()
    registry = PredictiveAnalyzerRegistry((LatencyTrendAnalyzer(),))
    baseline = PredictionEngine(registry=registry).analyze(context)
    again = PredictionEngine(registry=registry).analyze(context)
    assert len(baseline.signals) == len(again.signals)
    assert baseline.audit.degraded == again.audit.degraded


def test_crm_enterprise_qualification_timeline() -> None:
    engine = PreventiveIntelligenceEngine(
        registry=PreventiveAnalyzerRegistry((CrmLatencyPreventiveAnalyzer(),)),
    )
    result = engine.recommend(_analysis_input())
    rec = result.recommendations[0]
    assert 0.0 < rec.confidence <= 1.0
    assert "connector" in rec.description.lower() or "latency" in rec.description.lower()
    views = project_preventive_recommendations(result.recommendations, risk_type="HIGH_LATENCY_RISK")
    assert views[0].lifecycle_state == PreventiveRecommendationLifecycleState.PRESENTED.value

    outcome = PreventiveOutcomeEngine()
    outcome.record(
        RecommendationOutcomeEvaluation(
            recommendation_id=rec.recommendation_id,
            tenant_id=rec.tenant_id,
            analyzer_id=rec.analyzer_id,
            operator_decision=OperatorRecommendationDecision.ACCEPTED,
            effectiveness=RecommendationEffectiveness.TRUE_PREVENTION,
            evidence_refs=("outcome:incident_avoided",),
            evaluated_at=_AS_OF,
        ),
    )
    assert outcome.store.evaluations[0].effectiveness is RecommendationEffectiveness.TRUE_PREVENTION
